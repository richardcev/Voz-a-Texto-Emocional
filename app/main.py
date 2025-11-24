import os
import tempfile
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import whisper
import torch
from transformers import pipeline
from pysentimiento import create_analyzer


WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL", "small"
)  # "tiny", "base", "small", etc.

DEVICE = 0 if torch.cuda.is_available() else -1

# --------------
import re

USE_FASTER = os.getenv("USE_FASTER_WHISPER", "0") == "1"
FASTER_MODEL = os.getenv("FASTER_WHISPER_MODEL", os.getenv("WHISPER_MODEL", "small"))
# --------------


# ---- Flags/Devices controlados por ENV (cambios nuevos) ----
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" o "cuda"
TEXT_ON_CPU = os.getenv("TEXT_ON_CPU", "1") == "1"  # si 1 => pipelines de TEXTO en CPU
FORCE_PYSENTIMIENTO_CPU = os.getenv("FORCE_PYSENTIMIENTO_CPU", "1") == "1"
TEXT_DEVICE = -1 if TEXT_ON_CPU else (0 if torch.cuda.is_available() else -1)

#  ----------------->
BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_MASTER_DIR = (
    BASE_DIR
    / "models"
    / (
        "emotion_es_master_3c"
        if os.getenv("REDUCE_TO_3", "0") == "1"
        else "emotion_es_master_5c"
    )
)
AUDIO_MASTER_DIR = BASE_DIR / "models" / "audio_emotion_master"


FT_EMOTION_MODEL_PATH = os.getenv(
    "FT_EMOTION_MODEL_PATH", str(BASE_DIR / "models" / "emotion_es_bert_colombia")
)

if os.path.isdir(FT_EMOTION_MODEL_PATH):
    print(f"Cargando modelo fine-tuned de emociones ES desde: {FT_EMOTION_MODEL_PATH}")
    emotion_es_ft = pipeline(
        "text-classification",
        model=FT_EMOTION_MODEL_PATH,
        tokenizer=FT_EMOTION_MODEL_PATH,
        return_all_scores=True,
        device=TEXT_DEVICE,  # <- antes: DEVICE
    )
else:
    print("⚠️  Modelo fine-tuned ES no encontrado, FT deshabilitado.")
    emotion_es_ft = None

#  ----------------->


#  =====================>
# Carga texto master
if TEXT_MASTER_DIR.is_dir():
    print(f"Cargando modelo MASTER de emociones (texto) desde: {TEXT_MASTER_DIR}")
    emotion_es_master = pipeline(
        "text-classification",
        model=str(TEXT_MASTER_DIR),
        tokenizer=str(TEXT_MASTER_DIR),
        return_all_scores=True,
        device=TEXT_DEVICE,  # <- antes: DEVICE
    )
else:
    print("⚠️  Modelo MASTER texto ES no encontrado.")
    emotion_es_master = None

# Carga audio master
if AUDIO_MASTER_DIR.is_dir():
    print(f"Cargando modelo MASTER de emociones (audio) desde: {AUDIO_MASTER_DIR}")
    audio_emotion_master = pipeline(
        "audio-classification",
        model=str(AUDIO_MASTER_DIR),
        feature_extractor=str(AUDIO_MASTER_DIR),
        device=DEVICE,
        top_k=None,
    )
else:
    print("⚠️  Modelo MASTER audio no encontrado.")
    audio_emotion_master = None

#  <=====================

# -----------------
# --- NUEVAS ENV Y LOAD PARA PRO v2 ---
import json, torch
from transformers import pipeline

# Ruta del PRO v2
TEXT_PRO_V2_DIR = (
    BASE_DIR
    / "models"
    / (
        "text_emotion_pro_v2_3c"
        if os.getenv("REDUCE_TO_3", "0") == "1"
        else "text_emotion_pro_v2_5c"
    )
)
TEXT_PRO_V2_ON_CPU = bool(int(os.getenv("TEXT_PRO_V2_ON_CPU", "0")))
TEXT_PRO_V2_DEVICE = (
    -1 if TEXT_PRO_V2_ON_CPU else (0 if torch.cuda.is_available() else -1)
)


def _load_text_pro_v2():
    if TEXT_PRO_V2_DIR.exists():
        print(f"Cargando modelo PRO v2 texto desde: {TEXT_PRO_V2_DIR}")
        clf = pipeline(
            "text-classification",
            model=str(TEXT_PRO_V2_DIR),
            tokenizer=str(TEXT_PRO_V2_DIR),
            return_all_scores=True,
            device=TEXT_PRO_V2_DEVICE,
        )
        # umbrales + temperatura
        try:
            with open(TEXT_PRO_V2_DIR / "thresholds.json") as f:
                th = json.load(f)
        except Exception:
            th = None
        try:
            with open(TEXT_PRO_V2_DIR / "temperature.json") as f:
                temp = json.load(f).get("temperature", 1.0)
        except Exception:
            temp = 1.0
        return clf, th, float(temp)
    else:
        print("⚠️ Modelo PRO v2 texto no encontrado.")
        return None, None, 1.0


text_pro_v2, THRESH_PRO_V2, TEMP_PRO_V2 = _load_text_pro_v2()


def _apply_thresholds(scores, thresholds):
    if not thresholds:
        return max(scores, key=lambda x: x["score"])
    ok = [s for s in scores if s["score"] >= thresholds.get(s["label"], 0.5)]
    return (
        max(ok, key=lambda x: x["score"])
        if ok
        else max(scores, key=lambda x: x["score"])
    )


def _apply_temperature(scores, temperature: float):
    # scores: [{'label':..., 'score':...}] con score ~ softmax
    # re-aplicar temperatura t: p_i := softmax(log(p_i)/t)  (aprox)
    import math

    if abs(temperature - 1.0) < 1e-6:
        return scores
    # evitar p=0
    eps = 1e-12
    logps = [math.log(max(s["score"], eps)) for s in scores]
    scaled = [math.exp(lp / temperature) for lp in logps]
    Z = sum(scaled) + eps
    return [
        {"label": s["label"], "score": float(v / Z)} for s, v in zip(scores, scaled)
    ]


# ------------------
print(
    f"Cargando modelo Whisper: {WHISPER_MODEL_NAME} en {'cuda' if torch.cuda.is_available() and os.getenv('WHISPER_DEVICE','cuda')!='cpu' else 'cpu'}"
)

asr_backend = "whisper"
asr_model = None
faster_model = None

if USE_FASTER:
    try:
        from faster_whisper import WhisperModel

        device = (
            "cuda"
            if (
                torch.cuda.is_available()
                and os.getenv("WHISPER_DEVICE", "cuda") != "cpu"
            )
            else "cpu"
        )
        compute_type = os.getenv(
            "FASTER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8"
        )
        faster_model = WhisperModel(
            FASTER_MODEL, device=device, compute_type=compute_type
        )
        asr_backend = "faster-whisper"
        print(f"Usando Faster-Whisper ({FASTER_MODEL}) [{device}/{compute_type}]")
    except Exception as e:
        print(f"⚠️ No pude cargar Faster-Whisper: {e}. Voy con whisper estándar.")
        asr_backend = "whisper"


def split_words_with_timestamps(text: str, start: float, end: float):
    """
    Fallback si no hay word timestamps: reparte el tiempo del segmento
    proporcional al número de palabras (se conserva orden).
    """
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    words = cleaned.split(" ")
    dur = max(end - start, 1e-3)
    step = dur / max(len(words), 1)
    out, t = [], start
    for w in words:
        w_start = t
        w_end = min(end, t + step)
        out.append({"text": w, "start": float(w_start), "end": float(w_end)})
        t = w_end
    return out


# --------------


# --- FIN NUEVAS ENV Y LOAD PARA PRO v2 ---


app = FastAPI(
    title="Sistema Voz a Texto con Análisis Emocional",
    description="API para transcripción (Whisper) y detección de emociones/sentimiento en español/inglés.",
    version="1.0.0",
)

# CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # control in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ### Load models once at startup ==================================
print(f"Cargando modelo Whisper: {WHISPER_MODEL_NAME} en {WHISPER_DEVICE}")
if asr_backend == "whisper":
    asr_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
else:
    # Mantener Whisper estándar en CPU para los endpoints que lo usan,
    # mientras Faster-Whisper queda en GPU para karaoke/word-timestamps.
    asr_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")

# Emociones en texto en inglés (baseline, HuggingFace)
print("Cargando modelo de emociones en texto (inglés)...")
emotion_en_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=TEXT_DEVICE,  # <- antes: DEVICE
)

# Emociones en audio (SER)
print("Cargando modelo de emociones en audio (SER) en CPU…")
audio_emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=-1,  # CPU fijo para no tocar VRAM
)

# Sentimiento multilingüe
print("Cargando modelo de sentimiento multilingüe...")
sentiment_classifier = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    return_all_scores=True,
    device=TEXT_DEVICE,  # <- antes: DEVICE
)

# Emociones en texto español (pysentimiento)
print("Inicializando pysentimiento (emociones ES)...")
if FORCE_PYSENTIMIENTO_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # aísla en CPU
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
pysentimiento_emotion_es = create_analyzer(task="emotion", lang="es")


# ### Helpers ======================================================
def save_temp_file(upload_file: UploadFile, suffix: str = "") -> str:
    """Guarda el UploadFile en un archivo temporal y devuelve la ruta."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            return tmp.name
    finally:
        upload_file.file.close()


def delete_temp_file(path: str):
    if path and os.path.exists(path):
        os.remove(path)


def ensure_audio(file: UploadFile):
    if not file.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="El archivo debe ser de audio.")


# ### Endpoints ==================================================
# TRANSCRIPTION + EMOTIONS / SENTIMENT -------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe un audio (idealmente en español) usando Whisper.
    Devuelve texto completo y segmentos con tiempos.
    """
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        result = asr_model.transcribe(tmp_path, language="es")  # fuerza español
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_list = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
        ]

        return {
            "transcription": text,
            "segments": segment_list,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en transcripción: {str(e)}")
    finally:
        delete_temp_file(tmp_path)


# TRANSCRIPTION + EMOTIONS (EN) ---------------
@app.post("/transcribe/emotion-en")
async def transcribe_emotion_en(file: UploadFile = File(...)):
    """
    Transcribe el audio y analiza emociones del TEXTO usando
    un modelo de emociones en inglés (HuggingFace).
    Útil si el audio está en inglés.
    """
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        result = asr_model.transcribe(tmp_path)  # idioma auto
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_emotions = []
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            em_result = emotion_en_classifier(seg_text, top_k=None)[0]
            emotions = [
                {"label": e["label"], "score": float(e["score"])} for e in em_result
            ]

            if not emotions:
                continue

            top_emotion = max(emotions, key=lambda x: x["score"])

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            duration = max(end - start, 0.1)
            weight = duration
            total_weight += weight

            for e in emotions:
                global_scores[e["label"]] += e["score"] * weight

            segment_emotions.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "top_emotion": top_emotion,
                    "emotions": emotions,
                }
            )

        global_emotions = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_emotions.append(
                    {"label": label, "score": float(score_sum / total_weight)}
                )

        global_emotions_sorted = sorted(
            global_emotions, key=lambda x: x["score"], reverse=True
        )
        top_global_emotions = global_emotions_sorted[:3]

        return {
            "transcription": text,
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": top_global_emotions,
            "segments": segment_emotions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en emociones (texto-EN): {str(e)}"
        )
    finally:
        delete_temp_file(tmp_path)


# Emotions from audio (SER) --------------
@app.post("/audio-emotions")
async def audio_emotions(file: UploadFile = File(...)):
    """
    Analiza emociones directamente desde la voz (tono) usando
    superb/wav2vec2-base-superb-er.
    Devuelve lista de emociones ordenadas por score.
    """
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        results = audio_emotion_classifier(tmp_path, top_k=None)

        emotions = [{"label": r["label"], "score": float(r["score"])} for r in results]

        emotions_sorted = sorted(emotions, key=lambda x: x["score"], reverse=True)
        top_emotion = emotions_sorted[0] if emotions_sorted else None

        return {
            "emotions": emotions_sorted,
            "top_emotion": top_emotion,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en emociones de audio: {str(e)}"
        )
    finally:
        delete_temp_file(tmp_path)


# TRANSCRIPTION + SENTIMENT Multilingüe ---------------
@app.post("/transcribe/sentiment")
async def transcribe_sentiment(file: UploadFile = File(...)):
    """
    Transcribe el audio y analiza SENTIMIENTO (positivo / negativo / neutro, etc.)
    usando un modelo multilingüe (tabularisai).
    """
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        result = asr_model.transcribe(tmp_path)  # idioma auto
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_sentiments = []
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            sent_result = sentiment_classifier(seg_text, top_k=None)[0]
            sentiments = [
                {"label": s["label"], "score": float(s["score"])} for s in sent_result
            ]

            if not sentiments:
                continue

            top_sentiment = max(sentiments, key=lambda x: x["score"])

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            duration = max(end - start, 0.1)
            weight = duration
            total_weight += weight

            for s in sentiments:
                global_scores[s["label"]] += s["score"] * weight

            segment_sentiments.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "top_sentiment": top_sentiment,
                    "sentiments": sentiments,
                }
            )

        global_sentiments = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_sentiments.append(
                    {"label": label, "score": float(score_sum / total_weight)}
                )

        global_sentiments_sorted = sorted(
            global_sentiments, key=lambda x: x["score"], reverse=True
        )
        top_global_sentiments = global_sentiments_sorted[:3]

        return {
            "transcription": text,
            "global_sentiments": global_sentiments_sorted,
            "top_global_sentiments": top_global_sentiments,
            "segments": segment_sentiments,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en análisis de sentimiento: {str(e)}"
        )
    finally:
        delete_temp_file(tmp_path)


# TRANSCRIPTION + EMOTIONS (ES, pysentimiento) ---------------
@app.post("/transcribe/pysentimiento-emotion-es")
async def transcribe_pysentimiento_emotion_es(file: UploadFile = File(...)):
    """
    Transcribe el audio (español) y analiza emociones en el TEXTO
    usando pysentimiento (modelo entrenado para ES).
    """
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        result = asr_model.transcribe(tmp_path, language="es")
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_emotions = []
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            # pysentimiento
            emotion_result = pysentimiento_emotion_es.predict(seg_text)
            # emotion_result.probas es un dict: { 'alegría': 0.3, 'tristeza': 0.1, ... }
            emotions = [
                {"label": label, "score": float(score)}
                for label, score in emotion_result.probas.items()
            ]

            if not emotions:
                continue

            top_emotion = max(emotions, key=lambda x: x["score"])

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            duration = max(end - start, 0.1)
            weight = duration
            total_weight += weight

            for e in emotions:
                global_scores[e["label"]] += e["score"] * weight  # fix

            segment_emotions.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "top_emotion": top_emotion,
                    "emotions": emotions,
                }
            )

        global_emotions = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_emotions.append(
                    {"label": label, "score": float(score_sum / total_weight)}
                )

        global_emotions_sorted = sorted(
            global_emotions, key=lambda x: x["score"], reverse=True
        )
        top_global_emotions = global_emotions_sorted[:3]

        return {
            "transcription": text,
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": top_global_emotions,
            "segments": segment_emotions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en emociones (pysentimiento ES): {str(e)}",
        )
    finally:
        delete_temp_file(tmp_path)


# #### FAINE-TUNED ================================================================
# TRANSCRIPTION + EMOTIONS (ES, modelo fine-tuned) ---------------
@app.post("/transcribe/emotion-es-ft")
async def transcribe_emotion_es_ft(file: UploadFile = File(...)):
    """
    Transcribe el audio (español) y analiza emociones en el TEXTO
    usando el modelo BERT fine-tuned en pysentimiento/emociones_colombia.
    """
    if emotion_es_ft is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Modelo fine-tuned de emociones ES no disponible. "
                "Asegúrate de haber entrenado y guardado en models/emotion_es_bert_colombia."
            ),
        )

    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        result = asr_model.transcribe(tmp_path, language="es")
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_emotions = []
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            # Emociones con modelo fine-tuned
            em_result = emotion_es_ft(seg_text, top_k=None)
            emotions = [
                {"label": e["label"], "score": float(e["score"])} for e in em_result
            ]

            if not emotions:
                continue

            top_emotion = max(emotions, key=lambda x: x["score"])

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            duration = max(end - start, 0.1)
            weight = duration
            total_weight += weight

            for e in emotions:
                global_scores[e["label"]] += e["score"] * weight

            segment_emotions.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "top_emotion": top_emotion,
                    "emotions": emotions,
                }
            )

        global_emotions = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_emotions.append(
                    {"label": label, "score": float(score_sum / total_weight)}
                )

        global_emotions_sorted = sorted(
            global_emotions, key=lambda x: x["score"], reverse=True
        )
        top_global_emotions = global_emotions_sorted[:3]

        return {
            "transcription": text,
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": top_global_emotions,
            "segments": segment_emotions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en emociones (ES fine-tuned): {str(e)}",
        )
    finally:
        delete_temp_file(tmp_path)


@app.post("/transcribe/emotion-es-master")
async def transcribe_emotion_es_master(file: UploadFile = File(...)):
    if emotion_es_master is None:
        raise HTTPException(
            status_code=500, detail="Modelo MASTER texto ES no disponible."
        )
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")
    try:
        result = asr_model.transcribe(tmp_path, language="es")
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_emotions, global_scores, total_weight = [], defaultdict(float), 0.0
        for s in segments:
            seg_text = s.get("text", "").strip()
            if not seg_text:
                continue
            em_result = emotion_es_master(seg_text, top_k=None)
            emotions = [
                {"label": d["label"], "score": float(d["score"])} for d in em_result
            ]
            start, end = float(s.get("start", 0.0)), float(s.get("end", 0.0))
            dur = max(end - start, 0.1)
            total_weight += dur
            for item in emotions:
                global_scores[item["label"]] += item["score"] * dur
            segment_emotions.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "top_emotion": max(emotions, key=lambda x: x["score"]),
                    "emotions": emotions,
                }
            )
        global_emotions = (
            [
                {"label": k, "score": float(v / total_weight)}
                for k, v in global_scores.items()
            ]
            if total_weight > 0
            else []
        )
        global_emotions.sort(key=lambda x: x["score"], reverse=True)
        return {
            "transcription": text,
            "global_emotions": global_emotions,
            "top_global_emotions": global_emotions[:3],
            "segments": segment_emotions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MASTER texto ES error: {str(e)}")
    finally:
        delete_temp_file(tmp_path)


@app.post("/transcribe/emotion-text-pro-v2")
async def transcribe_emotion_text_pro_v2(file: UploadFile = File(...)):
    """
    Transcribe en ES (Whisper) y etiqueta emociones con el modelo PRO v2 (multi-corpus)
    usando:
      - Temperature scaling (calibración)
      - Umbral por clase (PR)
    """
    if text_pro_v2 is None:
        raise HTTPException(
            status_code=500, detail="Modelo PRO v2 de texto no disponible."
        )

    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")
    try:
        # Transcribir en español
        result = asr_model.transcribe(tmp_path, language="es")
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        segment_out, global_acc = [], {}
        total_w = 0.0

        for s in segments:
            seg_text = s.get("text", "").strip()
            if not seg_text:
                continue
            raw = text_pro_v2(seg_text, top_k=None)[0]  # [{'label':..., 'score':...}]
            # Calibrar temperaturas
            cal = _apply_temperature(raw, TEMP_PRO_V2)
            top = _apply_thresholds(cal, THRESH_PRO_V2)

            st, en = float(s.get("start", 0.0)), float(s.get("end", 0.0))
            dur = max(en - st, 0.1)
            total_w += dur
            for item in cal:
                global_acc[item["label"]] = (
                    global_acc.get(item["label"], 0.0) + item["score"] * dur
                )

            segment_out.append(
                {
                    "start": st,
                    "end": en,
                    "text": seg_text,
                    "top_emotion": top,
                    "emotions": cal,
                }
            )

        global_scores = (
            [{"label": k, "score": float(v / total_w)} for k, v in global_acc.items()]
            if total_w > 0
            else []
        )
        global_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "transcription": text,
            "global_emotions": global_scores,
            "top_global_emotions": global_scores[:3],
            "segments": segment_out,
            "thresholds_used": THRESH_PRO_V2,
            "temperature_used": TEMP_PRO_V2,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PRO v2 text error: {str(e)}")
    finally:
        delete_temp_file(tmp_path)


# /transcribe/karaoke ==============================
@app.post("/transcribe/karaoke")
async def transcribe_karaoke(file: UploadFile = File(...)):
    """
    Devuelve transcripción + segmentos + palabras con timestamps.
    - Si USE_FASTER_WHISPER=1 -> word timestamps reales (faster-whisper)
    - Si no -> fallback aproximado repartiendo tiempo entre palabras.
    """
    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")
    try:
        if asr_backend == "faster-whisper":
            # Faster-Whisper: timestamps a nivel de palabra
            segments, info = faster_model.transcribe(
                tmp_path,
                language="es",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200},
                word_timestamps=True,
            )
            all_segments = []
            global_text = []
            for seg in segments:
                seg_text = seg.text.strip()
                global_text.append(seg_text)
                words = []
                if seg.words:
                    for w in seg.words:
                        words.append(
                            {
                                "text": w.word.strip(),
                                "start": (
                                    float(w.start)
                                    if w.start is not None
                                    else float(seg.start)
                                ),
                                "end": (
                                    float(w.end)
                                    if w.end is not None
                                    else float(seg.end)
                                ),
                            }
                        )
                else:
                    words = split_words_with_timestamps(
                        seg_text, float(seg.start), float(seg.end)
                    )

                all_segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg_text,
                        "words": words,
                    }
                )

            return {
                "backend": "faster-whisper",
                "transcription": " ".join(global_text).strip(),
                "segments": all_segments,
                "duration": (
                    float(info.duration)
                    if hasattr(info, "duration") and info.duration
                    else None
                ),
                "language": info.language if hasattr(info, "language") else "es",
            }

        else:
            # Whisper estándar: sin word timestamps nativos -> fallback
            result = asr_model.transcribe(tmp_path, language="es")
            text = result.get("text", "").strip()
            segments = result.get("segments", []) or []
            all_segments = []
            for s in segments:
                st = float(s.get("start", 0.0))
                en = float(s.get("end", 0.0))
                seg_text = (s.get("text") or "").strip()
                words = split_words_with_timestamps(seg_text, st, en)
                all_segments.append(
                    {
                        "start": st,
                        "end": en,
                        "text": seg_text,
                        "words": words,
                    }
                )

            return {
                "backend": "whisper",
                "transcription": text,
                "segments": all_segments,
                "duration": None,
                "language": "es",
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Karaoke error: {str(e)}")
    finally:
        delete_temp_file(tmp_path)


# =========================================================
# TRANSCRIPTION + KARAOKE + EMOCIONES (ES MASTER) =========
# =========================================================
@app.post("/transcribe/karaoke-emotion-es-master")
async def transcribe_karaoke_emotion_es_master(file: UploadFile = File(...)):
    """
    Devuelve transcripción + segmentos con palabras (timestamps) + emociones por segmento
    usando el modelo MASTER de emociones en TEXTO (ES).
    - Si USE_FASTER_WHISPER=1 -> word timestamps reales (faster-whisper)
    - Si no -> fallback aproximado repartiendo tiempo entre palabras.
    """
    if emotion_es_master is None:
        raise HTTPException(
            status_code=500, detail="Modelo MASTER texto ES no disponible."
        )

    ensure_audio(file)
    tmp_path = save_temp_file(file, suffix=".wav")

    try:
        all_segments = []
        global_text_parts = []
        duration_total = None
        language_detected = "es"

        # ---------- ASR con palabras ----------
        if asr_backend == "faster-whisper":
            segments, info = faster_model.transcribe(
                tmp_path,
                language="es",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200},
                word_timestamps=True,
            )
            duration_total = (
                float(info.duration)
                if hasattr(info, "duration") and info.duration
                else None
            )
            language_detected = info.language if hasattr(info, "language") else "es"

            for seg in segments:
                seg_text = (seg.text or "").strip()
                global_text_parts.append(seg_text)
                words = []
                if seg.words:
                    for w in seg.words:
                        words.append(
                            {
                                "text": (w.word or "").strip(),
                                "start": (
                                    float(w.start)
                                    if w.start is not None
                                    else float(seg.start)
                                ),
                                "end": (
                                    float(w.end)
                                    if w.end is not None
                                    else float(seg.end)
                                ),
                            }
                        )
                else:
                    words = split_words_with_timestamps(
                        seg_text, float(seg.start), float(seg.end)
                    )

                all_segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg_text,
                        "words": words,
                    }
                )

        else:
            # Whisper estándar (sin palabras nativas) -> fallback
            result = asr_model.transcribe(tmp_path, language="es")
            language_detected = "es"
            segs = result.get("segments", []) or []
            for s in segs:
                st = float(s.get("start", 0.0))
                en = float(s.get("end", 0.0))
                seg_text = (s.get("text") or "").strip()
                global_text_parts.append(seg_text)
                words = split_words_with_timestamps(seg_text, st, en)
                all_segments.append(
                    {
                        "start": st,
                        "end": en,
                        "text": seg_text,
                        "words": words,
                    }
                )

        transcription_full = " ".join([t for t in global_text_parts if t]).strip()

        # ---------- Emociones por segmento + agregación global ----------
        global_scores = defaultdict(float)
        total_weight = 0.0
        enriched_segments = []

        for s in all_segments:
            seg_text = s["text"]
            if not seg_text:
                enriched_segments.append({**s, "top_emotion": None, "emotions": []})
                continue

            em_result = emotion_es_master(seg_text, top_k=None)
            emotions = [
                {"label": d["label"], "score": float(d["score"])} for d in em_result
            ]
            top_emotion = max(emotions, key=lambda x: x["score"]) if emotions else None

            st, en = float(s["start"]), float(s["end"])
            dur = max(en - st, 0.1)
            total_weight += dur
            for item in emotions:
                global_scores[item["label"]] += item["score"] * dur

            enriched_segments.append(
                {
                    **s,
                    "top_emotion": top_emotion,
                    "emotions": emotions,
                }
            )

        global_emotions = (
            [
                {"label": lab, "score": float(score_sum / total_weight)}
                for lab, score_sum in global_scores.items()
            ]
            if total_weight > 0
            else []
        )
        global_emotions.sort(key=lambda x: x["score"], reverse=True)

        return {
            "backend": (
                "faster-whisper" if asr_backend == "faster-whisper" else "whisper"
            ),
            "transcription": transcription_full,
            "segments": enriched_segments,
            "global_emotions": global_emotions,
            "top_global_emotions": global_emotions[:3],
            "duration": duration_total,
            "language": language_detected,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Karaoke+Emotions error: {str(e)}")
    finally:
        delete_temp_file(tmp_path)
