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

#  ----------------->
BASE_DIR = Path(__file__).resolve().parent.parent

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
        device=DEVICE,
    )
else:
    print("⚠️  Modelo fine-tuned ES no encontrado, FT deshabilitado.")
    emotion_es_ft = None

#  ----------------->


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
print(f"Cargando modelo Whisper: {WHISPER_MODEL_NAME}")
asr_model = whisper.load_model(WHISPER_MODEL_NAME)

# Emociones en texto en inglés (baseline, HuggingFace)
print("Cargando modelo de emociones en texto (inglés)...")
emotion_en_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=DEVICE,
)

# Emociones en audio (SER)
print("Cargando modelo de emociones en audio (SER)...")
audio_emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=DEVICE,
)

# Sentimiento multilingüe
print("Cargando modelo de sentimiento multilingüe...")
sentiment_classifier = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    return_all_scores=True,
    device=DEVICE,
)

# Emociones en texto español (pysentimiento)
print("Inicializando pysentimiento (emociones ES)...")
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
