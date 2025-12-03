from fastapi import FastAPI, UploadFile, File
import os, tempfile
from transformers import pipeline
from collections import defaultdict
from pysentimiento import create_analyzer
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from googletrans import Translator
import json

load_dotenv()

app = FastAPI(title="Whisper Transcription API")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Emotion classification (EN)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Audio emotion classifier (NO WHISPER, requires wav2vec2)
audio_emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# Multilingual sentiment
sentiment_classifier = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis"
)

# Pysentimiento ES emotions
pysentimiento_emotion_es = create_analyzer(task="emotion", lang="es")


translator = Translator() 


async def openai_transcribe_audio(tmp_path: str):
    """
    Transcribe audio usando el modelo Whisper-1 de OpenAI.
    Devuelve un objeto con el texto completo y los segmentos detallados.
    """
    with open(tmp_path, "rb") as audio_file:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript
    
async def translate_es_to_en(text: str) -> str:
    """Traduce texto ES -> EN usando googletrans."""
    # Según tu ejemplo, estás usando la versión async de googletrans
    result = await translator.translate(text, dest="en")
    return result.text



@app.post("/transcribe/emotion")
async def transcribe_emotion(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = await openai_transcribe_audio(tmp_path)
        text = transcript.text
        segments = transcript.segments or []

        segment_emotions = []
        global_scores = defaultdict(float)
        total_weight = 0

        for seg in segments:
            print(seg)
            # accesos por atributo, no .get()
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            raw = emotion_classifier(seg_text)[0]
            emotions = [{"label": e["label"], "score": float(e["score"])} for e in raw]
            top = max(emotions, key=lambda x: x["score"])

            start = float(seg.start or 0)
            end = float(seg.end or 0)
            duration = (end - start) or 1.0
            total_weight += duration

            for e in emotions:
                global_scores[e["label"]] += e["score"] * duration

            segment_emotions.append({
                "start": start,
                "end": end,
                "text": seg_text,
                "top_emotion": top,
                "emotions": emotions
            })


        global_emotions = [
            {"label": lbl, "score": float(score / total_weight)}
            for lbl, score in global_scores.items()
        ]
        global_emotions_sorted = sorted(global_emotions, key=lambda x: x["score"], reverse=True)

        return {
            "transcription": text,
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": global_emotions_sorted[:3],
            "segments": segment_emotions
        }

    finally:
        os.remove(tmp_path)

async def analyze_emotions_openai(text: str):
    """
    Analiza el texto usando GPT-4o-mini para simular la salida del modelo de Hugging Face.
    Devuelve las 7 emociones con sus scores.
    """
    system_prompt = (
        "You are an emotion classification model. Analyze the text provided and "
        "assign a confidence score (between 0.0 and 1.0) for EXACTLY these 7 emotions: "
        "[neutral, fear, anger, sadness, joy, surprise, disgust]. "
        "The sum of scores should approximate 1.0. "
        "Return ONLY a JSON object with this exact structure: "
        "{ 'emotions': [ {'label': 'neutral', 'score': 0.1}, {'label': 'fear', 'score': 0.0}, ... ] }"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Modelo rápido y barato
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to analyze: '{text}'"}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("emotions", [])
        
    except Exception as e:
        print(f"Error calling OpenAI for emotion analysis: {e}")
        # Fallback en caso de error para no romper el flujo
        return [{"label": "neutral", "score": 1.0}] + [
            {"label": e, "score": 0.0} for e in ["fear", "anger", "sadness", "joy", "surprise", "disgust"]
        ]

@app.post("/transcribe/emotion/openai")
async def transcribe_emotion(file: UploadFile = File(...)):
    # 1. Guardar archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2. Transcribir con OpenAI (Whisper API)
        transcript = await openai_transcribe_audio(tmp_path)
        
        # La API devuelve un objeto, accedemos a sus atributos
        text = transcript.text
        segments = transcript.segments or []

        segment_emotions = []
        global_scores = defaultdict(float)
        total_weight = 0.0

        # 3. Procesar segmentos
        for seg in segments:
            # En la respuesta de la API verbose_json, 'seg' es un objeto, usamos notación punto
            seg_text = (seg["text"] if isinstance(seg, dict) else seg.text).strip()
            
            if not seg_text:
                continue

            # --- AQUÍ ESTÁ EL CAMBIO PRINCIPAL ---
            # En lugar de usar emotion_classifier(local), llamamos a OpenAI
            raw_emotions = await analyze_emotions_openai(seg_text)

            # Convertir a formato seguro
            emotions = [
                {"label": e["label"], "score": float(e["score"])} 
                for e in raw_emotions
            ]
            
            # Encontrar la emoción dominante
            top_emotion = max(emotions, key=lambda x: x["score"]) if emotions else {"label": "neutral", "score": 0.0}

            # Obtener tiempos para el peso
            # La API de OpenAI devuelve start/end en el objeto de segmento
            start = float(seg["start"] if isinstance(seg, dict) else seg.start)
            end = float(seg["end"] if isinstance(seg, dict) else seg.end)
            
            duration = (end - start) or 1.0
            total_weight += duration

            # Sumar al score global ponderado
            for e in emotions:
                global_scores[e["label"]] += e["score"] * duration

            segment_emotions.append({
                "start": start,
                "end": end,
                "text": seg_text,
                "top_emotion": top_emotion,
                "emotions": emotions
            })

        # 4. Calcular promedios globales
        global_emotions = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_emotions.append({
                    "label": label,
                    "score": float(score_sum / total_weight)
                })
        
        # Ordenar resultados globales
        global_emotions_sorted = sorted(global_emotions, key=lambda x: x["score"], reverse=True)

        return {
            "transcription": text,
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": global_emotions_sorted[:3],
            "segments": segment_emotions
        }

    except Exception as e:
        # Captura errores generales (API key inválida, fichero corrupto, etc.)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Limpieza del archivo temporal
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/transcribe/emotion/es")
async def transcribe_emotion_es(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente (usa la extensión que quieras, aquí .wav)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1) Transcribir audio con OpenAI
        transcript = await openai_transcribe_audio(tmp_path)
        text = transcript.text  # texto completo
        segments = transcript.segments or []  # lista de TranscriptionSegment

        segment_emotions = []
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            # Texto original (español, en tu caso)
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            # 2) Traducir segmento ES -> EN
            translated_text = await translate_es_to_en(seg_text)

            # 3) Pasar el texto traducido al modelo de emociones EN
            raw = emotion_classifier(translated_text)[0]
            emotions = [{"label": e["label"], "score": float(e["score"])} for e in raw]
            top = max(emotions, key=lambda x: x["score"])

            # 4) Duración del segmento para ponderar emociones globales
            start = float(seg.start or 0.0)
            end = float(seg.end or 0.0)
            duration = (end - start) or 1.0
            total_weight += duration

            for e in emotions:
                global_scores[e["label"]] += e["score"] * duration

            # 5) Guardamos la emoción, pero con el texto original en ES
            segment_emotions.append({
                "start": start,
                "end": end,
                "text": seg_text,               # 👈 texto original (español)
                "translated_text": translated_text,  # opcional, por si quieres verlo
                "top_emotion": top,
                "emotions": emotions
            })

        # 6) Emociones globales ponderadas
        global_emotions = []
        if total_weight > 0:
            for lbl, score in global_scores.items():
                global_emotions.append({
                    "label": lbl,
                    "score": float(score / total_weight)
                })

        global_emotions_sorted = sorted(global_emotions, key=lambda x: x["score"], reverse=True)

        return {
            "transcription": text,  # transcripción completa (en ES)
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": global_emotions_sorted[:3],
            "segments": segment_emotions
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



@app.post("/audio-emotions")
async def audio_emotions(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = audio_emotion_classifier(tmp_path, top_k=None)
        emotions = [{"label": r["label"], "score": float(r["score"])} for r in results]
        emotions_sorted = sorted(emotions, key=lambda x: x["score"], reverse=True)

        return {"emotions": emotions_sorted, "top_emotion": emotions_sorted[0]}

    finally:
        os.remove(tmp_path)


@app.post("/transcribe/sentiment")
async def transcribe_sentiment(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = await openai_transcribe_audio(tmp_path)
        text = transcript.text
        segments = transcript.segments or []

        segment_results = []
        global_scores = defaultdict(float)
        total_weight = 0

        for seg in segments:
            # CORRECCIÓN: Acceso por atributo en lugar de .get()
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            # Clasificación de sentimiento
            sent_raw = sentiment_classifier(seg_text, top_k=None)
            sentiments = [{"label": r["label"], "score": float(r["score"])} for r in sent_raw]
            top_sentiment = max(sentiments, key=lambda x: x["score"])

            # CORRECCIÓN: Acceso por atributo para tiempos
            start = float(seg.start or 0)
            end = float(seg.end or 0)
            duration = (end - start) or 1.0
            total_weight += duration

            for s in sentiments:
                global_scores[s["label"]] += s["score"] * duration

            segment_results.append({
                "start": start,
                "end": end,
                "text": seg_text,
                "top_sentiment": top_sentiment,
                "sentiments": sentiments
            })

        global_sentiments = [
            {"label": lbl, "score": float(score / total_weight)}
            for lbl, score in global_scores.items()
        ]
        global_sorted = sorted(global_sentiments, key=lambda x: x["score"], reverse=True)

        return {
            "transcription": text,
            "global_sentiments": global_sorted,
            "top_global_sentiments": global_sorted[:3],
            "segments": segment_results
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/transcribe/pysentimiento-es")
async def transcribe_pysentimiento_es(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = await openai_transcribe_audio(tmp_path)
        text = transcript.text
        segments = transcript.segments or []

        segment_results = []
        total_weight = 0
        global_scores = defaultdict(float)

        for seg in segments:
            # CORRECCIÓN: Acceso por atributo en lugar de .get()
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            # Predicción con pysentimiento
            emo = pysentimiento_emotion_es.predict(seg_text)
            emotions = [{"label": k, "score": float(v)} for k, v in emo.probas.items()]
            top_emo = max(emotions, key=lambda x: x["score"])

            # CORRECCIÓN: Acceso por atributo para tiempos
            start = float(seg.start or 0)
            end = float(seg.end or 0)
            duration = (end - start) or 1.0
            total_weight += duration

            for e in emotions:
                global_scores[e["label"]] += e["score"] * duration

            segment_results.append({
                "start": start,
                "end": end,
                "text": seg_text,
                "top_emotion": top_emo,
                "emotions": emotions
            })

        global_emotions = [
            {"label": lbl, "score": float(score / total_weight)}
            for lbl, score in global_scores.items()
        ]
        global_sorted = sorted(global_emotions, key=lambda x: x["score"], reverse=True)

        return {
            "transcription": text,
            "global_emotions": global_sorted,
            "top_global_emotions": global_sorted[:3],
            "segments": segment_results
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)