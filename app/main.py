from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os
from transformers import pipeline
from collections import defaultdict

app = FastAPI(title="Whisper Transcription API")

model = whisper.load_model("small")

#Modelo base, emociones en texto 
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

#Modelo para extraer emociones directamente del audio
audio_emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# Sentimiento multilingüe
sentiment_classifier = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis"
)

#Returns 7 emotions:
#[neutral, fear, anger, sadness, joy, surprise, disgust]
@app.post("/transcribe/emotion")
async def transcribe_emotion(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Transcribir con Whisper (ya devuelve segmentos)
        result = model.transcribe(tmp_path)
        text = result["text"]
        segments = result.get("segments", [])

        segment_emotions = []
        # Para calcular emociones "globales" promediadas
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            # Clasificar emociones en este segmento
            em_result = emotion_classifier(seg_text)  # [[{label, score}, ...]]
            raw_emotions = em_result[0]

            emotions = [
                {"label": e["label"], "score": float(e["score"])}
                for e in raw_emotions
            ]

            # Top emoción del segmento
            top_emotion = max(emotions, key=lambda x: x["score"])

            # Peso del segmento para el promedio global
            # Puedes usar duración (end - start) o longitud del texto
            duration = float(seg.get("end", 0) - seg.get("start", 0)) or 1.0
            weight = duration
            total_weight += weight

            for e in emotions:
                global_scores[e["label"]] += e["score"] * weight

            segment_emotions.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg_text,
                "top_emotion": top_emotion,
                "emotions": emotions,
            })

        # Calcular emociones globales promediadas
        global_emotions = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_emotions.append({
                    "label": label,
                    "score": float(score_sum / total_weight)
                })

        # Ordenar globales
        global_emotions_sorted = sorted(global_emotions, key=lambda x: x["score"], reverse=True)
        top_global_emotions = global_emotions_sorted[:3]

        return {
            "transcription": text,
            "global_emotions": global_emotions_sorted,
            "top_global_emotions": top_global_emotions,
            "segments": segment_emotions
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/audio-emotions/")
async def audio_emotions(file: UploadFile = File(...)):
    """
    Retorna emociones detectadas directamente de la voz,
    usando superb/wav2vec2-base-superb-er.
    """
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # top_k=None → devuelve todas las etiquetas del modelo
        results = audio_emotion_classifier(tmp_path, top_k=None)

        emotions = [
            {"label": r["label"], "score": float(r["score"])}
            for r in results
        ]

        # Ordenar de mayor a menor score
        emotions_sorted = sorted(emotions, key=lambda x: x["score"], reverse=True)
        top_emotion = emotions_sorted[0] if emotions_sorted else None

        return {
            "emotions": emotions_sorted,
            "top_emotion": top_emotion
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



#Return 5 sentiments:
#[negative, positive, neutral, very negative, very positive]
@app.post("/transcribe/sentiment")
async def transcribe_sentiment(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Transcribir con Whisper (ya devuelve segmentos)
        result = model.transcribe(tmp_path)
        text = result["text"]
        segments = result.get("segments", [])

        segment_sentiments = []
        # Para calcular sentimientos "globales" promediados
        global_scores = defaultdict(float)
        total_weight = 0.0

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            # Clasificar SENTIMIENTO en este segmento
            # top_k=None devuelve todas las clases (1–5 stars)
            sent_result = sentiment_classifier(seg_text, top_k=None)

            sentiments = [
                {"label": r["label"], "score": float(r["score"])}
                for r in sent_result
            ]
            if not sentiments:
                continue

            # Top sentimiento del segmento
            top_sentiment = max(sentiments, key=lambda x: x["score"])

            # Peso del segmento para el promedio global
            duration = float(seg.get("end", 0) - seg.get("start", 0)) or 1.0
            weight = duration
            total_weight += weight

            for s in sentiments:
                global_scores[s["label"]] += s["score"] * weight

            segment_sentiments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg_text,
                "top_sentiment": top_sentiment,
                "sentiments": sentiments,
            })

        # Calcular sentimientos globales promediados
        global_sentiments = []
        if total_weight > 0:
            for label, score_sum in global_scores.items():
                global_sentiments.append({
                    "label": label,
                    "score": float(score_sum / total_weight)
                })

        # Ordenar globales
        global_sentiments_sorted = sorted(global_sentiments, key=lambda x: x["score"], reverse=True)
        top_global_sentiments = global_sentiments_sorted[:3]

        return {
            "transcription": text,
            "global_sentiments": global_sentiments_sorted,
            "top_global_sentiments": top_global_sentiments,
            "segments": segment_sentiments
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
