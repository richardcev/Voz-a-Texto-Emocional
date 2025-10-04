from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os

app = FastAPI(title="Whisper Transcription API")

# Cargamos el modelo una sola vez (medium)
model = whisper.load_model("medium")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcribir
    result = model.transcribe(tmp_path)
    text = result["text"]

    # Eliminar archivo temporal
    os.remove(tmp_path)

    return {"transcription": text}
