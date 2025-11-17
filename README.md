# 🎙️ Whisper Transcription & Emotion API

API REST construida con **FastAPI** que permite:

- Transcribir audio usando **OpenAI Whisper**
- Analizar **emociones** del texto (inglés y español)
- Analizar **sentimiento** (positivo/negativo, multilingüe)
- Analizar **emociones de la voz** directamente desde el audio

---

## 🚀 Requisitos

- Python 3.9+ (recomendado)
- `ffmpeg` instalado en el sistema (necesario para Whisper)

### Dependencias Python

```bash
pip install -r requirements.txt
````

> **Nota:** En algunos entornos, para `torch` conviene seguir las instrucciones oficiales de instalación de PyTorch según tu sistema y GPU.

### Instalar `ffmpeg`

* **Ubuntu/Debian**

  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

* **macOS (Homebrew)**

  ```bash
  brew install ffmpeg
  ```

---

## ⚙️ Configuración del modelo Whisper

Por defecto se usa el modelo:

```python
model_name = os.getenv("WHISPER_MODEL", "base")
```

Puedes cambiarlo con la variable de entorno `WHISPER_MODEL`:

* Linux / macOS:

  ```bash
  export WHISPER_MODEL=small
  ```

* Windows (PowerShell):

  ```powershell
  set WHISPER_MODEL=small
  ```

Modelos posibles: `tiny`, `base`, `small`, `medium`, `large`, etc.

---

## ▶️ Cómo levantar la API

Guarda el código en un archivo, por ejemplo:

```bash
main.py
```

Luego, ejecuta:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API quedará disponible en:

* Base URL: `http://127.0.0.1:8000`
* Documentación interactiva (Swagger): `http://127.0.0.1:8000/docs`

Desde `/docs` puedes probar los endpoints subiendo archivos `.wav` de forma sencilla.

---

## 📁 Endpoints

Todos los endpoints:

* Son `POST`
* Reciben un archivo de audio en el campo `file` (formato `multipart/form-data`)
* Devuelven JSON

### 1. `POST /transcribe/emotion`

Analiza **emociones en inglés a partir del texto** transcrito.

* Transcribe el audio con Whisper.
* Usa el modelo `j-hartmann/emotion-english-distilroberta-base`.
* Calcula:

  * Emociones por segmento (con tiempos `start` / `end`).
  * Emociones globales promediadas (ponderadas por la duración de cada segmento).

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/emotion" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mi_audio.wav"
```

#### Response (ejemplo simplificado)

```json
{
  "transcription": "This is an example text...",
  "global_emotions": [
    {"label": "joy", "score": 0.73},
    {"label": "sadness", "score": 0.12}
  ],
  "top_global_emotions": [
    {"label": "joy", "score": 0.73}
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 3.1,
      "text": "First segment...",
      "top_emotion": {"label": "joy", "score": 0.9},
      "emotions": [
        {"label": "joy", "score": 0.9},
        {"label": "sadness", "score": 0.05},
        {"label": "anger", "score": 0.03}
      ]
    }
  ]
}
```

📌 **Uso recomendado**: audios en **inglés** donde interese el contenido del texto y sus emociones.

---

### 2. `POST /audio-emotions/`

Analiza **emociones directamente desde la voz**, sin transcribir el texto.

* No usa Whisper.
* Usa el modelo `superb/wav2vec2-base-superb-er`.
* Trabaja sobre el archivo de audio crudo.

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/audio-emotions/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mi_audio.wav"
```

#### Response (ejemplo simplificado)

```json
{
  "emotions": [
    {"label": "angry", "score": 0.65},
    {"label": "sad", "score": 0.20},
    {"label": "neutral", "score": 0.10}
  ],
  "top_emotion": {
    "label": "angry",
    "score": 0.65
  }
}
```

📌 **Uso recomendado**: cuando te interesa **cómo se dice** (tono, voz) más que **qué se dice**.

---

### 3. `POST /transcribe/sentiment`

Analiza el **sentimiento del texto** (positivo / negativo / neutral, etc.) a partir del audio.

* Transcribe con Whisper.
* Usa el modelo multilingüe:

  * `tabularisai/multilingual-sentiment-analysis`
* Devuelve:

  * Sentimientos por segmento
  * Sentimientos globales ponderados

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/sentiment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mi_audio.wav"
```

#### Response (ejemplo simplificado)

```json
{
  "transcription": "Este es un ejemplo de audio...",
  "global_sentiments": [
    {"label": "positive", "score": 0.70},
    {"label": "neutral", "score": 0.20},
    {"label": "negative", "score": 0.10}
  ],
  "top_global_sentiments": [
    {"label": "positive", "score": 0.70}
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 4.0,
      "text": "Primer segmento...",
      "top_sentiment": {"label": "positive", "score": 0.9},
      "sentiments": [
        {"label": "positive", "score": 0.9},
        {"label": "neutral", "score": 0.08},
        {"label": "negative", "score": 0.02}
      ]
    }
  ]
}
```

📌 **Uso recomendado**: cuando necesites saber si el mensaje es **positivo, negativo o neutro** de forma general.

---

### 4. `POST /transcribe/pysentimiento-emotion-es`

Analiza **emociones en español** a partir del texto transcrito.

* Transcribe con Whisper.

* Usa `pysentimiento` con:

  ```python
  create_analyzer(task="emotion", lang="es")
  ```

* Devuelve emociones por segmento y globales, con etiquetas en español (`alegría`, `tristeza`, etc.).

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/pysentimiento-emotion-es" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mi_audio.wav"
```

#### Response (ejemplo simplificado)

```json
{
  "transcription": "Este es un ejemplo en español...",
  "global_emotions": [
    {"label": "alegría", "score": 0.60},
    {"label": "tristeza", "score": 0.15}
  ],
  "top_global_emotions": [
    {"label": "alegría", "score": 0.60}
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Primer segmento...",
      "top_emotion": {"label": "alegría", "score": 0.8},
      "emotions": [
        {"label": "alegría", "score": 0.8},
        {"label": "tristeza", "score": 0.1}
      ]
    }
  ]
}
```

📌 **Uso recomendado**: audios en **español**, donde el foco sea el contenido emocional del texto.

---

## 🔗 Independencia entre endpoints

* Cada endpoint funciona **de forma independiente**.
* Todos reciben un archivo de audio y devuelven su propio análisis.
* Comparten internamente:

  * El modelo de Whisper (para los endpoints `/transcribe/...`).
  * Los modelos/pipelines cargados al inicio del script.
* No es necesario llamar uno antes que otro; puedes usar solo el endpoint que necesites según el caso.

---

## 📌 Resumen de uso

| Endpoint                                    | Usa Whisper | Tipo de análisis                          | Idioma principal |
| ------------------------------------------- | ----------- | ----------------------------------------- | ---------------- |
| `POST /transcribe/emotion`                  | ✅           | Emociones desde el **texto** (inglés)     | Inglés           |
| `POST /audio-emotions/`                     | ❌           | Emociones desde la **voz**                | Independiente    |
| `POST /transcribe/sentiment`                | ✅           | **Sentimiento** (positivo/negativo, etc.) | Multilingüe      |
| `POST /transcribe/pysentimiento-emotion-es` | ✅           | Emociones desde el **texto** (español)    | Español          |

---

## 🧪 Pruebas rápidas

1. Levanta la API:

   ```bash
   uvicorn main:app --reload
   ```

2. Abre en el navegador:

   ```text
   http://127.0.0.1:8000/docs
   ```

3. Selecciona un endpoint → **Try it out** → sube un `.wav` → **Execute**.

---
