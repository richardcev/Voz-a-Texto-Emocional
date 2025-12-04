# 🧠 Sistema de Análisis Emocional y Transcripción de Voz (Thesis Project)

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green) ![OpenAI](https://img.shields.io/badge/AI-Whisper%20%2B%20GPT4o-orange) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow) ![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

Este repositorio contiene el backend de una plataforma de análisis de audio desarrollada como proyecto de titulación. El sistema es capaz de transcribir audio, traducir contenido y detectar matices emocionales utilizando una arquitectura híbrida que combina modelos **SOTA (State-of-the-Art)** de código abierto y servicios en la nube.

## 📋 Descripción del Proyecto

El objetivo principal es ir más allá de la simple transcripción de voz a texto (STT), proporcionando una capa de inteligencia emocional. El sistema procesa archivos de audio para:
1.  **Transcribir** el contenido hablado con alta precisión temporal.
2.  **Segmentar** el audio en frases coherentes.
3.  **Analizar** la carga emocional de cada segmento y del audio global.
4.  **Traducir** (Español -> Inglés) para aprovechar modelos entrenados en corpus anglosajones.

El análisis no es estático; el sistema calcula un **"Score Emocional Ponderado"** basado en la duración de cada segmento, otorgando mayor peso a las emociones expresadas durante períodos más largos de tiempo.

## 🛠️ Arquitectura y Modelos

El sistema integra múltiples modelos de IA para diferentes tareas:

| Tarea | Modelo / Tecnología | Fuente |
| :--- | :--- | :--- |
| **ASR (Transcription)** | `Whisper-1` | OpenAI API |
| **Emotion Analysis (LLM)** | `GPT-4o-mini` | OpenAI API |
| **Emotion (Text - EN)** | `distilroberta-base` (j-hartmann) | Hugging Face |
| **Emotion (Text - ES)** | `robertuito-emotion` (Pysentimiento) | Hugging Face |
| **Audio Classification** | `wav2vec2-base-superb-er` | Hugging Face |
| **Sentiment Analysis** | `multilingual-sentiment-analysis` | Hugging Face |
| **Translation** | `Google Translate API` | GoogleTrans |

## 🚀 Instalación y Despliegue

### Requisitos Previos
* **Docker** (Recomendado)
* **FFmpeg** (Si se ejecuta localmente sin Docker)
* Una API Key de **OpenAI**.

### 🐳 Despliegue con Docker (Recomendado)

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/thesis-audio-analysis.git](https://github.com/tu-usuario/thesis-audio-analysis.git)
    cd thesis-audio-analysis
    ```

2.  **Configurar Variables de Entorno:**
    Crea un archivo `.env` en la raíz:
    ```env
    OPENAI_API_KEY=sk-tu-clave-de-openai-aqui
    ```

3.  **Construir y Correr:**
    ```bash
    docker build -t audio-analysis-api .
    docker run -d -p 8000:8000 --env-file .env audio-analysis-api
    ```

### 🔧 Ejecución Local (Python)

1.  Crear entorno virtual e instalar dependencias:
    ```bash
    python -m venv venv
    source venv/bin/activate  # o venv\Scripts\activate en Windows
    pip install -r requirements.txt
    ```

2.  Iniciar el servidor:
    ```bash
    uvicorn app.main:app --reload
    ```

## 📡 Endpoints de la API

La API expone varios endpoints especializados según la estrategia de análisis deseada:

### 1. Análisis con LLM (Alta Precisión)
`POST /transcribe/emotion/openai`
* **Input:** Archivo de audio (`.wav`, `.mp3`, etc.).
* **Proceso:** Transcribe con Whisper y utiliza **GPT-4o-mini** para analizar el contexto semántico de cada segmento y asignar probabilidades a 7 emociones básicas (Ekman).
* **Ideal para:** Análisis contextual profundo donde la ironía o el sarcasmo pueden estar presentes.

### 2. Análisis Híbrido Español (Traducción)
`POST /transcribe/emotion/es`
* **Proceso:** 1. Transcribe audio en Español.
    2. Traduce cada segmento a Inglés.
    3. Aplica el modelo `DistilRoBERTa` (entrenado en inglés) para clasificación emocional.
* **Por qué:** Los modelos de emociones en inglés suelen tener mayor accuracy que los nativos en español.

### 3. Análisis Nativo Español (Pysentimiento)
`POST /transcribe/pysentimiento-es`
* **Proceso:** Transcribe y analiza directamente usando `Robertuito`, un modelo optimizado para tweets y texto en español.
* **Ideal para:** Audio informal, modismos latinoamericanos.

### 4. Análisis Acústico Puro (Wav2Vec)
`POST /audio-emotions`
* **Proceso:** No analiza el texto, sino las **ondas sonoras** (tono, pitch, velocidad).
* **Ideal para:** Detectar enojo o gritos, independientemente de lo que se diga.

## 📊 Estructura de Respuesta

Todos los endpoints de transcripción devuelven un JSON estructurado con métricas globales y detalladas:

```json
{
  "transcription": "Texto completo del audio...",
  "global_emotions": [
    { "label": "joy", "score": 0.85 },
    { "label": "neutral", "score": 0.15 }
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hola, estoy muy feliz",
      "top_emotion": { "label": "joy", "score": 0.99 },
      "emotions": [...]
    }
  ]
}