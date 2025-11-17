# 🎙️ Whisper Transcription & Emotion API

```sh
callcenter-emotion-system/
├── app/
│   └── main.py                  # FastAPI: Whisper + emociones
├── training/
│   └── train_emotion_model.py   # Script de fine-tuning (texto)
├── models/
│   └── emotion_es_roberta/      # Aquí se guardará tu modelo fine-tuned
├── requirements.txt
└── README.md
```


## Training
```sh
cd training
python train_emotion_model.py
```


## API
```sh
uvicorn app.main:app --reload --host 0.0.0.0 --port 7777
```
- `http://127.0.0.1:7777/docs`







## FINE-TUNNING
- Dependencies:
```sh
pip install datasets transformers accelerate evaluate
pip install "protobuf<5"
pip install scikit-learn
```

- Structure:
```sh
a_init_uide_tesis/
├── app/
│   └── main.py
├── models/
│   └── emotion_es_roberta/   # <-- aquí se guardará el modelo fine-tuned
├── training/
│   └── train_emotion_model.py
└── ...
```


### Train
```sh
# 1.
python training/train_emotion_model.py

# 2.
uvicorn app.main:app --reload --host 0.0.0.0 --port 7777

```

