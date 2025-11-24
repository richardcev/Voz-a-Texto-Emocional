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


# 3. other model ---
pip install soundfile
pip install "huggingface_hub[cli]"

python training/train_audio_emotion_mspmea.py
```


### New Fine Tunning
- train_text_emotion_master.py
- d

```sh
# Deps with 3.11
curl https://pyenv.run | bash
exec $SHELL -l
pyenv install 3.11.9
pyenv local 3.11.9   # in priject folder

# activate venv
python -m venv .venv311
source .venv311/bin/activate
pip install -U pip wheel setuptools


# PyTorch + torchaudio CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchaudio==2.5.1+cu121


pip install "transformers==4.44.2" "datasets==2.21.0" accelerate==0.34.2 evaluate==0.4.2 \
            huggingface_hub==0.26.2 packaging==24.2 \
            scikit-learn==1.5.2 matplotlib==3.9.2 soundfile==0.12.1 \
            fastapi==0.115.6 "uvicorn[standard]"==0.32.0 \
            openai-whisper==20240930 pysentimiento==0.7.3
```

Test env
```sh
python - <<'PY'
import torch, torchaudio, whisper, transformers, datasets
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| cuda:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__, "| datasets:", datasets.__version__)
print("whisper OK:", whisper.__package__)
PY
```


```sh
SEED=42 REDUCE_TO_3=0 EPOCHS=8 \
python training/train_text_emotion_master.py

```




#### T 3

- train_text_emotion_pro
```sh
pip uninstall -y huggingface_hub
pip install "huggingface_hub==0.26.2"
pip install -U "transformers==4.44.2" "tokenizers==0.19.1" "datasets==2.21.0" "evaluate==0.4.3" "accelerate==0.34.2"


#  --------
huggingface-cli login

export HUGGINGFACE_HUB_TOKEN="hf_token"

SEED=42 REDUCE_TO_3=0 K_FOLDS=5 EPOCHS=10 \
BASE_MODEL="PlanTL-GOB-ES/roberta-base-bne" \
BATCH_TRAIN=8 BATCH_EVAL=16 GRAD_ACCUM=2 MAX_LEN=160 \
python training/train_text_emotion_pro.py


```

```
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WHISPER_MODEL=tiny
export WHISPER_DEVICE=cpu
export TEXT_ON_CPU=1
export FORCE_PYSENTIMIENTO_CPU=1
uvicorn app.main:app --host 0.0.0.0 --port 7777 --workers 1

```

<!-- -------------------------- -->
```
# En tu desktop 12GB:
export WHISPER_MODEL=small             # o base/small para +calidad
export WHISPER_DEVICE=gpu              # usa GPU si quieres
export TEXT_PRO_V2_ON_CPU=0            # corre el clasificador en GPU
uvicorn app.main:app --host 0.0.0.0 --port 7777 --workers 1

```

<!-- - -->
```sh
pip install faster-whisper


# Recomendado para karaoke real:
pip install faster-whisper

export USE_FASTER_WHISPER=1
export FASTER_WHISPER_MODEL=medium
export WHISPER_DEVICE=cuda
uvicorn app.main:app --host 0.0.0.0 --port 7777 --workers 1

```

```
```

```
```

```
```

```
```




