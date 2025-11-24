PY=python

install:
	$(PY) -m pip install --upgrade pip wheel setuptools
	# PyTorch ya lo instalaste (2.5.1+cu121). Luego:
	pip install -r requirements.txt

train-text-master-5:
	SEED=42 REDUCE_TO_3=0 EPOCHS=8 $(PY) training/train_text_emotion_master.py

train-text-master-3:
	SEED=42 REDUCE_TO_3=1 EPOCHS=8 $(PY) training/train_text_emotion_master.py

train-audio-master-cv:
	SEED=42 K_FOLDS=5 $(PY) training/train_audio_emotion_master.py

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 7777 --reload

test-master-text:
	curl -F "file=@/path/audio_es.wav" http://127.0.0.1:7777/transcribe/emotion-es-master

test-master-audio:
	curl -F "file=@/path/audio_es.wav" http://127.0.0.1:7777/audio-emotions/master



# ----------------
train-text-pro-5:
	SEED=42 REDUCE_TO_3=0 K_FOLDS=5 EPOCHS=10 \
	BASE_MODEL="PlanTL-GOB-ES/roberta-base-bne" \
	BATCH_TRAIN=8 BATCH_EVAL=16 GRAD_ACCUM=2 MAX_LEN=160 \
	python training/train_text_emotion_pro.py

train-text-pro-3:
	SEED=42 REDUCE_TO_3=1 K_FOLDS=5 EPOCHS=10 \
	BASE_MODEL="PlanTL-GOB-ES/roberta-base-bne" \
	BATCH_TRAIN=8 BATCH_EVAL=16 GRAD_ACCUM=2 MAX_LEN=160 \
	python training/train_text_emotion_pro.py



