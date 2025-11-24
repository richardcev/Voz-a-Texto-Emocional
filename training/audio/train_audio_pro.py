# -*- coding: utf-8 -*-
"""
training/audio/train_audio_pro.py

Fine-tuning para SER (audio -> emoción) con Wav2Vec2ForSequenceClassification.
Exporta a:
  models/audio_emotion_pro_5c   (o ..._3c si REDUCE_TO_3=1)
Genera thresholds.json por clase (F1 óptimo).

ENV útiles:
  REDUCE_TO_3=0|1
  SEED=42
  DATASET=  (nombre HF)  O  MANIFEST_CSV=/ruta/manifest.csv
  BASE_MODEL=facebook/wav2vec2-base
  MAX_DURATION=10.0   (segundos, corta audios más largos)
  BATCH_TRAIN=4
  BATCH_EVAL=4
  EPOCHS=10
  LR=1e-5
  GRAD_ACCUM=2
  FP16=1|0
  CPU_ONLY=0|1
"""

import os, json, time, random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from datasets import load_dataset, Dataset, Audio, DatasetDict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# ---------------- Config ----------------
SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

REDUCE_TO_3 = bool(int(os.getenv("REDUCE_TO_3", "0")))
CPU_ONLY = bool(int(os.getenv("CPU_ONLY", "0")))
DEVICE = "cpu" if CPU_ONLY else ("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parents[2]  # <repo>/training/audio/.. -> <repo>
RUN_DIR = BASE_DIR / "experiments" / f"audio_pro_{int(time.time())}"
(RUN_DIR / "results").mkdir(parents=True, exist_ok=True)

MODEL_OUT = (
    BASE_DIR
    / "models"
    / ("audio_emotion_pro_3c" if REDUCE_TO_3 else "audio_emotion_pro_5c")
)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

DATASET = os.getenv("DATASET", "")  # ej: "superb" (si trajeras algo HF con 'audio')
MANIFEST_CSV = os.getenv("MANIFEST_CSV", "")  # CSV/TSV: filepath,label
BASE_MODEL = os.getenv("BASE_MODEL", "facebook/wav2vec2-base")

MAX_DURATION = float(os.getenv("MAX_DURATION", "10.0"))  # segundos
SAMPLE_RATE = 16000

BATCH_TRAIN = int(os.getenv("BATCH_TRAIN", "4"))
BATCH_EVAL = int(os.getenv("BATCH_EVAL", "4"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = float(os.getenv("LR", "1e-5"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
FP16 = bool(int(os.getenv("FP16", "1")))

# Clases destino
LABELS_5 = ["alegria", "miedo", "asco", "tristeza", "neutral"]
LABELS_3 = ["positivo", "negativo", "neutral"]
LABELS = LABELS_3 if REDUCE_TO_3 else LABELS_5

# Mapeo desde etiquetas crudas típicas de SER a nuestras clases
# (ajústalo a tu dataset real)
MAP_5 = {
    "happy": "alegria",
    "happiness": "alegria",
    "joy": "alegria",
    "fear": "miedo",
    "scared": "miedo",
    "disgust": "asco",
    "disgusted": "asco",
    "sad": "tristeza",
    "sadness": "tristeza",
    "neutral": "neutral",
    "calm": "neutral",
    "other": "neutral",
}
MAP_3 = {
    "alegria": "positivo",
    "miedo": "negativo",
    "asco": "negativo",
    "tristeza": "negativo",
    "neutral": "neutral",
}


def to_target(label_str: str) -> str:
    s = label_str.strip().lower()
    base = MAP_5.get(s, s)
    if base not in LABELS_5:
        # último fallback
        base = "neutral"
    if REDUCE_TO_3:
        return MAP_3[base]
    return base


# --------------- Carga datos ---------------
def load_from_manifest(path: str) -> DatasetDict:
    # lee CSV o TSV filepath,label
    p = Path(path)
    sep = "," if p.suffix.lower() == ".csv" else "\t"
    rows = []
    with open(p, "r", encoding="utf8") as f:
        header = f.readline().strip().split(sep)
        # fuerza nombres
        if header[0].lower() != "filepath" or header[1].lower() != "label":
            raise RuntimeError("Manifest debe tener columnas: filepath,label")
        for line in f:
            if not line.strip():
                continue
            fp, lab = line.strip().split(sep)[:2]
            rows.append({"audio": fp, "label": to_target(lab)})
    ds = Dataset.from_list(rows)
    # estratificar 80/10/10
    ds = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=SEED)
    test_valid = ds["test"].train_test_split(
        test_size=0.5, stratify_by_column="label", seed=SEED
    )
    ds = DatasetDict(
        train=ds["train"], validation=test_valid["train"], test=test_valid["test"]
    )
    # castear audio
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds


def load_from_hf(name: str) -> DatasetDict:
    # Debe tener columna 'audio' o similar; ajusta aquí si tu dataset difiere
    ds = load_dataset(name)
    # Normaliza a splits train/validation/test
    if isinstance(ds, Dataset):
        ds = DatasetDict(
            train=ds,
            validation=ds.select(range(int(0.1 * len(ds)))),
            test=ds.select(range(int(0.1 * len(ds)))),
        )
    else:
        if "validation" not in ds:
            split = ds["train"].train_test_split(test_size=0.2, seed=SEED)
            val_test = split["test"].train_test_split(test_size=0.5, seed=SEED)
            ds = DatasetDict(
                train=split["train"],
                validation=val_test["train"],
                test=val_test["test"],
            )
    # Asegura Audio
    for k in ds:
        if ds[k].features.get("audio") is None:
            raise RuntimeError(f"El dataset HF '{name}' debe tener columna 'audio'.")
        ds[k] = ds[k].cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Debe existir columna de label -> mapea a nuestras LABELS
    # Aquí suponemos una columna 'label' textual; adapta si es int
    def map_label(ex):
        # si label es int, reemplaza con tu mapping
        if isinstance(ex["label"], int):
            raise RuntimeError(
                "Este loader asume labels textuales. Adáptalo a tu dataset."
            )
        ex["label"] = to_target(str(ex["label"]))
        return ex

    ds = ds.map(map_label)
    return ds


if MANIFEST_CSV:
    ds = load_from_manifest(MANIFEST_CSV)
elif DATASET:
    ds = load_from_hf(DATASET)
else:
    raise RuntimeError("Define DATASET=<hf_name> o MANIFEST_CSV=/ruta/manifest.csv")

label2id = {n: i for i, n in enumerate(LABELS)}
id2label = {i: n for n, i in label2id.items()}

# -------------- Preproc --------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    BASE_MODEL, sampling_rate=SAMPLE_RATE
)
max_len_samples = int(MAX_DURATION * SAMPLE_RATE)


def preprocess(batch):
    # batch["audio"] es dict con 'array' y 'sampling_rate'
    arr = batch["audio"]["array"]
    # recorta/pad
    if arr.shape[0] > max_len_samples:
        arr = arr[:max_len_samples]
    # padding manual
    if arr.shape[0] < max_len_samples:
        pad = np.zeros(max_len_samples - arr.shape[0], dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    out = feature_extractor(arr, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    batch["input_values"] = out["input_values"][0]
    batch["labels"] = label2id[batch["label"]]
    return batch


ds = ds.map(preprocess, remove_columns=ds["train"].column_names, num_proc=1)
ds.set_format(type="torch", columns=["input_values", "labels"])

# -------------- Sampler balanceado --------------
y_train = np.array(ds["train"]["labels"])
class_counts = np.bincount(y_train, minlength=len(LABELS))
weights_per_class = 1.0 / np.maximum(class_counts, 1)
weights = weights_per_class[y_train]
sampler = WeightedRandomSampler(
    torch.tensor(weights, dtype=torch.double),
    num_samples=len(weights),
    replacement=True,
)

# -------------- Modelo --------------
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label,
    problem_type="single_label_classification",
)

if DEVICE == "cuda":
    model = model.to("cuda")


# -------------- Métricas --------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean().item()
    # F1 macro & UAR
    try:
        from sklearn.metrics import f1_score

        f1m = f1_score(labels, preds, average="macro")
        uar = balanced_accuracy_score(labels, preds)
    except Exception:
        f1m, uar = float("nan"), float("nan")
    return {"accuracy": float(acc), "f1_macro": float(f1m), "uar": float(uar)}


# -------------- Trainer --------------
args = TrainingArguments(
    output_dir=str(RUN_DIR / "checkpoints"),
    learning_rate=LR,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    report_to=[],
    seed=SEED,
    fp16=bool(FP16 and DEVICE == "cuda"),
    dataloader_num_workers=2,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=feature_extractor,  # para save_pretrained
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# Forzar dataloader con sampler balanceado
def _train_dl():
    return torch.utils.data.DataLoader(
        trainer.train_dataset,
        sampler=sampler,
        batch_size=trainer.args.train_batch_size,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )


trainer.get_train_dataloader = _train_dl

# -------------- Train --------------
trainer.train()

# -------------- Test + thresholds --------------
metrics_test = trainer.evaluate(ds["test"])
print("[TEST] =>", metrics_test)

logits = trainer.predict(ds["test"]).predictions
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
y_true = np.array(ds["test"]["labels"])
y_pred = probs.argmax(axis=1)

# thresholds por clase via PR
y_true_bin = label_binarize(y_true, classes=list(range(len(LABELS))))
thresholds, pr_aucs = {}, {}
for c in range(len(LABELS)):
    prec, rec, th = precision_recall_curve(y_true_bin[:, c], probs[:, c])
    pr_aucs[LABELS[c]] = float(auc(rec, prec))
    f1s = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    best_idx = np.nanargmax(f1s)
    thresholds[LABELS[c]] = float(th[best_idx - 1] if 0 < best_idx < len(th) else 0.5)

# Reporte y CM
rep = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
with open(RUN_DIR / "results" / "report.json", "w") as f:
    json.dump(
        {
            "test_metrics": metrics_test,
            "report": rep,
            "pr_auc": pr_aucs,
            "labels": LABELS,
        },
        f,
        indent=2,
    )

# -------------- Guardar modelo --------------
trainer.save_model(str(MODEL_OUT))
feature_extractor.save_pretrained(str(MODEL_OUT))
with open(MODEL_OUT / "thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("[DONE] Modelo PRO (audio) guardado en:", MODEL_OUT)
print("[DONE] Experimentos:", RUN_DIR)
