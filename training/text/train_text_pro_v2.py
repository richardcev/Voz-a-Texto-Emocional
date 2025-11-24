# -*- coding: utf-8 -*-
"""
training/text/train_text_pro_v2.py

Entrenamiento PRO v2 para emociones en texto (ES):
- Multi-corpus: acepta varios datasets de HF o CSVs locales (text,label)
- Unificación de etiquetas a 5C {alegria,miedo,asco,tristeza,neutral} o 3C (positivo,negativo,neutral)
- Focal Loss + Class-Balanced
- K-Fold CV + selección por F1 macro
- Umbrales por clase (PR curve) + Temperature Scaling (calibración)
- Checkpoints + export final a models/text_emotion_pro_v2_{5c|3c}

ENV útiles:
  DATASETS="pysentimiento/emociones_colombia,tu_usuario/emo_event_es"  # HF
  CSV_LIST="/abs/a.csv,/abs/b.csv"                                     # locales (text,label)
  REDUCE_TO_3=0|1
  BASE_MODEL="PlanTL-GOB-ES/roberta-large-bne"                         # con 12GB VRAM
  EPOCHS=6 BATCH_TRAIN=16 BATCH_EVAL=32 MAX_LEN=192 GRAD_ACCUM=1
  CUDA_VISIBLE_DEVICES=0  # para desktop
"""

import os, json, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

import evaluate

# ================= Config =================
SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)
REDUCE_TO_3 = bool(int(os.getenv("REDUCE_TO_3", "0")))
BASE_MODEL = os.getenv("BASE_MODEL", "PlanTL-GOB-ES/roberta-large-bne")  # desktop 12GB
DATASETS = [
    s.strip()
    for s in os.getenv("DATASETS", "pysentimiento/emociones_colombia").split(",")
    if s.strip()
]
CSV_LIST = [s.strip() for s in os.getenv("CSV_LIST", "").split(",") if s.strip()]

BASE_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = BASE_DIR / "experiments" / f"text_pro_v2_{int(time.time())}"
(RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "figs").mkdir(parents=True, exist_ok=True)

MODEL_OUT = (
    BASE_DIR
    / "models"
    / ("text_emotion_pro_v2_3c" if REDUCE_TO_3 else "text_emotion_pro_v2_5c")
)

EMO5 = ["alegria", "miedo", "asco", "tristeza", "neutral"]
MAP_3 = {
    "alegria": "positivo",
    "miedo": "negativo",
    "asco": "negativo",
    "tristeza": "negativo",
    "neutral": "neutral",
}
LAB5 = EMO5
LAB3 = ["positivo", "negativo", "neutral"]
LABELS = LAB3 if REDUCE_TO_3 else LAB5
id2label = {i: n for i, n in enumerate(LABELS)}
label2id = {n: i for i, n in enumerate(LABELS)}

KFOLDS = int(os.getenv("K_FOLDS", "5"))
MAX_LEN = int(os.getenv("MAX_LEN", "192"))
EPOCHS = int(os.getenv("EPOCHS", "6"))
LR = float(os.getenv("LR", "2e-5"))
BATCH_TRAIN = int(os.getenv("BATCH_TRAIN", "16"))
BATCH_EVAL = int(os.getenv("BATCH_EVAL", "32"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "1"))
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

acc = evaluate.load("accuracy")
f1m = evaluate.load("f1")


def compute_metrics(p):
    logits, labels = p
    preds = logits.argmax(-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1m.compute(predictions=preds, references=labels, average="macro")[
            "f1"
        ],
        "uar": balanced_accuracy_score(labels, preds),
    }


# ---------- Lectura de datos ----------
def load_hf(name: str) -> Dataset:
    ds = load_dataset(name)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split]


def load_csv(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            txt = (r.get("text") or r.get("texto") or "").strip()
            lab = (r.get("label") or r.get("etiqueta") or "").strip().lower()
            if not txt or not lab:
                continue
            rows.append({"text": txt, "label_name": lab})
    return Dataset.from_list(rows)


def normalize_5c(ex: dict) -> dict:
    # soporta datasets tipo pysentimiento (one-hot) y label_name directo
    if "label_name" in ex and ex["label_name"]:
        lab = ex["label_name"].lower()
    else:
        active = [
            e for e in ["alegria", "miedo", "asco", "tristeza"] if ex.get(e, 0) == 1
        ]
        lab = active[0] if active else "neutral"
    if lab not in EMO5:
        # mapear alias comunes o fuera de dominio a neutral
        m = {
            "joy": "alegria",
            "happiness": "alegria",
            "fear": "miedo",
            "disgust": "asco",
            "sadness": "tristeza",
            "anger": "asco",  # enfado -> asco (hecho práctico para 5C)
            "surprise": "neutral",
            "other": "neutral",
            "none": "neutral",
        }
        lab = m.get(lab, "neutral")
    ex["label_name"] = lab
    ex["y"] = LABELS.index(MAP_3[lab] if REDUCE_TO_3 else lab)
    return ex


def merge_datasets() -> Dataset:
    all_splits = []
    for name in DATASETS:
        try:
            all_splits.append(load_hf(name))
            print(f"[DATA] HF ok: {name}")
        except Exception as e:
            print(f"[WARN] HF fail {name}: {e}")
    for p in CSV_LIST:
        try:
            all_splits.append(load_csv(p))
            print(f"[DATA] CSV ok: {p}")
        except Exception as e:
            print(f"[WARN] CSV fail {p}: {e}")
    if not all_splits:
        raise RuntimeError("No se pudo cargar ningún dataset (HF o CSV).")
    ds = all_splits[0]
    for extra in all_splits[1:]:
        ds = concatenate_datasets([ds, extra])
    return ds


print("[DATA] Cargando datasets…")
raw = merge_datasets()
proc = raw.map(normalize_5c)
texts = np.array(proc["text"])
y = np.array(proc["y"], dtype=np.int64)


# ---------- Tokenizer/model ----------
def load_tok(model_id):  # solo tokenizer aquí
    return AutoTokenizer.from_pretrained(
        model_id, token=HF_TOKEN, use_fast=True, local_files_only=False
    )


tok = load_tok(BASE_MODEL)


def encode(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)


# ---------- Losses ----------
def class_balanced_weights(y_labels, beta=0.999):
    counts = np.bincount(y_labels, minlength=len(LABELS)).astype(np.float32)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-8)
    w = w / w.sum() * len(LABELS)
    return torch.tensor(w, dtype=torch.float)


def focal_loss(logits, labels, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, labels, reduction="none", weight=alpha)
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


# ---------- Temperature scaling ----------
class TempScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.t.clamp_min(1e-3)


def fit_temperature(logits, labels, max_iter=200, lr=0.01):
    device = logits.device
    scaler = TempScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    def _loss():
        opt.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(_loss)
    return float(scaler.t.detach().cpu().item())


# ---------- KFold ----------
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
best_f1 = -1.0
fold_summaries = []
best_artifacts = {}

for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(y)), y), start=1):
    print(f"\n===== FOLD {fold}/{KFOLDS} =====")
    X_tr, y_tr = texts[tr_idx], y[tr_idx]
    X_te, y_te = texts[te_idx], y[te_idx]

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED
    )

    ds_trn = (
        Dataset.from_dict({"text": X_trn, "labels": y_trn})
        .map(encode, batched=True)
        .remove_columns(["text"])
        .with_format("torch")
    )
    ds_val = (
        Dataset.from_dict({"text": X_val, "labels": y_val})
        .map(encode, batched=True)
        .remove_columns(["text"])
        .with_format("torch")
    )
    ds_tst = (
        Dataset.from_dict({"text": X_te, "labels": y_te})
        .map(encode, batched=True)
        .remove_columns(["text"])
        .with_format("torch")
    )

    # Oversampling
    class_sample_count = np.bincount(y_trn, minlength=len(LABELS))
    weights_per_class = 1.0 / np.maximum(class_sample_count, 1)
    sample_weights = weights_per_class[y_trn]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        token=HF_TOKEN,
    )

    cb_weights = class_balanced_weights(y_trn, beta=0.999)

    def compute_loss_cb(model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = focal_loss(
            outputs.logits,
            labels,
            alpha=cb_weights.to(outputs.logits.device),
            gamma=2.0,
        )
        return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(RUN_DIR / "checkpoints" / f"fold_{fold}"),
        learning_rate=LR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        weight_decay=0.02,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tok,
        train_dataset=ds_trn,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.compute_loss = compute_loss_cb

    # Dataloader con sampler
    def _custom_train_dl():
        return torch.utils.data.DataLoader(
            trainer.train_dataset,
            sampler=sampler,
            batch_size=trainer.args.train_batch_size,
            collate_fn=trainer.data_collator,
            num_workers=2,
            pin_memory=True,
        )

    trainer.get_train_dataloader = _custom_train_dl

    trainer.train()

    metrics_test = trainer.evaluate(ds_tst)
    print("TEST:", metrics_test)

    # Predicciones test
    logits = torch.tensor(trainer.predict(ds_tst).predictions)
    probs = torch.softmax(logits, dim=-1).numpy()
    y_pred = probs.argmax(axis=1)

    # Umbrales por clase (PR)
    y_true_bin = label_binarize(y_te, classes=list(range(len(LABELS))))
    thresholds, pr_aucs = {}, {}
    for c in range(len(LABELS)):
        prec, rec, th = precision_recall_curve(y_true_bin[:, c], probs[:, c])
        pr_aucs[LABELS[c]] = float(auc(rec, prec))
        f1s = (2 * prec * rec) / np.clip(prec + rec, 1e-9, None)
        best_idx = np.nanargmax(f1s)
        thresholds[LABELS[c]] = float(
            th[best_idx - 1] if 0 < best_idx < len(th) else 0.5
        )

    # Temperature scaling en VALID (recalibra logits)
    val_logits = torch.tensor(trainer.predict(ds_val).predictions)
    val_labels = torch.tensor(y_val)
    temp = fit_temperature(val_logits.to(model.device), val_labels.to(model.device))
    print(f"[CAL] temperature={temp:.4f}")

    # Métricas
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(LABELS))))
    report = classification_report(y_te, y_pred, target_names=LABELS, output_dict=True)
    try:
        auc_ovr = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc_ovr = None

    with open(RUN_DIR / "results" / f"fold_{fold}.json", "w") as f:
        json.dump(
            {
                "test_metrics": metrics_test,
                "classification_report": report,
                "auc_macro_ovr": auc_ovr,
                "pr_auc": pr_aucs,
                "thresholds": thresholds,
                "labels": LABELS,
                "temperature": temp,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.colorbar()
    ticks = np.arange(len(LABELS))
    plt.xticks(ticks, LABELS, rotation=45)
    plt.yticks(ticks, LABELS)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(RUN_DIR / "figs" / f"cm_fold_{fold}.png", bbox_inches="tight")
    plt.close()

    fold_summaries.append(metrics_test)

    # Guardar mejor por F1
    if metrics_test.get("eval_f1_macro", -1) > best_f1:
        best_f1 = metrics_test["eval_f1_macro"]
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(MODEL_OUT))
        tok.save_pretrained(str(MODEL_OUT))
        with open(MODEL_OUT / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
        with open(MODEL_OUT / "labels.json", "w") as f:
            json.dump(LABELS, f, indent=2)
        with open(MODEL_OUT / "temperature.json", "w") as f:
            json.dump({"temperature": temp}, f, indent=2)


# Resumen KFold
def mean_std(key):
    vals = [r.get(key, np.nan) for r in fold_summaries]
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}


summary = {
    "accuracy": mean_std("eval_accuracy"),
    "f1_macro": mean_std("eval_f1_macro"),
    "uar": mean_std("eval_uar"),
    "folds": KFOLDS,
    "seed": SEED,
    "base_model": BASE_MODEL,
    "labels": LABELS,
    "datasets": DATASETS,
    "csvs": CSV_LIST,
}
with open(RUN_DIR / "results" / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[DONE] Mejor modelo guardado en:", MODEL_OUT)
print("[DONE] Experimentos:", RUN_DIR)
