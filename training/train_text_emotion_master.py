# -*- coding: utf-8 -*-
import os, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate
from sklearn.metrics import balanced_accuracy_score

# ======================= Config ==========================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)

# 1 -> colapsa a {positivo, negativo, neutral}
REDUCE_TO_3 = bool(int(os.getenv("REDUCE_TO_3", "0")))

BASE_MODEL_NAME = os.getenv(
    "BASE_MODEL_NAME",
    # Ligero para 4GB VRAM. Si quieres BERT base, cámbialo,
    # pero mantén los parámetros de ahorro de VRAM.
    "dccuchile/distilbert-base-spanish-uncased",
)
DATASET_NAME = os.getenv("DATASET_NAME", "pysentimiento/emociones_colombia")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EXP_DIR = BASE_DIR / "experiments" / f"text_master_{int(time.time())}"
(EXP_DIR / "results").mkdir(parents=True, exist_ok=True)
(EXP_DIR / "figs").mkdir(parents=True, exist_ok=True)

OUT_MODEL_DIR = MODELS_DIR / (
    "emotion_es_master_3c" if REDUCE_TO_3 else "emotion_es_master_5c"
)

EMO_COLS = ["alegria", "miedo", "asco", "tristeza"]
MAP_3 = {
    "alegria": "positivo",
    "miedo": "negativo",
    "asco": "negativo",
    "tristeza": "negativo",
    "neutral": "neutral",
}
LABELS_5 = ["alegria", "miedo", "asco", "tristeza", "neutral"]
LABELS_3 = ["positivo", "negativo", "neutral"]
LABELS = LABELS_3 if REDUCE_TO_3 else LABELS_5

# ======================= Datos ===========================
print(f"[DATA] Cargando {DATASET_NAME}…")
raw = load_dataset(DATASET_NAME)["train"]


def to_label(example):
    active = [e for e in EMO_COLS if example.get(e, 0) == 1]
    lab = active[0] if active else "neutral"
    if REDUCE_TO_3:
        lab = MAP_3[lab]
    example["y"] = LABELS.index(lab)
    example["label_name"] = lab
    return example


proc = raw.map(to_label)
X = proc["text"]
y = np.array(proc["y"], dtype=np.int64)

# 80/10/10 estratificado
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
)


def to_ds(texts, labels):
    return [{"text": t, "labels": int(l)} for t, l in zip(texts, labels)]


ds_train = to_ds(X_train, y_train)
ds_val = to_ds(X_val, y_val)
ds_test = to_ds(X_test, y_test)

# ===================== Tokenización ======================
tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
id2label = {i: n for i, n in enumerate(LABELS)}
label2id = {n: i for i, n in enumerate(LABELS)}


def tokenize(batch):
    # 96 para reducir VRAM. Sube si te alcanza (128).
    return tok(batch["text"], truncation=True, padding="max_length", max_length=96)


hf_train = (
    Dataset.from_list(ds_train)
    .map(tokenize, batched=True)
    .remove_columns(["text"])
    .with_format("torch")
)
hf_val = (
    Dataset.from_list(ds_val)
    .map(tokenize, batched=True)
    .remove_columns(["text"])
    .with_format("torch")
)
hf_test = (
    Dataset.from_list(ds_test)
    .map(tokenize, batched=True)
    .remove_columns(["text"])
    .with_format("torch")
)

# =================== Class Weights =======================
counts = np.bincount(y_train, minlength=len(LABELS))
weights = 1.0 / (counts + 1e-9)
weights = weights / weights.sum() * len(LABELS)
class_weights = torch.tensor(weights, dtype=torch.float)

# ==================== Modelo & Métricas ==================
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)
# ahorro VRAM
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

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


# ================== TrainingArguments ====================
args = TrainingArguments(
    output_dir=str(EXP_DIR / "checkpoints"),
    eval_strategy="epoch",  # futuro-proof (evaluation_strategy -> eval_strategy)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    # Ahorro VRAM: lotes chicos + acumulación
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # 4*4 => 16 efectivo
    learning_rate=2e-5,
    num_train_epochs=int(os.getenv("EPOCHS", "8")),
    weight_decay=0.02,
    label_smoothing_factor=0.1,
    logging_steps=50,
    seed=SEED,
    report_to=[],
    save_total_limit=2,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    eval_accumulation_steps=16,
)


# Loss con pesos
def weighted_loss(model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
    loss = loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss


trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tok,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.compute_loss = weighted_loss

print("[TRAIN] Iniciando…")
trainer.train()

print("[EVAL] Test…")
test_metrics = trainer.evaluate(hf_test)
print(test_metrics)

# ============= Predicciones detalladas & AUC =============
pred = trainer.predict(hf_test)
probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
y_pred = probs.argmax(axis=1)

report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABELS))))

try:
    y_bin = label_binarize(y_test, classes=list(range(len(LABELS))))
    auc_ovr = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
except Exception:
    auc_ovr = None


# Bootstrap 95% CI para F1-macro
def bootstrap_ci(y_true, y_pred, B=1000, alpha=0.05):
    from sklearn.metrics import f1_score

    n = len(y_true)
    scores = []
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


f1_lo, f1_hi = bootstrap_ci(np.array(y_test), np.array(y_pred))

# ================ Guardar artefactos ======================
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT_MODEL_DIR))
tok.save_pretrained(str(OUT_MODEL_DIR))

with open(EXP_DIR / "results" / "args.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "reduce_to_3": REDUCE_TO_3,
            "base_model": BASE_MODEL_NAME,
            "dataset": DATASET_NAME,
            "class_weights": weights.tolist(),
            "test_metrics": test_metrics,
            "auc_macro_ovr": auc_ovr,
            "f1_macro_ci95": [f1_lo, f1_hi],
            "labels": LABELS,
        },
        f,
        indent=2,
    )

with open(EXP_DIR / "results" / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

fig = plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Text SER)")
plt.colorbar()
ticks = np.arange(len(LABELS))
plt.xticks(ticks, LABELS, rotation=45)
plt.yticks(ticks, LABELS)
plt.tight_layout()
plt.ylabel("True")
plt.xlabel("Pred")
fig.savefig(EXP_DIR / "figs" / "confusion_matrix.png", bbox_inches="tight")
plt.close(fig)

print("[DONE] Modelo:", OUT_MODEL_DIR)
print("[DONE] Experimentos:", EXP_DIR)
