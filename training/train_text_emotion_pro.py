# -*- coding: utf-8 -*-
import os, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
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

# ----------------- Config -----------------
SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)
REDUCE_TO_3 = bool(int(os.getenv("REDUCE_TO_3", "0")))

# Puedes pasar BASE_MODEL por env, pero ya no se cargará por adelantado.
BASE_MODEL = os.getenv("BASE_MODEL", "PlanTL-GOB-ES/roberta-base-bne")
DATASET = os.getenv("DATASET", "pysentimiento/emociones_colombia")

BASE_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = BASE_DIR / "experiments" / f"text_pro_{int(time.time())}"
(RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "figs").mkdir(parents=True, exist_ok=True)

MODEL_OUT = (
    BASE_DIR
    / "models"
    / ("text_emotion_pro_3c" if REDUCE_TO_3 else "text_emotion_pro_5c")
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

KFOLDS = int(os.getenv("K_FOLDS", "5"))
MAX_LEN = int(os.getenv("MAX_LEN", "160"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = float(os.getenv("LR", "2e-5"))
BATCH_TRAIN = int(os.getenv("BATCH_TRAIN", "8"))
BATCH_EVAL = int(os.getenv("BATCH_EVAL", "16"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Orden de modelos candidatos (con fallback). El primero intentado será BASE_MODEL.
CANDIDATES = [
    BASE_MODEL,
    "PlanTL-GOB-ES/roberta-base-bne",  # org oficial
    "BSC-TeMU/roberta-base-bne",  # mirror/complementario (si existe y es público)
    "bertin-project/bertin-roberta-base-spanish",
    "dccuchile/bert-base-spanish-wwm-cased",
    "distilbert/distilbert-base-multilingual-cased",
]

# ----------------- Datos -----------------
print(f"[DATA] Cargando {DATASET}…")
raw = load_dataset(DATASET)["train"]


def to_label(ex):
    active = [e for e in EMO_COLS if ex.get(e, 0) == 1]
    lab = active[0] if active else "neutral"
    if REDUCE_TO_3:
        lab = MAP_3[lab]
    ex["y"] = LABELS.index(lab)
    ex["label_name"] = lab
    return ex


proc = raw.map(to_label)
texts = np.array(proc["text"])
y = np.array(proc["y"], dtype=np.int64)

id2label = {i: n for i, n in enumerate(LABELS)}
label2id = {n: i for i, n in enumerate(LABELS)}

# ----------------- Métricas -----------------
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


# ----- Focal Loss + Class-Balanced (effective num) -----
def class_balanced_weights(y_labels, beta=0.999):
    counts = np.bincount(y_labels, minlength=len(LABELS)).astype(np.float32)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(LABELS)
    return torch.tensor(weights, dtype=torch.float)


def focal_loss(logits, labels, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, labels, reduction="none", weight=alpha)
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def load_tok_and_model(model_id, num_labels, id2label, label2id):
    tok = AutoTokenizer.from_pretrained(
        model_id, token=HF_TOKEN, use_fast=True, local_files_only=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        token=HF_TOKEN,
    )
    return tok, model


# -------- Selección de modelo/tokenizer con fallback (¡ahora sí!) --------
last_err = None
tok = None
base_used = None
for mid in CANDIDATES:
    try:
        tok, _ = load_tok_and_model(mid, len(LABELS), id2label, label2id)
        base_used = mid
        print(f"[MODEL] Usando {mid}")
        break
    except Exception as e:
        print(f"[WARN] Falló {mid}: {e}")
        last_err = e
if tok is None:
    raise RuntimeError(f"No pude cargar ningún tokenizer. Último error: {last_err}")


# encode depende de 'tok', por eso lo definimos después de seleccionarlo
def encode(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)


# ----------------- K-Fold CV -----------------
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

best_f1 = -1.0
fold_summaries = []

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

    # Oversampling con WeightedRandomSampler
    class_sample_count = np.bincount(y_trn, minlength=len(LABELS))
    weights_per_class = 1.0 / np.maximum(class_sample_count, 1)
    sample_weights = weights_per_class[y_trn]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Cargar modelo (mismo checkpoint que el tokenizer elegido)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_used,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        token=HF_TOKEN,
    )

    # Pesos class-balanced y compute_loss (focal)
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
        evaluation_strategy="epoch",  # en 4.44 sigue siendo válido
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        save_total_limit=2,
        fp16=True,  # ahorro VRAM
        dataloader_num_workers=2,
        gradient_checkpointing=True,  # más ahorro VRAM
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

    # Forzar dataloader con sampler
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

    # Train
    trainer.train()

    # Eval test
    metrics_test = trainer.evaluate(ds_tst)
    print("TEST:", metrics_test)

    # Predicciones y umbrales por clase (vía PR)
    logits = trainer.predict(ds_tst).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred = probs.argmax(axis=1)

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

    # Guardar mejor global por F1
    if metrics_test.get("eval_f1_macro", -1) > best_f1:
        best_f1 = metrics_test["eval_f1_macro"]
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(MODEL_OUT))
        tok.save_pretrained(str(MODEL_OUT))
        with open(MODEL_OUT / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)


# Resumen K-fold
def mean_std(key):
    vals = [r.get(key, np.nan) for r in fold_summaries]
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}


summary = {
    "accuracy": mean_std("eval_accuracy"),
    "f1_macro": mean_std("eval_f1_macro"),
    "uar": mean_std("eval_uar"),
    "folds": KFOLDS,
    "seed": SEED,
    "base_model": base_used,
    "dataset": DATASET,
    "labels": LABELS,
}
with open(RUN_DIR / "results" / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[DONE] Mejor modelo guardado en:", MODEL_OUT)
print("[DONE] Experimentos:", RUN_DIR)
