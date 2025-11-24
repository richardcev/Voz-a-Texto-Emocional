# -*- coding: utf-8 -*-
import os, json, time, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset, Audio, ClassLabel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
import evaluate

# ====================== Config ============================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL = os.getenv("AUDIO_BASE_MODEL", "facebook/wav2vec2-large-xlsr-53")
DATASET = os.getenv("AUDIO_DATASET", "jaimebellver/SER-MSPMEA-Spanish")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "4.0"))
KFOLDS = int(os.getenv("K_FOLDS", "5"))

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "experiments" / f"audio_master_{int(time.time())}"
(EXP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(EXP_DIR / "results").mkdir(parents=True, exist_ok=True)
(EXP_DIR / "figs").mkdir(parents=True, exist_ok=True)
MODEL_OUT = BASE_DIR / "models" / "audio_emotion_master"

# ====================== Datos =============================
print(f"[DATA] Cargando {DATASET} …")
raw = load_dataset(DATASET)["train"]
raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

emo_feat = raw.features["emotion"]
labels = (
    list(emo_feat.names)
    if isinstance(emo_feat, ClassLabel)
    else sorted(set(raw["emotion"]))
)
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}


def add_label(ex):
    ex["label"] = (
        lab2id[ex["emotion"]] if isinstance(ex["emotion"], str) else int(ex["emotion"])
    )
    return ex


raw = raw.map(add_label)

# ================== Feature Extractor =====================
fe = AutoFeatureExtractor.from_pretrained(BASE_MODEL)


def augment(x, sr):
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, 0.005, size=x.shape)
    if np.random.rand() < 0.5:
        shift = int(0.02 * sr)
        x = np.roll(x, shift if np.random.rand() < 0.5 else -shift)
    return x


def preprocess(batch, train=False):
    arrays = []
    for a in batch["audio"]:
        arr = a["array"]
        if train:
            arr = augment(arr, SAMPLE_RATE)
        arrays.append(arr)
    out = fe(
        arrays,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        truncation=True,
        max_length=int(SAMPLE_RATE * MAX_SECONDS),
    )
    batch["input_values"] = out["input_values"]
    return batch


# =================== K-Fold CV ============================
acc = evaluate.load("accuracy")
f1m = evaluate.load("f1")


def metrics_fn(p):
    logits, labels_np = p
    preds = logits.argmax(-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels_np)["accuracy"],
        "f1_macro": f1m.compute(
            predictions=preds, references=labels_np, average="macro"
        )["f1"],
        "uar": balanced_accuracy_score(labels_np, preds),
    }


skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
y_all = np.array(raw["label"])
idxs = np.arange(len(raw))

fold_results = []
best_model = None
best_f1 = -1.0

for fold, (train_idx, test_idx) in enumerate(skf.split(idxs, y_all), start=1):
    print(f"\n===== FOLD {fold}/{KFOLDS} =====")
    train_split = raw.select(train_idx)
    test_split = raw.select(test_idx)
    val_splits = train_split.train_test_split(
        test_size=0.1, seed=SEED, stratify_by_column="label"
    )
    train_split, val_split = val_splits["train"], val_splits["test"]

    # pesos por fold
    y_tr = np.array(train_split["label"])
    cnt = np.bincount(y_tr, minlength=len(labels))
    w = 1.0 / (cnt + 1e-9)
    w = w / w.sum() * len(labels)
    class_w = torch.tensor(w, dtype=torch.float)

    ds_tr = (
        train_split.map(lambda b: preprocess(b, train=True), batched=True)
        .remove_columns(
            [c for c in train_split.column_names if c not in ["input_values", "label"]]
        )
        .with_format("torch")
    )
    ds_va = (
        val_split.map(lambda b: preprocess(b, train=False), batched=True)
        .remove_columns(
            [c for c in val_split.column_names if c not in ["input_values", "label"]]
        )
        .with_format("torch")
    )
    ds_te = (
        test_split.map(lambda b: preprocess(b, train=False), batched=True)
        .remove_columns(
            [c for c in test_split.column_names if c not in ["input_values", "label"]]
        )
        .with_format("torch")
    )

    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), label2id=lab2id, id2label=id2lab
    )
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    def weighted_loss(model, inputs, return_outputs=False):
        labels_t = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_w.to(logits.device))
        loss = loss_fn(logits, labels_t)
        return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(EXP_DIR / "checkpoints" / f"fold_{fold}"),
        learning_rate=1e-4,
        per_device_train_batch_size=2,  # audio ocupa más VRAM
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 2*4 => 8 efectivo
        num_train_epochs=6,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=SEED,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        eval_accumulation_steps=16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=fe,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.compute_loss = weighted_loss

    trainer.train()
    metrics_test = trainer.evaluate(ds_te)
    print("TEST:", metrics_test)

    preds_logits = trainer.predict(ds_te).predictions
    probs = torch.softmax(torch.tensor(preds_logits), dim=-1).numpy()
    y_pred = probs.argmax(axis=1)
    y_true = np.array(ds_te["label"])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )

    try:
        y_bin = label_binarize(y_true, classes=list(range(len(labels))))
        auc_ovr = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc_ovr = None

    with open(EXP_DIR / "results" / f"fold_{fold}.json", "w") as f:
        json.dump(
            {
                "metrics_test": metrics_test,
                "classification_report": report,
                "auc_macro_ovr": auc_ovr,
                "labels": labels,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(EXP_DIR / "figs" / f"cm_fold_{fold}.png", bbox_inches="tight")
    plt.close()

    fold_results.append(metrics_test)
    if metrics_test.get("eval_f1_macro", -1) > best_f1:
        best_f1 = metrics_test["eval_f1_macro"]
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(MODEL_OUT))
        fe.save_pretrained(str(MODEL_OUT))


# resumen
def mean_std(key):
    vals = [r.get(key, np.nan) for r in fold_results]
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}


summary = {
    "accuracy": mean_std("eval_accuracy"),
    "f1_macro": mean_std("eval_f1_macro"),
    "uar": mean_std("eval_uar"),
    "folds": KFOLDS,
    "seed": SEED,
    "base_model": BASE_MODEL,
    "dataset": DATASET,
}
with open(EXP_DIR / "results" / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[DONE] Mejor modelo guardado en:", MODEL_OUT)
print("[DONE] Experimentos:", EXP_DIR)
