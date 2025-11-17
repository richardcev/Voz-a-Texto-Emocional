from pathlib import Path
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# -----------------------------
# Configuración
# -----------------------------

# Modelo base en español (BERT)
BASE_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

# Dataset de emociones en español (Colombia)
DATASET_NAME = "pysentimiento/emociones_colombia"

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "emotion_es_bert_colombia"

# Columnas de emoción en el dataset
EMOTION_COLUMNS = ["alegria", "miedo", "asco", "tristeza"]

# Nombres de clases para clasificación
LABEL_NAMES = ["alegria", "miedo", "asco", "tristeza", "neutral"]  # 5 clases


def main():
    # 1. Cargar dataset
    print(f"Cargando dataset: {DATASET_NAME}")
    raw_dataset = load_dataset(DATASET_NAME)
    full_ds = raw_dataset["train"]
    print(full_ds)

    # 2. Convertir columnas binarias en una sola etiqueta 'labels'
    def add_single_label(example):
        active = [emo for emo in EMOTION_COLUMNS if example[emo] == 1]

        if len(active) == 1:
            label_name = active[0]
        elif len(active) == 0:
            label_name = "neutral"
        else:
            # Varias emociones activas -> nos quedamos con la primera
            label_name = active[0]

        example["label_name"] = label_name
        example["labels"] = LABEL_NAMES.index(label_name)
        return example

    full_ds = full_ds.map(add_single_label)

    # 3. Nos quedamos solo con texto + labels
    keep_cols = ["text", "labels"]
    full_ds = full_ds.remove_columns(
        [c for c in full_ds.column_names if c not in keep_cols]
    )

    # 4. Crear splits train/validation/test: 80/10/10
    print("Creando splits train/validation/test...")
    train_test = full_ds.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

    dataset = {
        "train": train_test["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }

    print(dataset["train"])
    print(dataset["validation"])
    print(dataset["test"])

    # 5. Tokenizador y modelo base
    print("Cargando modelo base:", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=label2id,
    )

    # 6. Preprocesamiento (tokenización)
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    print("Tokenizando dataset...")
    encoded_dataset = {}
    for split_name, split_ds in dataset.items():
        enc = split_ds.map(preprocess_function, batched=True)
        enc = enc.remove_columns(["text"])
        enc.set_format("torch")
        encoded_dataset[split_name] = enc

    # 7. Métricas (accuracy + F1 macro)
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = {}
        metrics.update(accuracy.compute(predictions=preds, references=labels))
        metrics.update(
            f1_metric.compute(predictions=preds, references=labels, average="macro")
        )
        return metrics

    # 8. Parámetros de entrenamiento (simple, compatible)
    ckpt_dir = BASE_DIR / "models" / "emotion_es_bert_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 9. Entrenar
    print("Entrenando modelo...")
    trainer.train()

    # 10. Evaluar en test
    print("Evaluando en test...")
    test_metrics = trainer.evaluate(encoded_dataset["test"])
    print("Resultados en test:", test_metrics)

    # 11. Guardar modelo final
    print("Guardando modelo en:", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
