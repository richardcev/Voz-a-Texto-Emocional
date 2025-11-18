import numpy as np
from pathlib import Path

from datasets import load_dataset, Audio, ClassLabel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# -----------------------------
# Configuración
# -----------------------------

BASE_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
DATASET_NAME = "jaimebellver/SER-MSPMEA-Spanish"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "audio_emotion_mspmea"

SAMPLE_RATE = 16_000  # 16 kHz, según la ficha del dataset


def main():
    # 1. Cargar dataset de HF
    print(f"Cargando dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # Este dataset solo tiene split "train"
    full_ds = dataset["train"]
    print(full_ds)

    # 2. Obtener lista de etiquetas
    emotion_feat = full_ds.features["emotion"]

    if isinstance(emotion_feat, ClassLabel):
        label_names = list(emotion_feat.names)
    else:
        label_names = sorted(set(full_ds["emotion"]))

    print("Etiquetas:", label_names)

    label2id = {lab: i for i, lab in enumerate(label_names)}
    id2label = {i: lab for lab, i in label2id.items()}

    def encode_label(example):
        # si ya viene como string, lo mapeamos
        if isinstance(example["emotion"], str):
            example["label"] = label2id[example["emotion"]]
        else:
            # por si viene como int
            example["label"] = int(example["emotion"])
        return example

    full_ds = full_ds.map(encode_label)

    # 3. Asegurarnos de que el audio está a 16kHz
    full_ds = full_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # 4. Crear splits train/val/test 80/10/10
    print("Creando splits train/validation/test...")
    train_test = full_ds.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

    dataset_splits = {
        "train": train_test["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }

    print(dataset_splits["train"])
    print(dataset_splits["validation"])
    print(dataset_splits["test"])

    # 5. Cargar feature extractor y modelo base
    print("Cargando modelo base:", BASE_MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_NAME)

    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    # 6. Preprocesamiento de audio
    def preprocess_function(batch):
        # batch["audio"] es una lista de dicts {"array": np.array, "sampling_rate": ...}
        audio_arrays = [a["array"] for a in batch["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            truncation=True,
            max_length=SAMPLE_RATE * 4,  # ~4 segundos
        )
        batch["input_values"] = inputs["input_values"]
        return batch

    print("Extrayendo características de audio...")
    encoded_splits = {}
    for split_name, split_ds in dataset_splits.items():
        enc = split_ds.map(
            preprocess_function,
            batched=True,
        )
        # eliminamos columnas que ya no hacen falta
        cols_to_remove = [
            c for c in enc.column_names if c not in ["input_values", "label"]
        ]
        enc = enc.remove_columns(cols_to_remove)
        enc.set_format(
            type="torch",
            columns=["input_values", "label"],
        )
        encoded_splits[split_name] = enc

    # 7. Métricas
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = {}
        metrics.update(accuracy.compute(predictions=preds, references=labels))
        metrics.update(
            f1_metric.compute(
                predictions=preds,
                references=labels,
                average="macro",
            )
        )
        return metrics

    # 8. Parámetros de entrenamiento (simples, compatibles)
    ckpt_dir = BASE_DIR / "models" / "audio_emotion_mspmea_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_splits["train"],
        eval_dataset=encoded_splits["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # 9. Entrenar
    print("Entrenando modelo de audio...")
    trainer.train()

    # 10. Evaluar en test
    print("Evaluando en test...")
    test_metrics = trainer.evaluate(encoded_splits["test"])
    print("Resultados en test:", test_metrics)

    # 11. Guardar modelo final
    print("Guardando modelo en:", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    feature_extractor.save_pretrained(str(OUTPUT_DIR))
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
