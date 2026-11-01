from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Experiment 2 configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

LABELS = [
    "contradiction",
    "different context",
    "inconsistent",
    "neutral",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

EXPERIMENT_NAME = "distilbert_lr1e5_wd01_warmup10"
MODEL_DIR = MODELS_DIR / EXPERIMENT_NAME
METRICS_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_metrics.json"
PREDICTIONS_PATH = (
    RESULTS_DIR / f"{EXPERIMENT_NAME}_validation_predictions.csv"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(filename):
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    data = pd.read_csv(path)
    required = ["sentence1", "sentence2", "gold_label"]

    if not set(required).issubset(data.columns):
        raise ValueError(
            f"{filename} must contain these columns: {required}"
        )

    if data[required].isnull().any().any():
        raise ValueError(f"{filename} contains missing values.")

    found_labels = set(data["gold_label"].astype(str).unique())
    unknown_labels = found_labels - set(LABELS)

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels in {filename}: {sorted(unknown_labels)}. "
            "Verify the label names before training."
        )

    return data


class SentencePairDataset(Dataset):
    def __init__(self, dataframe):
        self.sentence1 = dataframe["sentence1"].astype(str).tolist()
        self.sentence2 = dataframe["sentence2"].astype(str).tolist()
        self.labels = [
            LABEL_TO_ID[str(label)]
            for label in dataframe["gold_label"].tolist()
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sentence1[index],
            self.sentence2[index],
            self.labels[index],
        )


def make_collate_function(tokenizer):
    def collate(batch):
        sentence1 = [item[0] for item in batch]
        sentence2 = [item[1] for item in batch]

        labels = torch.tensor(
            [item[2] for item in batch],
            dtype=torch.long,
        )

        encoded = tokenizer(
            sentence1,
            sentence2,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded["labels"] = labels
        return encoded

    return collate


def evaluate_model(model, loader, device):
    model.eval()
    all_predictions = []
    all_actual = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            batch_size = batch["labels"].size(0)

            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size

            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_actual.extend(batch["labels"].cpu().tolist())

    actual_names = [ID_TO_LABEL[i] for i in all_actual]
    predicted_names = [ID_TO_LABEL[i] for i in all_predictions]

    metrics = {
        "validation_loss": total_loss / total_examples,
        "accuracy": accuracy_score(actual_names, predicted_names),
        "macro_f1": f1_score(
            actual_names,
            predicted_names,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual_names,
            predicted_names,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }

    report = classification_report(
        actual_names,
        predicted_names,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_names,
        predicted_names,
        labels=LABELS,
    )

    return metrics, report, matrix, actual_names, predicted_names


def main():
    set_seed(RANDOM_STATE)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("========== DISTILBERT EXPERIMENT 2 ==========")
    print("Device:", device)
    print("Learning rate:", LEARNING_RATE)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Warmup ratio:", WARMUP_RATIO)

    # IMPORTANT: only training and validation data are loaded.
    # The final test set remains untouched.
    train = load_split("train.csv")
    validation = load_split("validation.csv")

    missing_labels = (
        set(LABELS) - set(train["gold_label"].astype(str).unique())
    )

    if missing_labels:
        raise ValueError(
            f"Training split is missing labels: {sorted(missing_labels)}"
        )

    print("Training examples:", len(train))
    print("Validation examples:", len(validation))
    print("\nTraining label counts:")
    print(
        train["gold_label"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    train_dataset = SentencePairDataset(train)
    validation_dataset = SentencePairDataset(validation)
    collate_fn = make_collate_function(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    counts = train["gold_label"].value_counts()

    class_weights = torch.tensor(
        [
            len(train) / (len(LABELS) * counts[label])
            for label in LABELS
        ],
        dtype=torch.float32,
        device=device,
    )

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    best_results = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_examples = 0

        print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

        for batch_number, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            labels_batch = batch.pop("labels")
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = loss_function(outputs.logits, labels_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            batch_size = labels_batch.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size

            if batch_number % 50 == 0:
                print(
                    f"Processed {batch_number}/"
                    f"{len(train_loader)} batches"
                )

        train_loss = running_loss / total_examples

        (
            metrics,
            report,
            matrix,
            actual_names,
            predicted_names,
        ) = evaluate_model(model, validation_loader, device)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {metrics['validation_loss']:.4f}")
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Validation weighted-F1: {metrics['weighted_f1']:.4f}")

        # Save the checkpoint with the best validation macro-F1.
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]

            best_results = {
                "model": MODEL_NAME,
                "experiment": EXPERIMENT_NAME,
                "training_examples": len(train),
                "validation_examples": len(validation),
                "random_state": RANDOM_STATE,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "epochs_configured": EPOCHS,
                "best_epoch": epoch + 1,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "class_weighted_loss": True,
                "evaluation_split": "validation",
                "final_test_used": False,
                **metrics,
                "labels": LABELS,
                "confusion_matrix": matrix.tolist(),
                "classification_report": report,
            }

            best_predictions = validation.copy()
            best_predictions["predicted_label"] = predicted_names
            best_predictions["actual_label_check"] = actual_names

            # Separate filenames preserve the original baseline.
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            best_predictions.to_csv(
                PREDICTIONS_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            with open(METRICS_PATH, "w", encoding="utf-8") as file:
                json.dump(best_results, file, indent=4)

            print("Saved this experiment's best checkpoint.")

    if best_results is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    print("\n========== BEST VALIDATION RESULTS ==========")
    print("Best epoch:", best_results["best_epoch"])
    print(f"Accuracy: {best_results['accuracy']:.4f}")
    print(f"Macro-F1: {best_results['macro_f1']:.4f}")
    print(f"Weighted-F1: {best_results['weighted_f1']:.4f}")

    print("\nPer-class results:")
    for label in LABELS:
        values = best_results["classification_report"][label]
        print(
            f"{label:<22} "
            f"precision={values['precision']:.2f} "
            f"recall={values['recall']:.2f} "
            f"f1={values['f1-score']:.2f} "
            f"support={int(values['support'])}"
        )

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Label order:", LABELS)
    print(np.array(best_results["confusion_matrix"]))

    print("\nModel saved to:", MODEL_DIR)
    print("Metrics saved to:", METRICS_PATH)
    print("Validation predictions saved to:", PREDICTIONS_PATH)
    print("Final test set was not loaded or used.")


if __name__ == "__main__":
    main()