
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5


# --------------------------------------------------
# REPRODUCIBILITY AND DEVICE
# --------------------------------------------------

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_split(filename):
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/create_splits.py first."
        )

    data = pd.read_csv(path)
    required = ["sentence1", "sentence2", "gold_label"]

    if not set(required).issubset(data.columns):
        raise ValueError(
            f"{filename} must contain {required}"
        )

    if data[required].isnull().any().any():
        raise ValueError(
            f"{filename} contains missing text or labels."
        )

    return data


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class SentencePairDataset(Dataset):
    def __init__(self, dataframe, label_to_id):
        self.sentence1 = dataframe["sentence1"].astype(str).tolist()
        self.sentence2 = dataframe["sentence2"].astype(str).tolist()
        self.labels = [
            label_to_id[label]
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


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(model, loader, id_to_label):
    model.eval()

    all_predictions = []
    all_actual = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            batch_size = batch["labels"].size(0)
            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size

            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_actual.extend(
                batch["labels"].cpu().numpy().tolist()
            )

    label_ids = list(range(len(id_to_label)))
    labels = [id_to_label[i] for i in label_ids]

    actual_names = [id_to_label[i] for i in all_actual]
    predicted_names = [id_to_label[i] for i in all_predictions]

    metrics = {
        "validation_loss": total_loss / total_examples,
        "accuracy": accuracy_score(actual_names, predicted_names),
        "macro_f1": f1_score(
            actual_names,
            predicted_names,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual_names,
            predicted_names,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
    }

    report = classification_report(
        actual_names,
        predicted_names,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_names,
        predicted_names,
        labels=labels,
    )

    return metrics, report, matrix, actual_names, predicted_names


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("========== DISTILBERT EXPERIMENT ==========")
    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)

    train = load_split("train.csv")
    validation = load_split("validation.csv")

    # The held-out test set is deliberately not loaded.
    labels = sorted(train["gold_label"].unique())

    label_to_id = {
        label: index for index, label in enumerate(labels)
    }
    id_to_label = {
        index: label for label, index in label_to_id.items()
    }

    if set(validation["gold_label"].unique()) - set(labels):
        raise ValueError(
            "Validation contains a label missing from training."
        )

    print("Training examples:", len(train))
    print("Validation examples:", len(validation))
    print("Classes:", labels)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)

    print("\nLoading pretrained tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading pretrained classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(DEVICE)

    train_dataset = SentencePairDataset(train, label_to_id)
    validation_dataset = SentencePairDataset(validation, label_to_id)

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

    # Class weights are calculated using training labels only.
    counts = train["gold_label"].value_counts()
    class_weights = [
        len(train) / (len(labels) * counts[label])
        for label in labels
    ]

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_results = None
    best_epoch = None

    model_dir = MODELS_DIR / "distilbert_baseline"

    # --------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_examples = 0

        print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

        for batch_number, batch in enumerate(
            train_loader, start=1
        ):
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            labels_batch = batch.pop("labels")

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = loss_function(
                outputs.logits,
                labels_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

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
        ) = evaluate_model(
            model,
            validation_loader,
            id_to_label,
        )

        print(f"Training loss: {train_loss:.4f}")
        print(
            f"Validation loss: {metrics['validation_loss']:.4f}"
        )
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
        print(
            f"Validation weighted-F1: "
            f"{metrics['weighted_f1']:.4f}"
        )

        # Select the best epoch by validation Macro-F1.
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1

            # Store all results from this exact checkpoint.
            best_results = {
                "model": MODEL_NAME,
                "experiment": "DistilBERT sentence-pair classification",
                "training_examples": len(train),
                "validation_examples": len(validation),
                "random_state": RANDOM_STATE,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "epochs_configured": EPOCHS,
                "best_epoch": best_epoch,
                "learning_rate": LEARNING_RATE,
                "class_weighted_loss": True,
                "evaluation_split": "validation",
                "final_test_used": False,
                **metrics,
                "labels": labels,
                "confusion_matrix": matrix.tolist(),
                "classification_report": report,
            }

            # Save validation predictions from this same epoch.
            predictions_df = validation.copy()
            predictions_df["predicted_label"] = predicted_names
            predictions_df.to_csv(
                RESULTS_DIR / "distilbert_validation_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # Save the exact model checkpoint selected above.
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            with open(
                RESULTS_DIR / "distilbert_metrics.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(best_results, file, indent=4)

            print("Best validation model and results saved.")

    # --------------------------------------------------
    # FINAL SUMMARY: ALL FROM THE BEST EPOCH
    # --------------------------------------------------

    if best_results is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    print("\n========== BEST DISTILBERT RESULTS ==========")
    print("Best epoch:", best_results["best_epoch"])
    print(f"Accuracy: {best_results['accuracy']:.4f}")
    print(f"Macro-F1: {best_results['macro_f1']:.4f}")
    print(f"Weighted-F1: {best_results['weighted_f1']:.4f}")

    print("\nPer-class report from best epoch:")
    report = best_results["classification_report"]

    print(
        f"{'Class':<22}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'F1-score':>11}"
        f"{'Support':>10}"
    )

    for label in labels:
        values = report[label]
        print(
            f"{label:<22}"
            f"{values['precision']:>11.2f}"
            f"{values['recall']:>10.2f}"
            f"{values['f1-score']:>11.2f}"
            f"{int(values['support']):>10}"
        )

    for average_name in ["macro avg", "weighted avg"]:
        values = report[average_name]
        print(
            f"{average_name:<22}"
            f"{values['precision']:>11.2f}"
            f"{values['recall']:>10.2f}"
            f"{values['f1-score']:>11.2f}"
            f"{int(values['support']):>10}"
        )

    print("\nConfusion matrix from best epoch:")
    print("Label order:", labels)
    print(np.array(best_results["confusion_matrix"]))

    print("\nSaved model:", model_dir)
    print(
        "Saved metrics:",
        RESULTS_DIR / "distilbert_metrics.json",
    )
    print(
        "Saved validation predictions:",
        RESULTS_DIR / "distilbert_validation_predictions.csv",
    )
    print("\nFinal test set was not used.")


if __name__ == "__main__":
    main()
