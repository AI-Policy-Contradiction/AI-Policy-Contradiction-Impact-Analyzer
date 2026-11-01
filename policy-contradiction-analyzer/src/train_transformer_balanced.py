
import os
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# EXPERIMENT 3: DISTILBERT WITH BALANCED SAMPLING
# Uses train.csv and validation.csv only.
# The final test set is NOT loaded or used.
# ============================================================

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
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
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
VALIDATION_PATH = os.path.join("data", "processed", "validation.csv")

MODEL_OUTPUT_DIR = os.path.join("models", "distilbert_balanced_sampling")
METRICS_PATH = os.path.join(
    "results", "distilbert_balanced_sampling_metrics.json"
)
PREDICTIONS_PATH = os.path.join(
    "results", "distilbert_balanced_sampling_validation_predictions.csv"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PolicyPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

        required_columns = ["sentence1", "sentence2", "gold_label"]
        missing_columns = [
            column for column in required_columns
            if column not in self.dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        if self.dataframe[required_columns].isnull().any().any():
            raise ValueError(
                "Dataset contains missing sentence or label values."
            )

        unknown_labels = (
            set(self.dataframe["gold_label"].unique()) - set(LABELS)
        )

        if unknown_labels:
            raise ValueError(
                f"Unexpected labels found: {sorted(unknown_labels)}"
            )

        self.labels = [
            LABEL_TO_ID[label]
            for label in self.dataframe["gold_label"]
        ]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        encoding = self.tokenizer(
            str(row["sentence1"]),
            str(row["sentence2"]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )

        return item


def evaluate(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_labels.extend(
                batch["labels"].cpu().numpy().tolist()
            )

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(LABELS))),
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "true_ids": all_labels,
        "predicted_ids": all_predictions,
    }


def main():
    set_seed(SEED)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    if not os.path.exists(VALIDATION_PATH):
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    print("Loading training and validation data only...")
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(validation_df)}")
    print("Final test set: NOT loaded or used.")

    print("\nTraining label distribution:")
    print(train_df["gold_label"].value_counts().reindex(LABELS, fill_value=0))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = PolicyPairDataset(train_df, tokenizer)
    validation_dataset = PolicyPairDataset(validation_df, tokenizer)

    # Balanced sampling: examples from less frequent classes are
    # sampled more often during training.
    # Do NOT add class weights to the loss as well.
    class_counts = np.bincount(
        train_dataset.labels,
        minlength=len(LABELS),
    )

    if np.any(class_counts == 0):
        raise ValueError(
            "At least one label is missing from the training data."
        )

    class_weights_for_sampling = 1.0 / class_counts

    sample_weights = [
        class_weights_for_sampling[label]
        for label in train_dataset.labels
    ]

    generator = torch.Generator()
    generator.manual_seed(SEED)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    print("\nLoading fresh DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    # Ordinary unweighted loss because sampling is balanced.
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(
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
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"  Batch {step}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f}"
                )

        average_train_loss = total_loss / len(train_loader)
        validation_results = evaluate(
            model, validation_loader, device
        )

        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": float(average_train_loss),
            "validation_accuracy": validation_results["accuracy"],
            "validation_macro_f1": validation_results["macro_f1"],
            "validation_weighted_f1": validation_results["weighted_f1"],
        }
        history.append(epoch_results)

        print(f"Training loss: {average_train_loss:.4f}")
        print(
            f"Validation accuracy: "
            f"{validation_results['accuracy']:.4f}"
        )
        print(
            f"Validation macro-F1: "
            f"{validation_results['macro_f1']:.4f}"
        )
        print(
            f"Validation weighted-F1: "
            f"{validation_results['weighted_f1']:.4f}"
        )

        # Save the best checkpoint based on validation macro-F1.
        if validation_results["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_results["macro_f1"]
            best_epoch = epoch + 1

            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

            best_predictions = validation_results["predicted_ids"]
            best_true_ids = validation_results["true_ids"]

            print("Saved new best validation checkpoint.")

    # Reload the best validation checkpoint for final validation reporting.
    print("\nReloading best validation checkpoint...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_OUTPUT_DIR
    )
    best_model.to(device)

    final_results = evaluate(
        best_model, validation_loader, device
    )

    true_labels = [
        ID_TO_LABEL[label_id]
        for label_id in final_results["true_ids"]
    ]
    predicted_labels = [
        ID_TO_LABEL[label_id]
        for label_id in final_results["predicted_ids"]
    ]

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS,
    )

    predictions_df = validation_df.reset_index(drop=True).copy()
    predictions_df["predicted_label"] = predicted_labels
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    results_to_save = {
        "experiment": "DistilBERT with balanced sampling",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs_configured": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "sampling": "WeightedRandomSampler with replacement",
        "loss": "Unweighted CrossEntropyLoss",
        "best_epoch": best_epoch,
        "accuracy": final_results["accuracy"],
        "macro_f1": final_results["macro_f1"],
        "weighted_f1": final_results["weighted_f1"],
        "labels": LABELS,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "training_history": history,
        "test_set_used": False,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results_to_save, file, indent=4)

    contradiction_recall = report["contradiction"]["recall"]

    print("\n" + "=" * 55)
    print("EXPERIMENT 3 RESULTS — VALIDATION SET ONLY")
    print("=" * 55)
    print(f"Best epoch: {best_epoch}")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Macro-F1: {final_results['macro_f1']:.4f}")
    print(f"Weighted-F1: {final_results['weighted_f1']:.4f}")
    print(f"Contradiction recall: {contradiction_recall:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    print(f"\nModel saved to: {MODEL_OUTPUT_DIR}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Validation predictions saved to: {PREDICTIONS_PATH}")
    print("Final test set remains untouched.")


if __name__ == "__main__":
    main()
