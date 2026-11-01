
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_raw_data
from preprocessing import prepare_english_data


# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10


# --------------------------------------------------
# CREATE DATASET SPLITS
# --------------------------------------------------

def create_splits():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare the original dataset
    raw_data = load_raw_data()
    prepared_data = prepare_english_data(raw_data)

    rows_after_preparation = len(prepared_data)

    print("\n========== DATASET CHECK ==========")
    print("Original dataset rows:", len(raw_data))
    print("Rows after preparation:", rows_after_preparation)

    # 2. Identify duplicate English sentence pairs
    duplicate_mask = prepared_data.duplicated(
        subset=["sentence1", "sentence2"],
        keep="first"
    )

    duplicate_count = int(duplicate_mask.sum())

    print("Duplicate pairs found after preparation:", duplicate_count)

    # Save duplicate examples for investigation
    duplicates = prepared_data.loc[duplicate_mask].copy()
    duplicates_path = PROCESSED_DIR / "duplicate_pairs.csv"
    duplicates.to_csv(duplicates_path, index=False)

    # 3. Remove duplicate sentence pairs
    data = prepared_data.drop_duplicates(
        subset=["sentence1", "sentence2"],
        keep="first"
    ).reset_index(drop=True)

    print("Rows after deduplication:", len(data))
    print("Duplicate pairs removed:", rows_after_preparation - len(data))
    print("Unique examples available:", len(data))

    # 4. Validate labels before splitting
    expected_labels = {
        "neutral",
        "inconsistent",
        "different context",
        "contradiction",
    }

    actual_labels = set(data["gold_label"].dropna().unique())

    if actual_labels != expected_labels:
        raise ValueError(
            "Unexpected label set.\n"
            f"Expected: {sorted(expected_labels)}\n"
            f"Found: {sorted(actual_labels)}"
        )

    # Every label must have enough examples for stratified splitting
    label_counts = data["gold_label"].value_counts()

    if (label_counts < 3).any():
        raise ValueError(
            "At least one label has fewer than 3 examples. "
            "Stratified splitting cannot proceed safely."
        )

    # 5. Reserve 20% as the final test set
    # Do not use this set for model selection or tuning.
    development, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["gold_label"],
    )

    # 6. Split 10% of the development set into validation data
    # This gives approximately 72% training, 8% validation,
    # and 20% final testing overall.
    train, validation = train_test_split(
        development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=development["gold_label"],
    )

    # 7. Save all splits
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    development.to_csv(
        PROCESSED_DIR / "development_80pct.csv",
        index=False,
    )

    # 8. Save a reproducibility report
    report = {
        "original_examples": int(len(raw_data)),
        "rows_after_preparation": int(rows_after_preparation),
        "duplicate_pairs_removed": duplicate_count,
        "unique_examples": int(len(data)),
        "development_examples": int(len(development)),
        "training_examples": int(len(train)),
        "validation_examples": int(len(validation)),
        "final_test_examples": int(len(test)),
        "random_state": RANDOM_STATE,
        "stratified": True,
        "test_size": TEST_SIZE,
        "validation_fraction_of_development": VALIDATION_SIZE,
        "label_column": "gold_label",
        "text_columns": ["sentence1", "sentence2"],
        "duplicate_detection_columns": ["sentence1", "sentence2"],
        "final_test_reserved_for_final_evaluation": True,
    }

    report_path = PROCESSED_DIR / "split_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # 9. Print the split summary
    print("\n========== SPLIT SUMMARY ==========")
    print("Training examples:", len(train))
    print("Validation examples:", len(validation))
    print("Final test examples:", len(test))
    print("Development examples:", len(development))
    print("Total split examples:", len(train) + len(validation) + len(test))

    print("\nTraining label distribution:")
    print(train["gold_label"].value_counts().sort_index())

    print("\nValidation label distribution:")
    print(validation["gold_label"].value_counts().sort_index())

    print("\nFinal test label distribution:")
    print(test["gold_label"].value_counts().sort_index())

    print("\nFiles saved in:", PROCESSED_DIR)
    print("Duplicate examples saved to:", duplicates_path)
    print("Split report saved to:", report_path)


if __name__ == "__main__":
    create_splits()
