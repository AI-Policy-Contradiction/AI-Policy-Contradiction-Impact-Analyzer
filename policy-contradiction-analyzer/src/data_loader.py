
"""
Dataset loader for the AI Policy Contradiction & Impact Analyzer.

Dataset:
annotated_regulatory_sentences_english.csv

Languages:
- English: sentence1, sentence2
- German: sentence1_DE, sentence2_DE

Label:
- gold_label
"""

from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annotated_regulatory_sentences_english.csv"
)


def load_raw_data():
    """Load the original dataset without modifying it."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Place the CSV inside data/raw/."
        )

    # The supplied dataset uses semicolons as separators.
    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    return df


def validate_dataset(df):
    """Check that the required dataset columns exist."""

    required_columns = [
        "sentence1_DE",
        "sentence2_DE",
        "gold_label",
        "sentence1",
        "sentence2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


if __name__ == "__main__":
    dataset = load_raw_data()
    validate_dataset(dataset)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("-" * 40)
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print("\nColumn names:")
    print(dataset.columns.tolist())

    print("\nLabel distribution:")
    print(dataset["gold_label"].value_counts())

    print("\nFirst 3 English sentence pairs:")
    print(
        dataset[
            ["sentence1", "sentence2", "gold_label"]
        ].head(3).to_string(index=False)
    )
