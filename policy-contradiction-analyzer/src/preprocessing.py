
"""
Data validation and preprocessing utilities for the
AI Policy Contradiction & Impact Analyzer.

This module:
1. Checks missing values.
2. Identifies exact duplicate sentence pairs.
3. Validates relationship labels.
4. Prepares English sentence pairs for modeling.
5. Preserves the original raw dataset.
"""

import pandas as pd


# Columns required for the English classification task
TEXT_COLUMNS = ["sentence1", "sentence2"]
LABEL_COLUMN = "gold_label"

EXPECTED_LABELS = {
    "neutral",
    "inconsistent",
    "different context",
    "contradiction",
}


def validate_data(df):
    """Validate the dataset structure and contents."""

    required_columns = TEXT_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    return True


def analyze_missing_values(df):
    """Count missing values in each column."""

    return df.isnull().sum()


def find_duplicate_pairs(df):
    """
    Find duplicate English sentence pairs.

    A pair is considered duplicated only when both sentences
    match an earlier pair in the same order.
    """

    return df.duplicated(
        subset=TEXT_COLUMNS,
        keep="first"
    )


def validate_labels(df):
    """Check whether all labels belong to the expected classes."""

    actual_labels = set(
        df[LABEL_COLUMN].dropna().unique()
    )

    unexpected = actual_labels - EXPECTED_LABELS
    missing = EXPECTED_LABELS - actual_labels

    return {
        "actual_labels": sorted(actual_labels),
        "unexpected_labels": sorted(unexpected),
        "missing_expected_labels": sorted(missing),
        "valid": len(unexpected) == 0,
    }


def prepare_english_data(df):
    """
    Create a modeling dataframe containing English sentence pairs
    and labels. The input dataframe is not modified.
    """

    validate_data(df)

    prepared = df[
        TEXT_COLUMNS + [LABEL_COLUMN]
    ].copy()

    # Remove leading/trailing whitespace only.
    for column in TEXT_COLUMNS:
        prepared[column] = prepared[column].astype("string").str.strip()

    # Keep rows with complete text and labels.
    prepared = prepared.dropna(
        subset=TEXT_COLUMNS + [LABEL_COLUMN]
    ).copy()

    return prepared


def run_data_analysis(df):
    """Print a basic dataset validation report."""

    validate_data(df)

    print("\nDATASET VALIDATION REPORT")
    print("=" * 45)

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    print("\nMissing values per column:")
    print(analyze_missing_values(df))

    duplicate_mask = find_duplicate_pairs(df)

    print(
        "\nDuplicate English sentence pairs:",
        int(duplicate_mask.sum())
    )

    label_report = validate_labels(df)

    print("\nLabel validation:")
    print(label_report)

    prepared = prepare_english_data(df)

    print("\nPrepared English data:")
    print(f"Rows with complete required fields: {len(prepared)}")

    print("\nPrepared label distribution:")
    print(prepared[LABEL_COLUMN].value_counts())

    return prepared


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw_data = load_raw_data()
    prepared_data = run_data_analysis(raw_data)
