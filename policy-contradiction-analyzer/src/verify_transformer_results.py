
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "results" / "distilbert_validation_predictions.csv"
METRICS_PATH = ROOT / "results" / "distilbert_metrics.json"
OUTPUT_PATH = ROOT / "results" / "distilbert_verified_metrics.json"


def find_column(df, candidates):
    """Find a column using case-insensitive candidate names."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    return None


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {METRICS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    print("========== SAVED PREDICTIONS ==========")
    print(f"Rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    actual_col = find_column(
        df,
        ["actual_label", "actual", "true_label",
         "gold_label", "y_true", "true", "label"]
    )

    predicted_col = find_column(
        df,
        ["predicted_label", "predicted", "prediction",
         "pred_label", "y_pred", "pred"]
    )

    if actual_col is None or predicted_col is None:
        print("\nCould not identify the actual/predicted columns.")
        print("Send me the columns and first 5 rows printed above.")
        return

    actual = df[actual_col].astype(str).str.strip()
    predicted = df[predicted_col].astype(str).str.strip()

    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Actual and predicted labels have invalid lengths.")

    labels = sorted(set(actual) | set(predicted))

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(
        actual, predicted, labels=labels,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        actual, predicted, labels=labels,
        average="weighted", zero_division=0
    )

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(actual, predicted, labels=labels)

    print("\n========== INDEPENDENTLY RECALCULATED METRICS ==========")
    print(f"Examples evaluated: {len(df)}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy:.2%})")
    print(f"Macro-F1:           {macro_f1:.4f}")
    print(f"Weighted-F1:        {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        actual, predicted, labels=labels, zero_division=0
    ))
    print("Label order:", labels)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        original_metrics = json.load(file)

    verified = {
        "source_predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "source_metrics": str(METRICS_PATH.relative_to(ROOT)),
        "number_of_examples": len(df),
        "actual_label_column": str(actual_col),
        "predicted_label_column": str(predicted_col),
        "labels": labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "original_metrics_json": original_metrics,
        "note": (
            "Metrics recalculated from the saved validation predictions. "
            "Confirm which epoch those predictions represent before "
            "reporting them as best-epoch results."
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(verified, file, indent=2, ensure_ascii=False)

    print("\nSaved independently recalculated metrics to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
