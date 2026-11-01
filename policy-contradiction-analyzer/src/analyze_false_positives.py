
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "results" / "distilbert_validation_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "contradiction_false_positives.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {
        "sentence1",
        "sentence2",
        "gold_label",
        "predicted_label",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in ["gold_label", "predicted_label"]:
        df[col] = df[col].astype(str).str.strip()

    # Model predicted contradiction, but the gold label is another class.
    false_positives = df[
        (df["predicted_label"] == "contradiction")
        & (df["gold_label"] != "contradiction")
    ].copy()

    print("========== CONTRADICTION FALSE POSITIVES ==========")
    print(f"Validation examples: {len(df)}")
    print(
        "Predicted contradictions:",
        (df["predicted_label"] == "contradiction").sum(),
    )
    print(f"False positives: {len(false_positives)}")

    if false_positives.empty:
        print("\nNo false positives found.")
    else:
        print("\nFalse-positive sentence pairs:")

        for index, row in false_positives.iterrows():
            print(f"\n--- Example {index} ---")
            print(f"Actual label: {row['gold_label']}")
            print(f"Predicted label: {row['predicted_label']}")
            print(f"Sentence 1: {row['sentence1']}")
            print(f"Sentence 2: {row['sentence2']}")

    false_positives.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved false positives to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
