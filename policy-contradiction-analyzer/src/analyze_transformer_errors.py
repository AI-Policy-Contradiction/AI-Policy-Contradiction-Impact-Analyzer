
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "results" / "distilbert_validation_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "contradiction_false_negatives.csv"


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

    df["gold_label"] = df["gold_label"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip()

    # Actual contradictions that the model predicted as another class.
    false_negatives = df[
        (df["gold_label"] == "contradiction")
        & (df["predicted_label"] != "contradiction")
    ].copy()

    print("========== CONTRADICTION ERROR ANALYSIS ==========")
    print(f"Validation examples: {len(df)}")
    print(f"Actual contradictions: {(df['gold_label'] == 'contradiction').sum()}")
    print(f"Correctly detected contradictions: {((df['gold_label'] == 'contradiction') & (df['predicted_label'] == 'contradiction')).sum()}")
    print(f"Missed contradictions: {len(false_negatives)}")

    if false_negatives.empty:
        print("\nNo contradiction false negatives found.")
    else:
        print("\nMissed contradiction predictions:")
        for index, row in false_negatives.iterrows():
            print(f"\n--- Example {index} ---")
            print(f"Model predicted: {row['predicted_label']}")
            print(f"Sentence 1: {row['sentence1']}")
            print(f"Sentence 2: {row['sentence2']}")

    false_negatives.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved missed examples to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
