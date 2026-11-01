
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_COLUMN = "gold_label"
TEXT_COLUMNS = ["sentence1", "sentence2"]

RANDOM_STATE = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_split(filename):
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/create_splits.py first."
        )

    data = pd.read_csv(path)

    required_columns = TEXT_COLUMNS + [LABEL_COLUMN]
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )

    if data[required_columns].isnull().any().any():
        raise ValueError(
            f"{filename} contains missing text or label values."
        )

    return data


# --------------------------------------------------
# COMBINE THE TWO SENTENCES
# --------------------------------------------------

def combine_sentences(data):
    return (
        data["sentence1"].astype(str)
        + " [PAIR] "
        + data["sentence2"].astype(str)
    )


# --------------------------------------------------
# BUILD THE BASELINE MODEL
# --------------------------------------------------

def build_model():
    word_features = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )

    char_features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )

    features = FeatureUnion([
        ("word_tfidf", word_features),
        ("char_tfidf", char_features),
    ])

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


# --------------------------------------------------
# MAIN TRAINING AND VALIDATION
# --------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train.csv")
    validation = load_split("validation.csv")

    print("========== BASELINE MODEL ==========")
    print("Training examples:", len(train))
    print("Validation examples:", len(validation))
    print("Labels:", sorted(train[LABEL_COLUMN].unique()))

    X_train = combine_sentences(train)
    y_train = train[LABEL_COLUMN]

    X_validation = combine_sentences(validation)
    y_validation = validation[LABEL_COLUMN]

    model = build_model()

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # Evaluate only on the validation set.
    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    labels = sorted(
        set(y_validation.unique()) | set(predictions)
    )

    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=labels,
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Macro-F1:   {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    print("\nPer-class classification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("Label order:", labels)
    print(matrix)

    # Save the trained baseline model.
    model_path = MODELS_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)

    # Save validation predictions for reproducibility.
    prediction_output = validation.copy()
    prediction_output["predicted_label"] = predictions
    prediction_output.to_csv(
        RESULTS_DIR / "baseline_validation_predictions.csv",
        index=False,
    )

    # Save metrics and per-class results.
    results = {
        "model": "TF-IDF word and character features + Logistic Regression",
        "training_examples": int(len(train)),
        "validation_examples": int(len(validation)),
        "random_state": RANDOM_STATE,
        "evaluation_split": "validation",
        "final_test_used": False,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels_in_confusion_matrix": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    metrics_path = RESULTS_DIR / "baseline_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n========== FILES SAVED ==========")
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print(
        "Validation predictions:",
        RESULTS_DIR / "baseline_validation_predictions.csv",
    )


if __name__ == "__main__":
    main()
