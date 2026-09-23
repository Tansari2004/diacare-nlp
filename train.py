"""Train, evaluate and persist the DiaCare support-message classifiers."""

from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "messages.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

np.random.seed(SEED)


def train_classifier(df: pd.DataFrame, target: str):
    """Train one classifier and return its pipeline and evaluation summary."""
    texts = df["text"].astype(str).fillna("")
    labels = df[target].astype(str)
    counts = Counter(labels)
    can_stratify = labels.nunique() > 1 and min(counts.values()) >= 2

    train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels if can_stratify else None,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=20_000,
        strip_accents="unicode",
    )
    base_model = LinearSVC(class_weight="balanced" if target == "priority" else None)
    minimum_class_count = min(counts.values()) if counts else 0

    if minimum_class_count >= 3:
        estimator = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    elif minimum_class_count >= 2:
        estimator = CalibratedClassifierCV(base_model, method="sigmoid", cv=2)
    else:
        estimator = base_model

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", estimator)])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipeline,
        train_texts,
        train_labels,
        cv=folds,
        scoring="f1_macro",
    )

    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(holdout_texts)
    class_order = [str(value) for value in pipeline.classes_]
    report = classification_report(
        holdout_labels,
        predictions,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(holdout_labels, predictions, labels=class_order)

    print(f"\n=== {target.upper()} CROSS-VALIDATION ===")
    print(f"Macro F1 mean: {cv_scores.mean():.3f}")
    print(f"Macro F1 std:  {cv_scores.std():.3f}")
    print(f"\n=== {target.upper()} HOLDOUT TEST REPORT ===")
    print(
        classification_report(
            holdout_labels,
            predictions,
            labels=class_order,
            digits=3,
            zero_division=0,
        )
    )
    print("Confusion matrix:\n", matrix)

    metrics = {
        "cv_macro_f1_mean": round(float(cv_scores.mean()), 6),
        "cv_macro_f1_std": round(float(cv_scores.std()), 6),
        "holdout_accuracy": round(float(report["accuracy"]), 6),
        "holdout_macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "class_order": class_order,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    return pipeline, metrics


def main():
    dataframe = pd.read_csv(DATA_PATH)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\nTRAINING CATEGORY MODEL...")
    category_model, category_metrics = train_classifier(dataframe, "category")

    print("\nTRAINING PRIORITY MODEL...")
    priority_model, priority_metrics = train_classifier(dataframe, "priority")

    joblib.dump(category_model, MODELS_DIR / "category_model.joblib")
    joblib.dump(priority_model, MODELS_DIR / "priority_model.joblib")

    results = {
        "dataset": {
            "name": "DiaCare Support Message Dataset",
            "samples": int(len(dataframe)),
            "synthetic": True,
            "holdout_samples": int(round(len(dataframe) * 0.2)),
            "holdout_fraction": 0.2,
            "random_seed": SEED,
        },
        "category": category_metrics,
        "priority": priority_metrics,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
