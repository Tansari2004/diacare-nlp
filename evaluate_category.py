"""Retrospective category comparison on the already observed, frozen holdout.

No tuning or resplitting. Fits simple baselines only on the original training
pool; exercises /classify with the existing category model and fixed threshold.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from evaluation import classification_metrics, load_groups, split_groups
import triage_api

ROOT = Path(__file__).resolve().parent


def main():
    frame = pd.read_csv(ROOT / "data/messages.csv")
    groups = load_groups(frame, ROOT / "data/message_groups.csv")
    train, test = split_groups(frame, groups)
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    assert manifest.index[manifest.split.eq("holdout")].tolist() == test.tolist()
    training, holdout = frame.iloc[train], frame.iloc[test]
    labels = sorted(frame.category.unique())
    baselines = {
        "majority_training_category": DummyClassifier(strategy="most_frequent"),
        "tfidf_multinomial_nb": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB()),
    }
    results = {"scope": "Category suggestion with manual priority review",
               "evaluation_status": "Retrospective diagnostic on an already observed synthetic holdout; not a fresh final test",
               "holdout_rows": len(test), "baseline_training_rows": len(train),
               "threshold": triage_api.CATEGORY_REVIEW_THRESHOLD,
               "threshold_provenance": "Fixed engineering default, not selected from this holdout; confidence calibration not established",
               "independent_annotation_review": "pending", "fresh_final_evaluation": "pending",
               "baselines": {}}
    for name, baseline in baselines.items():
        baseline.fit(training.text, training.category)
        predictions = baseline.predict(holdout.text)
        results["baselines"][name] = classification_metrics(holdout.category, predictions, labels)
    with TestClient(triage_api.app) as client:
        responses = [client.post("/classify", json={"text": text}) for text in holdout.text]
    assert all(response.status_code == 200 for response in responses)
    rows = [response.json() for response in responses]
    accepted = np.array([row["category"] is not None for row in rows])
    correct = np.array([row["suggested_category"] == actual for row, actual in zip(rows, holdout.category)])
    results["category_only_api"] = {
        "all_suggestions": classification_metrics(holdout.category, [row["suggested_category"] for row in rows], labels),
        "category_suggestions_accepted": int(accepted.sum()),
        "category_review_required": int((~accepted).sum()),
        "category_coverage": float(accepted.mean()),
        "accepted_category_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "accepted_category_errors": int((accepted & ~correct).sum()),
        "manual_priority_reviews": len(rows),
        "automatic_priority_decisions": sum(row["priority"] is not None for row in rows),
    }
    output = ROOT / "results/category_only_metrics.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame([{"row_index": int(index), "actual_category": actual, **row}
                  for index, actual, row in zip(test, holdout.category, rows)]).to_csv(
        ROOT / "results/category_only_predictions.csv", index=False)
    print(json.dumps({"category_only_api": results["category_only_api"],
                      "baselines": {name: value["holdout_accuracy"] for name, value in results["baselines"].items()}}, indent=2))


if __name__ == "__main__":
    main()
