"""Fit both models on one family-grouped split and evaluate the complete API."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from evaluation import (SEED, calibration_folds, classification_metrics, joint_metrics,
                        load_groups, split_groups)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/messages.csv"
GROUPS_PATH = BASE_DIR / "data/message_groups.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def train_classifier(training, target, folds):
    # Calibrate the entire pipeline: each calibration fit learns its own vocabulary.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20_000, strip_accents="unicode")),
        ("clf", LinearSVC(class_weight="balanced" if target == "priority" else None, random_state=SEED)),
    ])
    model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=folds)
    model.fit(training.text.astype(str), training[target].astype(str))
    return model


def main():
    frame = pd.read_csv(DATA_PATH)
    groups = load_groups(frame, GROUPS_PATH)
    train_index, test_index = split_groups(frame, groups)
    training = frame.iloc[train_index].reset_index(drop=True)
    holdout = frame.iloc[test_index].reset_index(drop=True)
    train_groups = groups.iloc[train_index].reset_index(drop=True)
    folds = calibration_folds(training, train_groups)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Record shared split and calibration membership before fitting.
    manifest = pd.read_csv(GROUPS_PATH)[["row_index", "text_sha256", "family"]]
    manifest["split"] = "train"
    manifest.loc[test_index, "split"] = "holdout"
    manifest["calibration_fold"] = -1
    for fold, (_, valid) in enumerate(folds):
        manifest.loc[train_index[valid], "calibration_fold"] = fold
    manifest.to_csv(RESULTS_DIR / "split_manifest.csv", index=False)

    models = {target: train_classifier(training, target, folds) for target in ("category", "priority")}
    raw = {target: model.predict(holdout.text) for target, model in models.items()}
    for target, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{target}_model.joblib")

    # Exercise actual HTTP route, keyword override, serialization and loaded artifacts.
    import triage_api
    with TestClient(triage_api.app) as client:
        assert client.get("/ready").status_code == 200
        responses = [client.post("/triage", json={"text": text}) for text in holdout.text]
    if any(response.status_code != 200 for response in responses):
        raise RuntimeError("The complete API did not return a prediction for every holdout message")
    api = [response.json() for response in responses]
    api_category = [r["category"] for r in api]
    api_priority = [str(r["priority"]) for r in api]
    model_joint = joint_metrics(holdout.category, holdout.priority, raw["category"], raw["priority"])
    api_joint = joint_metrics(holdout.category, holdout.priority, api_category, api_priority)
    raw_correct = ((raw["category"] == holdout.category.to_numpy()) &
                   (raw["priority"] == holdout.priority.astype(str).to_numpy()))
    api_correct = ((np.asarray(api_category) == holdout.category.to_numpy()) &
                   (np.asarray(api_priority) == holdout.priority.astype(str).to_numpy()))
    results = {
        "dataset": {
            "name": "DiaCare Support Message Dataset", "samples": len(frame), "synthetic": True,
            "families": groups.nunique(), "training_samples": len(training), "holdout_samples": len(holdout),
            "training_families": train_groups.nunique(), "holdout_families": groups.iloc[test_index].nunique(),
            "actual_holdout_fraction": len(holdout) / len(frame), "random_seed": SEED,
            "split_method": "First fold of 5-fold StratifiedGroupKFold, joint category:priority strata; seed 42",
            "calibration_method": "3 family-disjoint folds on training pool, calibrating complete TF-IDF/SVM pipelines",
            "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "group_annotations_sha256": hashlib.sha256(GROUPS_PATH.read_bytes()).hexdigest(),
            "family_overlap": 0, "normalized_text_overlap": 0,
            "limitations": "Manually reviewed broad intent families; not a guarantee every semantic relationship was grouped. Synthetic holdout, not real-world validation.",
        },
        **{target: classification_metrics(holdout[target], raw[target], models[target].classes_) for target in models},
        "model_joint": model_joint,
        "api": {
            "category": classification_metrics(holdout.category, api_category, models["category"].classes_),
            "priority": classification_metrics(holdout.priority, api_priority, models["priority"].classes_),
            "joint": api_joint,
            "keyword_overrides": sum(r["gate_triggered"] for r in api),
            "overrides_fixing_joint_error": int(np.sum(~raw_correct & api_correct)),
            "overrides_breaking_correct_joint_prediction": int(np.sum(raw_correct & ~api_correct)),
            "successful_http_responses": len(api),
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    records = []
    for i, response in enumerate(api):
        records.append({
            "row_index": int(test_index[i]), "family": groups.iloc[test_index[i]], "text": holdout.text.iloc[i],
            "actual_category": holdout.category.iloc[i], "actual_priority": int(holdout.priority.iloc[i]),
            "model_category": raw["category"][i], "model_priority": raw["priority"][i],
            "api_category": response["category"], "api_priority": response["priority"],
            "prediction_source": response["prediction_source"],
            "category_confidence": response["category_confidence"], "priority_confidence": response["priority_confidence"],
            "rule_matches": " | ".join(response["rule_matches"]),
        })
    pd.DataFrame(records).to_csv(RESULTS_DIR / "holdout_predictions.csv", index=False)
    print(json.dumps({"dataset": results["dataset"], "model_joint": model_joint, "api_joint": api_joint}, indent=2))


if __name__ == "__main__":
    main()
