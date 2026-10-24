"""Shared, family-disjoint evaluation and auditable split construction."""
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42


def normalize_text(text):
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def load_groups(frame, path):
    annotations = pd.read_csv(path, keep_default_na=False)
    if len(annotations) != len(frame) or annotations.row_index.tolist() != list(range(len(frame))):
        raise ValueError("Group annotations must cover every source row in order")
    hashes = frame.text.map(lambda text: hashlib.sha256(str(text).encode()).hexdigest())
    if (hashes.tolist() != annotations.text_sha256.tolist() or
            frame.text.astype(str).tolist() != annotations.text.astype(str).tolist()):
        raise ValueError("Source changed: review the family annotations before splitting")
    if annotations.family.eq("").any():
        raise ValueError("Every message needs a reviewed family")
    audit = pd.DataFrame({"normalized": frame.text.map(normalize_text), "family": annotations.family})
    if audit.groupby("normalized").family.nunique().max() != 1:
        raise ValueError("Normalized duplicate texts cross family annotations")
    return annotations.family


def split_groups(frame, groups):
    # First fold, fixed before model fitting; no search over seeds or performance.
    strata = frame.category.astype(str) + ":" + frame.priority.astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(splitter.split(frame.text, strata, groups))
    if set(groups.iloc[train]) & set(groups.iloc[test]):
        raise ValueError("Family overlap in shared holdout")
    if set(frame.text.iloc[train].map(normalize_text)) & set(frame.text.iloc[test].map(normalize_text)):
        raise ValueError("Duplicate overlap in shared holdout")
    for target in ("category", "priority"):
        if set(frame[target].iloc[train]) != set(frame[target]):
            raise ValueError(f"Training split is missing a {target} class")
    return train, test


def calibration_folds(frame, groups):
    strata = frame.category.astype(str) + ":" + frame.priority.astype(str)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(splitter.split(frame.text, strata, groups))
    for train, valid in folds:
        if set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise ValueError("Family overlap in calibration folds")
        for target in ("category", "priority"):
            if (set(frame[target].iloc[train]) != set(frame[target]) or
                    set(frame[target].iloc[valid]) != set(frame[target])):
                raise ValueError(f"Calibration fold missing {target} class")
    return folds


def classification_metrics(actual, predicted, classes):
    actual = np.asarray(actual).astype(str)
    predicted = np.asarray(predicted).astype(str)
    classes = [str(c) for c in classes]
    report = classification_report(actual, predicted, labels=classes, output_dict=True, zero_division=0)
    return {
        "correct": int(np.sum(actual == predicted)), "samples": len(actual),
        "holdout_accuracy": float(np.mean(actual == predicted)),
        "holdout_macro_f1": report["macro avg"]["f1-score"], "class_order": classes,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
    }


def joint_metrics(actual_category, actual_priority, predicted_category, predicted_priority):
    correct = ((np.asarray(actual_category).astype(str) == np.asarray(predicted_category).astype(str)) &
               (np.asarray(actual_priority).astype(str) == np.asarray(predicted_priority).astype(str)))
    return {"correct_both": int(correct.sum()), "samples": len(correct), "accuracy": float(correct.mean())}
