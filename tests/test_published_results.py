import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_published_rows_and_split_match_the_pinned_source():
    results = json.loads((ROOT / "results/metrics.json").read_text())
    source = list(csv.DictReader((ROOT / "data/messages.csv").open()))
    manifest = list(csv.DictReader((ROOT / "results/split_manifest.csv").open()))
    predictions = list(csv.DictReader((ROOT / "results/holdout_predictions.csv").open()))
    assert results["dataset"]["source_sha256"] == hashlib.sha256((ROOT / "data/messages.csv").read_bytes()).hexdigest()
    assert results["dataset"]["group_annotations_sha256"] == hashlib.sha256((ROOT / "data/message_groups.csv").read_bytes()).hexdigest()
    expected = {r["row_index"] for r in manifest if r["split"] == "holdout"}
    assert {r["row_index"] for r in predictions} == expected
    for row in predictions:
        actual = source[int(row["row_index"])]
        assert row["text"] == actual["text"]
        assert row["actual_category"] == actual["category"]
        assert row["actual_priority"] == actual["priority"]


def test_published_joint_and_individual_counts_reconcile():
    results = json.loads((ROOT / "results/metrics.json").read_text())
    rows = list(csv.DictReader((ROOT / "results/holdout_predictions.csv").open()))
    for system in ("model", "api"):
        for target in ("category", "priority"):
            correct = sum(r[f"{system}_{target}"] == r[f"actual_{target}"] for r in rows)
            metric = results[target] if system == "model" else results["api"][target]
            assert metric["correct"] == correct
            assert metric["samples"] == len(rows)
        correct_both = sum(all(r[f"{system}_{t}"] == r[f"actual_{t}"] for t in ("category", "priority")) for r in rows)
        joint = results["model_joint"] if system == "model" else results["api"]["joint"]
        assert joint["correct_both"] == correct_both
        assert joint["accuracy"] == correct_both / len(rows)
