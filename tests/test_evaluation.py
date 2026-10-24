from pathlib import Path

import pandas as pd
import pytest

from evaluation import (calibration_folds, joint_metrics, load_groups, normalize_text, split_groups)

ROOT = Path(__file__).resolve().parents[1]


def source():
    frame = pd.read_csv(ROOT / "data/messages.csv")
    groups = load_groups(frame, ROOT / "data/message_groups.csv")
    return frame, groups


def test_shared_holdout_has_no_family_or_normalized_duplicate_overlap():
    frame, groups = source()
    train, test = split_groups(frame, groups)
    assert len(train) == 490 and len(test) == 140
    assert set(train).isdisjoint(test) and len(set(train) | set(test)) == len(frame)
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    assert set(frame.text.iloc[train].map(normalize_text)).isdisjoint(frame.text.iloc[test].map(normalize_text))
    assert split_groups(frame, groups)[1].tolist() == test.tolist()


def test_calibration_families_never_cross_training_and_validation():
    frame, groups = source()
    train, _ = split_groups(frame, groups)
    frame = frame.iloc[train].reset_index(drop=True)
    groups = groups.iloc[train].reset_index(drop=True)
    folds = calibration_folds(frame, groups)
    validation_rows = []
    for fitting, valid in folds:
        assert set(groups.iloc[fitting]).isdisjoint(groups.iloc[valid])
        validation_rows.extend(valid.tolist())
    assert sorted(validation_rows) == list(range(len(frame)))


def test_changed_source_requires_reannotation():
    frame, _ = source()
    frame.loc[0, "text"] = "Changed source message"
    with pytest.raises(ValueError, match="Source changed"):
        load_groups(frame, ROOT / "data/message_groups.csv")


def test_duplicate_cannot_be_assigned_to_another_family(tmp_path):
    frame, _ = source()
    annotation = pd.read_csv(ROOT / "data/message_groups.csv")
    duplicate = frame.text.duplicated()
    annotation.loc[duplicate[duplicate].index[0], "family"] = "different-family"
    path = tmp_path / "groups.csv"
    annotation.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate texts cross"):
        load_groups(frame, path)


def test_joint_accuracy_requires_both_answers_on_the_same_row():
    result = joint_metrics(["billing", "medical"], ["1", "5"], ["billing", "other"], ["2", "5"])
    assert result == {"correct_both": 0, "samples": 2, "accuracy": 0.0}
