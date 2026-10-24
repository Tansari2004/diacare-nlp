import numpy as np
from fastapi.testclient import TestClient

import triage_api


def test_emergency_gate_reports_only_matched_phrases():
    assert triage_api.emergency_matches("I have chest pain and feel faint.") == ["chest pain", "faint"]


def test_regular_message_does_not_trigger_emergency_gate():
    assert triage_api.emergency_matches("I need help changing my appointment.") == []


def test_loaded_models_return_separate_confidences():
    triage_api.load_models()
    result = triage_api.triage(triage_api.TriageRequest(text="I was charged twice for my subscription."))
    assert result["category"] == "billing"
    assert 1 <= result["priority"] <= 5
    assert 0 <= result["category_confidence"] <= 1
    assert 0 <= result["priority_confidence"] <= 1
    assert "confidence" not in result
    assert result["prediction_source"] == "models"
    assert result["rule_matches"] == []


def test_api_readiness_and_explicit_rule_response():
    with TestClient(triage_api.app) as client:
        assert client.get("/health").json()["models_loaded"] is True
        assert client.get("/ready").status_code == 200
        result = client.post("/triage", json={"text": "I have chest pain and feel faint."})
        assert result.status_code == 200
        data = result.json()
        assert data["priority"] == 5 and data["category"] == "medical"
        assert data["rule_matches"] == ["chest pain", "faint"]
        assert data["prediction_source"] == "keyword_rule"
        assert data["category_confidence"] is None and data["priority_confidence"] is None
        assert "confidence" not in data


def test_uncertain_priority_is_not_hidden_by_category_confidence(monkeypatch):
    class FixedModel:
        def __init__(self, classes, probs):
            self.classes_ = np.array(classes)
            self.probs = np.array([probs])
        def predict_proba(self, _):
            return self.probs
    monkeypatch.setattr(triage_api, "cat_model", FixedModel(["billing", "other"], [.90, .10]))
    monkeypatch.setattr(triage_api, "pri_model", FixedModel(["1", "2", "3", "4", "5"], [.21, .22, .20, .19, .18]))
    result = triage_api.triage(triage_api.TriageRequest(text="A billing question"))
    assert result["category_confidence"] == .90
    assert result["priority_confidence"] == .22
    assert result["priority"] == 2


def test_missing_models_fail_readiness_and_all_prediction_paths(monkeypatch):
    with TestClient(triage_api.app) as client:
        monkeypatch.setattr(triage_api, "cat_model", None)
        monkeypatch.setattr(triage_api, "pri_model", None)
        health = client.get("/health")
        assert health.status_code == 200  # Process liveness is separate from readiness.
        assert health.json()["models_loaded"] is False
        assert client.get("/ready").status_code == 503
        for text in ["A billing question", "I have chest pain"]:
            assert client.post("/triage", json={"text": text}).status_code == 503


def test_load_failure_does_not_leave_one_stale_model(monkeypatch):
    monkeypatch.setattr(triage_api, "cat_model", object())
    monkeypatch.setattr(triage_api, "pri_model", object())
    def failed_load(_):
        raise OSError("Missing artifact")
    monkeypatch.setattr(triage_api, "_load_joblib", failed_load)
    triage_api.load_models()
    assert triage_api.cat_model is None and triage_api.pri_model is None


def test_blank_message_rejected():
    with TestClient(triage_api.app) as client:
        assert client.post("/triage", json={"text": "   "}).status_code == 422
