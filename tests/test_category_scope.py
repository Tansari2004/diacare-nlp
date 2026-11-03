import numpy as np
import pytest
from fastapi.testclient import TestClient

import triage_api


class FixedCategory:
    classes_ = np.array(["billing", "medical"])

    def __init__(self, confidence):
        self.confidence = confidence

    def predict_proba(self, _):
        return np.array([[self.confidence, 1 - self.confidence]])


@pytest.mark.parametrize("text", [
    "I do not have chest pain; I need a receipt.",
    "Could this be a billing or appointment problem?",
    "Please cancel my visit and refund the duplicate payment.",
    "The whatchamacallit is behaving strangely.",
    "I have chest pain.",
])
def test_every_message_leaves_priority_to_person(monkeypatch, text):
    with TestClient(triage_api.app) as client:
        monkeypatch.setattr(triage_api, "cat_model", FixedCategory(.9))
        # Category-only operation has no dependency on the priority artifact.
        monkeypatch.setattr(triage_api, "pri_model", None)
        result = client.post("/classify", json={"text": text})
        assert result.status_code == 200
        body = result.json()
        assert body["priority"] is None
        assert body["priority_decision"] == "manual_review_required"
        assert body["human_review_required"] is True
        assert "gate_triggered" not in body


def test_low_score_abstains_from_category(monkeypatch):
    monkeypatch.setattr(triage_api, "cat_model", FixedCategory(.55))
    result = triage_api.classify(triage_api.TriageRequest(text="A question"))
    assert result["category"] is None
    assert result["suggested_category"] == "billing"
    assert result["category_decision"] == "needs_review"


def test_category_endpoint_rejects_invalid_input_and_unavailable_model(monkeypatch):
    with TestClient(triage_api.app) as client:
        for text in ["", "   ", "x" * 5001]:
            assert client.post("/classify", json={"text": text}).status_code == 422
        monkeypatch.setattr(triage_api, "cat_model", None)
        assert client.post("/classify", json={"text": "A question"}).status_code == 503
