from fastapi.testclient import TestClient

import triage_api


def test_emergency_gate_reports_only_matched_phrases():
    matches = triage_api.emergency_matches("I have chest pain and feel faint.")
    assert matches == ["chest pain", "faint"]


def test_regular_message_does_not_trigger_emergency_gate():
    assert triage_api.emergency_matches("I need help changing my appointment.") == []


def test_loaded_models_return_a_billing_prediction():
    triage_api.load_models()
    response = triage_api.triage(
        triage_api.TriageRequest(text="I was charged twice for my subscription.")
    )

    assert response["category"] == "billing"
    assert 1 <= response["priority"] <= 5
    assert 0 <= response["confidence"] <= 1
    assert response["gate_triggered"] is False


def test_api_health_and_emergency_response():
    with TestClient(triage_api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["models_loaded"] is True

        result = client.post(
            "/triage", json={"text": "I have chest pain and feel faint."}
        )
        assert result.status_code == 200
        assert result.json()["priority"] == 5
        assert result.json()["reason_phrases"] == ["chest pain", "faint"]
