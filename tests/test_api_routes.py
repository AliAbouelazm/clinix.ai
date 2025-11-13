"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from src.api.fastapi_app import app
from src.database.db_utils import init_schema

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database before each test."""
    init_schema()


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_triage_endpoint():
    """Test triage endpoint."""
    request_data = {
        "age": 35,
        "sex": "M",
        "symptom_text": "I have mild headache for 2 days"
    }
    
    response = client.post("/triage", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "patient_id" in data
    assert "risk_score" in data
    assert "triage_label" in data
    assert "explanation" in data
    assert 0.0 <= data["risk_score"] <= 1.0


def test_patient_history_endpoint():
    """Test patient history endpoint."""
    request_data = {
        "age": 40,
        "sex": "F",
        "symptom_text": "Chest pain"
    }
    
    triage_response = client.post("/triage", json=request_data)
    patient_id = triage_response.json()["patient_id"]
    
    history_response = client.get(f"/patient/{patient_id}/history")
    
    assert history_response.status_code == 200
    data = history_response.json()
    assert "patient_id" in data
    assert "history" in data
    assert len(data["history"]) > 0

