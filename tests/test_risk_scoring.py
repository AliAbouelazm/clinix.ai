"""Tests for risk scoring."""

import pytest
from src.models.risk_scoring import compute_risk_score
from pathlib import Path
from src.config import MODEL_PATH


def test_risk_score_range():
    """Test that risk score is between 0 and 1."""
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet")
    
    parsed_symptoms = {
        "symptom_categories": ["chest_pain"],
        "severity": 5.0,
        "duration_days": 3,
        "red_flags": [],
        "pattern": "constant"
    }
    
    risk_score = compute_risk_score(parsed_symptoms, age=50, sex="M")
    
    assert 0.0 <= risk_score <= 1.0


def test_risk_score_with_red_flags():
    """Test that red flags increase risk score."""
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet")
    
    parsed_symptoms_low = {
        "symptom_categories": ["headache"],
        "severity": 3.0,
        "duration_days": 1,
        "red_flags": [],
        "pattern": "constant"
    }
    
    parsed_symptoms_high = {
        "symptom_categories": ["chest_pain"],
        "severity": 9.0,
        "duration_days": 1,
        "red_flags": ["severe_chest_pain"],
        "pattern": "acute"
    }
    
    risk_low = compute_risk_score(parsed_symptoms_low, age=30, sex="M")
    risk_high = compute_risk_score(parsed_symptoms_high, age=30, sex="M")
    
    assert risk_high >= risk_low

