"""Tests for feature engineering."""

import pytest
from src.data_preprocessing.create_clinical_features import create_feature_vector, feature_vector_to_array


def test_create_feature_vector_structure():
    """Test feature vector has expected keys."""
    parsed_symptoms = {
        "symptom_categories": ["chest_pain", "fever"],
        "severity": 7.5,
        "duration_days": 5,
        "red_flags": ["severe_chest_pain"],
        "pattern": "progressive"
    }
    
    features = create_feature_vector(parsed_symptoms, age=45, sex="M")
    
    assert "symptom_count" in features
    assert "severity_score" in features
    assert "red_flag_binary" in features
    assert "age" in features
    assert "sex_encoded" in features
    assert features["red_flag_binary"] == 1.0
    assert features["symptom_count"] == 2.0


def test_feature_vector_to_array():
    """Test conversion to numpy array."""
    feature_dict = {
        "symptom_count": 2.0,
        "severity_score": 0.75,
        "age": 45.0
    }
    
    feature_order = ["age", "severity_score", "symptom_count"]
    array = feature_vector_to_array(feature_dict, feature_order)
    
    assert array.shape == (3,)
    assert array[0] == 45.0
    assert array[1] == 0.75
    assert array[2] == 2.0

