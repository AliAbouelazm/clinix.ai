"""Create clinical features from parsed symptoms and demographics."""

import numpy as np
from typing import Dict, Any, List

from src.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def create_feature_vector(
    parsed_symptoms: Dict[str, Any],
    age: int = None,
    sex: str = None
) -> Dict[str, float]:
    """
    Create numeric feature vector from parsed symptoms and demographics.
    
    Args:
        parsed_symptoms: Parsed symptom dictionary
        age: Patient age
        sex: Patient sex (M/F)
        
    Returns:
        Dictionary of feature names and values
    """
    symptom_categories = parsed_symptoms.get("symptom_categories", [])
    severity = parsed_symptoms.get("severity", 0.0)
    duration_days = parsed_symptoms.get("duration_days", 0)
    red_flags = parsed_symptoms.get("red_flags", [])
    pattern = parsed_symptoms.get("pattern", "constant")
    
    features = {}
    
    features["symptom_count"] = float(len(symptom_categories))
    features["severity_score"] = float(severity) / 10.0
    features["red_flag_binary"] = 1.0 if len(red_flags) > 0 else 0.0
    features["red_flag_count"] = float(len(red_flags))
    features["duration_days"] = float(duration_days)
    features["duration_normalized"] = min(float(duration_days) / 30.0, 1.0)
    
    pattern_encoding = {
        "intermittent": 0.25,
        "constant": 0.5,
        "progressive": 0.75,
        "acute": 1.0
    }
    features["pattern_encoded"] = pattern_encoding.get(pattern, 0.5)
    
    if age is not None:
        features["age"] = float(age)
        features["age_normalized"] = min(float(age) / 100.0, 1.0)
    else:
        features["age"] = 50.0
        features["age_normalized"] = 0.5
    
    if sex:
        features["sex_encoded"] = 1.0 if sex.upper() in ["M", "MALE"] else 0.0
    else:
        features["sex_encoded"] = 0.5
    
    symptom_category_features = _encode_symptom_categories(symptom_categories)
    features.update(symptom_category_features)
    
    return features


def _encode_symptom_categories(categories: List[str]) -> Dict[str, float]:
    """Encode symptom categories as binary features."""
    common_symptoms = [
        "chest_pain",
        "shortness_of_breath",
        "fever",
        "headache",
        "abdominal_pain",
        "nausea",
        "dizziness",
        "fatigue"
    ]
    
    features = {}
    for symptom in common_symptoms:
        features[f"symptom_{symptom}"] = 1.0 if symptom in categories else 0.0
    
    return features


def feature_vector_to_array(feature_dict: Dict[str, float], feature_order: List[str] = None) -> np.ndarray:
    """
    Convert feature dictionary to numpy array.
    
    Args:
        feature_dict: Dictionary of features
        feature_order: Optional list specifying feature order
        
    Returns:
        Numpy array of features
    """
    if feature_order is None:
        feature_order = sorted(feature_dict.keys())
    
    return np.array([feature_dict.get(f, 0.0) for f in feature_order])

