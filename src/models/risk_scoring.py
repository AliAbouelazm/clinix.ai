"""Risk scoring utilities."""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.config import MODEL_PATH, MODELS_DIR
from src.data_preprocessing.create_clinical_features import create_feature_vector, feature_vector_to_array


def load_model():
    """Load trained risk classification model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    return joblib.load(MODEL_PATH)


def load_feature_names() -> list:
    """Load feature names used during training."""
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    return joblib.load(feature_names_path)


def compute_risk_score(
    parsed_symptoms: Dict[str, Any],
    age: int = None,
    sex: str = None
) -> float:
    """
    Compute risk score from parsed symptoms and demographics.
    
    Args:
        parsed_symptoms: Parsed symptom dictionary
        age: Patient age
        sex: Patient sex
        
    Returns:
        Risk score between 0 and 1
    """
    model = load_model()
    feature_names = load_feature_names()
    
    feature_vector = create_feature_vector(parsed_symptoms, age, sex)
    X = feature_vector_to_array(feature_vector, feature_names).reshape(1, -1)
    
    if hasattr(model, "predict_proba"):
        risk_score = model.predict_proba(X)[0][1]
    else:
        risk_score = float(model.predict(X)[0])
    
    return float(risk_score)

