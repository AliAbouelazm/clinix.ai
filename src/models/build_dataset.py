"""Build training datasets from medical data."""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.data_preprocessing.load_medical_data import load_medical_dataset
from src.data_preprocessing.clean_medical_data import clean_medical_data
from src.data_preprocessing.create_clinical_features import create_feature_vector, feature_vector_to_array
from src.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def build_training_dataset() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build training dataset from medical data.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    df = load_medical_dataset()
    df_clean = clean_medical_data(df)
    
    feature_vectors = []
    labels = []
    
    for _, row in df_clean.iterrows():
        parsed_symptoms = {
            "symptom_categories": ["symptom_" + str(i) for i in range(int(row.get("symptom_count", 1)))],
            "severity": float(row.get("severity", 5.0)),
            "duration_days": int(row.get("duration_days", 3)),
            "red_flags": ["flag_" + str(i) for i in range(int(row.get("red_flag_count", 0)))],
            "pattern": "progressive" if row.get("red_flag_count", 0) > 0 else "constant"
        }
        
        features = create_feature_vector(
            parsed_symptoms,
            age=int(row.get("age", 50)),
            sex=str(row.get("sex", "M"))
        )
        
        feature_vectors.append(features)
        labels.append(int(row.get("risk_label", 0)))
    
    feature_order = sorted(feature_vectors[0].keys())
    X = np.array([feature_vector_to_array(fv, feature_order) for fv in feature_vectors])
    y = np.array(labels)
    
    return X, y, feature_order


def get_train_test_split(test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Get train/test split of dataset.
    
    Args:
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    X, y, feature_names = build_training_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names

