"""Train baseline risk classification model."""

import logging
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.build_dataset import get_train_test_split
from src.config import MODEL_PATH, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


def train_model(model_type: str = "logistic_regression") -> dict:
    """
    Train baseline risk classification model.
    
    Args:
        model_type: "logistic_regression" or "random_forest"
        
    Returns:
        Dictionary with training metrics
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train, X_test, y_train, y_test, feature_names = get_train_test_split()
    
    logger.info(f"Training {model_type} on {len(X_train)} samples")
    
    if model_type == "logistic_regression":
        model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return {
        "model_type": model_type,
        "accuracy": accuracy,
        "n_features": len(feature_names),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model("logistic_regression")

