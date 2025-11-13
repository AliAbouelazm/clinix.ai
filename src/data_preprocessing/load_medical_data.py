"""Load medical datasets for training."""

import pandas as pd
import logging
from pathlib import Path

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def load_medical_dataset(filename: str = "example_medical_dataset.csv") -> pd.DataFrame:
    """
    Load medical dataset from raw data directory.
    
    Args:
        filename: Name of CSV file
        
    Returns:
        DataFrame with medical data
    """
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        logger.warning(f"Dataset file not found: {filepath}, creating sample data")
        return _create_sample_dataset()
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return _create_sample_dataset()


def _create_sample_dataset() -> pd.DataFrame:
    """Create sample medical dataset for demonstration."""
    import numpy as np
    from src.config import RANDOM_SEED
    
    np.random.seed(RANDOM_SEED)
    n_samples = 500
    
    data = {
        "age": np.random.randint(18, 80, n_samples),
        "sex": np.random.choice(["M", "F"], n_samples),
        "symptom_count": np.random.randint(1, 5, n_samples),
        "severity": np.random.uniform(1, 10, n_samples),
        "red_flag_count": np.random.randint(0, 3, n_samples),
        "duration_days": np.random.randint(1, 30, n_samples),
        "risk_label": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample dataset with {len(df)} rows")
    return df

