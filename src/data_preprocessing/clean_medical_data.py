"""Clean and preprocess medical data."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_medical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean medical dataset.
    
    Args:
        df: Raw medical dataset
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    df_clean = df_clean.dropna(subset=["age"])
    
    df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce")
    df_clean = df_clean[df_clean["age"].between(0, 120)]
    
    if "sex" in df_clean.columns:
        df_clean["sex"] = df_clean["sex"].str.upper().str.strip()
        df_clean = df_clean[df_clean["sex"].isin(["M", "F", "MALE", "FEMALE"])]
        df_clean["sex"] = df_clean["sex"].map({"M": "M", "F": "F", "MALE": "M", "FEMALE": "F"})
    
    numeric_cols = ["severity", "symptom_count", "red_flag_count", "duration_days"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    logger.info(f"Cleaned dataset: {len(df_clean)} rows remaining")
    return df_clean

