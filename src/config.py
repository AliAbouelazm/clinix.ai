"""Configuration constants and settings for clinix.ai."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DB_PATH = DATA_DIR / "clinic.db"
DB_URL = f"sqlite:///{DB_PATH}"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MODEL_PATH = MODELS_DIR / "risk_classifier.pkl"

RANDOM_SEED = 42

RISK_THRESHOLD_URGENT = 0.8
RISK_THRESHOLD_CONSULT = 0.4

TRIAGE_LABELS = {
    "urgent": "Seek care now",
    "consult": "Consult GP",
    "self_care": "Monitor at home"
}

