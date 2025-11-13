"""Triage decision engine."""

from typing import Dict, Any, Tuple

from src.config import RISK_THRESHOLD_URGENT, RISK_THRESHOLD_CONSULT, TRIAGE_LABELS
from src.models.risk_scoring import compute_risk_score
from src.llm_interface.llm_parser import generate_explanation


def classify_triage(risk_score: float) -> str:
    """
    Classify risk score into triage category.
    
    Args:
        risk_score: Risk score between 0 and 1
        
    Returns:
        Triage label: "urgent", "consult", or "self_care"
    """
    if risk_score >= RISK_THRESHOLD_URGENT:
        return "urgent"
    elif risk_score >= RISK_THRESHOLD_CONSULT:
        return "consult"
    else:
        return "self_care"


def run_triage(
    parsed_symptoms: Dict[str, Any],
    age: int = None,
    sex: str = None
) -> Tuple[float, str, str]:
    """
    Run complete triage pipeline.
    
    Args:
        parsed_symptoms: Parsed symptom dictionary
        age: Patient age
        sex: Patient sex
        
    Returns:
        Tuple of (risk_score, triage_label, explanation)
    """
    risk_score = compute_risk_score(parsed_symptoms, age, sex)
    triage_label = classify_triage(risk_score)
    
    red_flags = parsed_symptoms.get("red_flags", [])
    explanation = generate_explanation(risk_score, triage_label, parsed_symptoms, red_flags)
    
    return risk_score, triage_label, explanation

