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


def _check_critical_symptoms(parsed_symptoms: Dict[str, Any], raw_text: str = None) -> bool:
    """
    Check for critical symptoms that require immediate urgent care.
    
    Args:
        parsed_symptoms: Parsed symptom dictionary
        raw_text: Optional raw symptom text for additional checks
        
    Returns:
        True if critical symptoms detected
    """
    symptom_categories = parsed_symptoms.get("symptom_categories", [])
    red_flags = parsed_symptoms.get("red_flags", [])
    severity = parsed_symptoms.get("severity", 0)
    
    if not raw_text:
        raw_text = parsed_symptoms.get("raw_text", "")
    
    critical_symptoms = [
        "chest_pain", "heart", "cardiac", "shortness_of_breath",
        "bleeding", "hemorrhage", "unconscious", "severe_chest_pain"
    ]
    
    text_lower = (raw_text or "").lower()
    
    if severity >= 9.0:
        return True
    
    if len(red_flags) >= 2:
        return True
    
    if "chest_pain" in symptom_categories and "shortness_of_breath" in symptom_categories:
        return True
    
    if any(flag in ["severe_chest_pain", "difficulty_breathing", "loss_of_consciousness"] for flag in red_flags):
        return True
    
    if any(word in text_lower for word in ["dying", "death", "can't breathe", "can't breath", "bleeding", "blood"]):
        if "chest" in text_lower or "heart" in text_lower or "breath" in text_lower:
            return True
    
    if "bleeding" in text_lower or "blood" in text_lower:
        if severity >= 7.0:
            return True
    
    return False


def run_triage(
    parsed_symptoms: Dict[str, Any],
    age: int = None,
    sex: str = None,
    raw_text: str = None
) -> Tuple[float, str, str]:
    """
    Run complete triage pipeline.
    
    Args:
        parsed_symptoms: Parsed symptom dictionary
        age: Patient age
        sex: Patient sex
        raw_text: Optional raw symptom text for critical symptom checks
        
    Returns:
        Tuple of (risk_score, triage_label, explanation)
    """
    is_critical = _check_critical_symptoms(parsed_symptoms, raw_text)
    
    if is_critical:
        risk_score = 0.95
        triage_label = "urgent"
        red_flags = parsed_symptoms.get("red_flags", [])
        if not red_flags:
            red_flags = ["critical_symptoms_detected"]
        explanation = f"CRITICAL: Based on the symptoms described (chest pain, shortness of breath, bleeding, or other severe indicators), immediate medical attention is required. This case has been automatically classified as urgent. Please seek emergency care immediately."
    else:
        risk_score = compute_risk_score(parsed_symptoms, age, sex)
        triage_label = classify_triage(risk_score)
        red_flags = parsed_symptoms.get("red_flags", [])
        explanation = generate_explanation(risk_score, triage_label, parsed_symptoms, red_flags)
    
    return risk_score, triage_label, explanation


