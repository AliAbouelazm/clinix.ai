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
    if not raw_text:
        raw_text = parsed_symptoms.get("raw_text", "")
    
    text_lower = (raw_text or "").lower().strip()
    
    if not text_lower:
        return False
    
    symptom_categories = parsed_symptoms.get("symptom_categories", [])
    red_flags = parsed_symptoms.get("red_flags", [])
    severity = parsed_symptoms.get("severity", 0)
    
    if "dying" in text_lower:
        return True
    
    if "death" in text_lower and ("feel" in text_lower or "im" in text_lower or "i'm" in text_lower or "i am" in text_lower):
        return True
    
    if "heart" in text_lower and ("hurt" in text_lower or "pain" in text_lower or "hurting" in text_lower or "hurts" in text_lower):
        return True
    
    if severity >= 9.0:
        return True
    
    if len(red_flags) >= 1 and severity >= 7.0:
        return True
    
    if len(red_flags) >= 2:
        return True
    
    if "chest_pain" in symptom_categories and "shortness_of_breath" in symptom_categories:
        return True
    
    if any(flag in ["severe_chest_pain", "difficulty_breathing", "loss_of_consciousness", "critical_severity"] for flag in red_flags):
        return True
    
    if any(phrase in text_lower for phrase in ["can't breathe", "can't breath", "cant breathe", "struggling to breathe"]):
        if "chest" in text_lower or "heart" in text_lower:
            return True
    
    if ("bleeding" in text_lower or "blood" in text_lower) and severity >= 7.0:
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
    if not raw_text:
        raw_text = parsed_symptoms.get("raw_text", "")
    
    text_lower = (raw_text or "").lower().strip()
    
    if text_lower:
        if "dying" in text_lower or "im dying" in text_lower or "i'm dying" in text_lower:
            return (0.95, "urgent", "CRITICAL: The phrase 'dying' was detected in your symptoms. This requires immediate medical attention. Please seek emergency care immediately.")
        
        if "bleeding" in text_lower or "blood" in text_lower:
            if "heart" in text_lower or "chest" in text_lower or "pain" in text_lower:
                return (0.95, "urgent", "CRITICAL: Bleeding combined with heart/chest symptoms requires immediate medical evaluation. Please seek emergency care immediately.")
            if any(word in text_lower for word in ["heavy", "lot", "much", "severe", "bad"]):
                return (0.95, "urgent", "CRITICAL: Significant bleeding detected. This requires immediate medical attention. Please seek emergency care immediately.")
        
        if "heart" in text_lower and any(word in text_lower for word in ["hurt", "pain", "hurting", "hurts", "aching", "ache"]):
            return (0.95, "urgent", "CRITICAL: Heart pain or discomfort requires immediate medical evaluation. Please seek emergency care immediately.")
        
        if "chest" in text_lower and "pain" in text_lower and ("breath" in text_lower or "short" in text_lower):
            return (0.95, "urgent", "CRITICAL: Chest pain with breathing difficulties is a medical emergency. Please seek emergency care immediately.")
        
        if "heart" in text_lower and ("bleeding" in text_lower or "blood" in text_lower):
            return (0.95, "urgent", "CRITICAL: Heart symptoms combined with bleeding is a medical emergency. Please seek emergency care immediately.")
    
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


