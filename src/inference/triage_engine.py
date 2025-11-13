"""Triage decision engine with layered spectrum-based risk assessment."""

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


def _layer1_critical_life_threatening(raw_text: str) -> float:
    """
    Layer 1: Life-threatening symptoms - highest priority.
    Returns risk score contribution (0.0 to 1.0).
    """
    if not raw_text:
        return 0.0
    
    text_lower = raw_text.lower().strip()
    risk = 0.0
    
    if "dying" in text_lower or "death" in text_lower:
        risk = max(risk, 0.95)
    
    if "heart" in text_lower and any(word in text_lower for word in ["hurt", "pain", "hurting", "hurts", "aching"]):
        risk = max(risk, 0.90)
    
    if "chest" in text_lower and "pain" in text_lower and ("breath" in text_lower or "short" in text_lower):
        risk = max(risk, 0.90)
    
    if ("bleeding" in text_lower or "blood" in text_lower) and ("heart" in text_lower or "chest" in text_lower):
        risk = max(risk, 0.95)
    
    return risk


def _layer2_severe_injuries(raw_text: str, parsed_symptoms: Dict[str, Any]) -> float:
    """
    Layer 2: Severe injuries and trauma.
    Returns risk score contribution (0.0 to 1.0).
    """
    if not raw_text:
        return 0.0
    
    text_lower = raw_text.lower().strip()
    risk = 0.0
    
    injuries = [cat for cat in parsed_symptoms.get("symptom_categories", []) if "fracture" in cat or "dislocation" in cat or "trauma" in cat]
    
    if "broken" in text_lower:
        if "arm" in text_lower or "leg" in text_lower or "foot" in text_lower or "ankle" in text_lower:
            risk = max(risk, 0.85)
        else:
            risk = max(risk, 0.75)
    
    if any(phrase in text_lower for phrase in ["wrong way", "facing wrong", "out of place", "dislocated"]):
        risk = max(risk, 0.80)
    
    if len(injuries) >= 2:
        risk = max(risk, 0.90)
    elif len(injuries) == 1:
        risk = max(risk, 0.75)
    
    if "traumatic_injury" in parsed_symptoms.get("red_flags", []):
        risk = max(risk, 0.80)
    
    if "multiple_injuries" in parsed_symptoms.get("red_flags", []):
        risk = max(risk, 0.90)
    
    return risk


def _layer3_severity_spectrum(parsed_symptoms: Dict[str, Any]) -> float:
    """
    Layer 3: Severity spectrum analysis.
    Returns risk score contribution (0.0 to 1.0).
    """
    severity = parsed_symptoms.get("severity", 5.0)
    
    if severity >= 9.5:
        return 0.95
    elif severity >= 9.0:
        return 0.90
    elif severity >= 8.5:
        return 0.85
    elif severity >= 8.0:
        return 0.75
    elif severity >= 7.0:
        return 0.65
    elif severity >= 6.0:
        return 0.50
    elif severity >= 5.0:
        return 0.35
    elif severity >= 4.0:
        return 0.25
    else:
        return 0.15


def _layer4_red_flags(parsed_symptoms: Dict[str, Any]) -> float:
    """
    Layer 4: Red flags analysis.
    Returns risk score contribution (0.0 to 1.0).
    """
    red_flags = parsed_symptoms.get("red_flags", [])
    
    if not red_flags:
        return 0.0
    
    critical_flags = ["severe_chest_pain", "difficulty_breathing", "loss_of_consciousness", "critical_severity", "active_bleeding"]
    severe_flags = ["fracture", "dislocation", "traumatic_injury", "multiple_injuries"]
    
    risk = 0.0
    
    if any(flag in critical_flags for flag in red_flags):
        risk = max(risk, 0.85)
    
    if any(flag in severe_flags for flag in red_flags):
        risk = max(risk, 0.70)
    
    if len(red_flags) >= 3:
        risk = max(risk, 0.90)
    elif len(red_flags) >= 2:
        risk = max(risk, 0.75)
    elif len(red_flags) >= 1:
        risk = max(risk, 0.50)
    
    return risk


def _layer5_symptom_combinations(parsed_symptoms: Dict[str, Any]) -> float:
    """
    Layer 5: Symptom combination analysis.
    Returns risk score contribution (0.0 to 1.0).
    """
    symptom_categories = parsed_symptoms.get("symptom_categories", [])
    
    if "chest_pain" in symptom_categories and "shortness_of_breath" in symptom_categories:
        return 0.90
    
    if "trauma" in symptom_categories and len(symptom_categories) >= 3:
        return 0.80
    
    if len(symptom_categories) >= 4:
        return 0.60
    elif len(symptom_categories) >= 3:
        return 0.45
    elif len(symptom_categories) >= 2:
        return 0.30
    
    return 0.15


def _compute_spectrum_risk_score(
    raw_text: str,
    parsed_symptoms: Dict[str, Any]
) -> float:
    """
    Compute risk score using layered spectrum approach.
    
    Combines multiple layers with weighted contributions.
    
    Args:
        raw_text: Raw symptom text
        parsed_symptoms: Parsed symptom dictionary
        
    Returns:
        Risk score between 0 and 1
    """
    layer1 = _layer1_critical_life_threatening(raw_text)
    layer2 = _layer2_severe_injuries(raw_text, parsed_symptoms)
    layer3 = _layer3_severity_spectrum(parsed_symptoms)
    layer4 = _layer4_red_flags(parsed_symptoms)
    layer5 = _layer5_symptom_combinations(parsed_symptoms)
    
    if layer1 >= 0.90:
        return layer1
    
    if layer2 >= 0.80:
        return layer2
    
    combined = max(layer1, layer2, layer3, layer4, layer5)
    
    if combined >= 0.75:
        return min(combined + 0.10, 0.95)
    
    return combined


def run_triage(
    parsed_symptoms: Dict[str, Any],
    age: int = None,
    sex: str = None,
    raw_text: str = None
) -> Tuple[float, str, str]:
    """
    Run complete triage pipeline with layered spectrum-based assessment.
    
    Args:
        parsed_symptoms: Parsed symptom dictionary
        age: Patient age
        sex: Patient sex
        raw_text: Raw symptom text (REQUIRED for accurate assessment)
        
    Returns:
        Tuple of (risk_score, triage_label, explanation)
    """
    if not raw_text:
        raw_text = parsed_symptoms.get("raw_text", "")
    
    risk_score = _compute_spectrum_risk_score(raw_text, parsed_symptoms)
    
    triage_label = classify_triage(risk_score)
    red_flags = parsed_symptoms.get("red_flags", [])
    explanation = generate_explanation(risk_score, triage_label, parsed_symptoms, red_flags)
    
    return risk_score, triage_label, explanation
