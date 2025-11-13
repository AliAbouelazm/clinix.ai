"""LLM interface for parsing symptom text into structured features."""

import json
import logging
from typing import Dict, Any

from src.config import LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY
from src.llm_interface.prompt_templates import SYMPTOM_PARSING_PROMPT, EXPLANATION_PROMPT

logger = logging.getLogger(__name__)


def parse_symptom_text(raw_text: str) -> Dict[str, Any]:
    """
    Parse free-text symptom description into structured features.
    
    Args:
        raw_text: Patient's symptom description
        
    Returns:
        Dictionary with keys: symptom_categories, severity, duration_days, pattern, red_flags
    """
    if not raw_text or not raw_text.strip():
        return {
            "symptom_categories": ["general_discomfort"],
            "severity": 5.0,
            "duration_days": 3,
            "pattern": "constant",
            "red_flags": []
        }
    
    try:
        if LLM_PROVIDER == "openai":
            result = _parse_with_openai(raw_text)
        elif LLM_PROVIDER == "anthropic":
            result = _parse_with_anthropic(raw_text)
        else:
            result = _mock_parse(raw_text)
        
        if result.get("severity", 0) == 0:
            result["severity"] = 5.0
        
        return result
    except Exception as e:
        logger.error(f"Error parsing symptoms: {e}")
        result = _mock_parse(raw_text)
        if result.get("severity", 0) == 0:
            result["severity"] = 5.0
        return result


def _parse_with_openai(raw_text: str) -> Dict[str, Any]:
    """Parse using OpenAI API."""
    try:
        from openai import OpenAI
        
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found, using mock parser")
            return _mock_parse(raw_text)
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = SYMPTOM_PARSING_PROMPT.format(symptom_text=raw_text)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical assistant that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return parsed
    except ImportError:
        logger.warning("OpenAI library not installed, using mock parser")
        return _mock_parse(raw_text)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return _mock_parse(raw_text)


def _parse_with_anthropic(raw_text: str) -> Dict[str, Any]:
    """Parse using Anthropic API."""
    try:
        from anthropic import Anthropic
        
        if not ANTHROPIC_API_KEY:
            logger.warning("Anthropic API key not found, using mock parser")
            return _mock_parse(raw_text)
        
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = SYMPTOM_PARSING_PROMPT.format(symptom_text=raw_text)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text.strip()
        parsed = json.loads(content)
        return parsed
    except ImportError:
        logger.warning("Anthropic library not installed, using mock parser")
        return _mock_parse(raw_text)
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return _mock_parse(raw_text)


def _detect_injuries(text_lower: str) -> tuple:
    """Detect injuries and fractures in text."""
    injuries = []
    injury_severity = 0.0
    
    fracture_keywords = ["broken", "fracture", "cracked", "shattered", "snapped"]
    dislocation_keywords = ["dislocated", "out of place", "wrong way", "facing wrong", "misaligned", "popped out"]
    trauma_keywords = ["hit", "struck", "fell", "fall", "accident", "crash", "collision"]
    
    body_parts = ["arm", "leg", "foot", "ankle", "wrist", "hand", "finger", "toe", "shoulder", "elbow", "knee", "hip", "rib", "spine", "neck", "back"]
    
    for part in body_parts:
        if part in text_lower:
            if any(keyword in text_lower for keyword in fracture_keywords):
                injuries.append(f"fracture_{part}")
                injury_severity = max(injury_severity, 8.5)
            elif any(keyword in text_lower for keyword in dislocation_keywords):
                injuries.append(f"dislocation_{part}")
                injury_severity = max(injury_severity, 8.0)
            elif any(keyword in text_lower for keyword in trauma_keywords):
                injuries.append(f"trauma_{part}")
                injury_severity = max(injury_severity, 7.0)
    
    if "broken" in text_lower:
        injury_severity = max(injury_severity, 8.5)
    
    if any(keyword in text_lower for keyword in dislocation_keywords):
        injury_severity = max(injury_severity, 8.0)
    
    return injuries, injury_severity


def _calculate_severity_spectrum(text_lower: str, injuries: list, injury_severity: float) -> float:
    """Calculate severity on a spectrum from 0-10."""
    severity = 5.0
    
    if injury_severity > 0:
        severity = max(severity, injury_severity)
    
    if any(phrase in text_lower for phrase in ["dying", "death", "dead", "kill me", "im dying", "i'm dying"]):
        severity = 10.0
    elif any(word in text_lower for word in ["severe", "extreme", "intense", "unbearable", "critical", "emergency"]):
        severity = max(severity, 9.0)
    elif any(word in text_lower for word in ["very bad", "really bad", "terrible", "awful"]):
        severity = max(severity, 8.0)
    elif "significant" in text_lower:
        if "bleeding" in text_lower or "blood" in text_lower:
            if "won't stop" in text_lower or "wont stop" in text_lower or "not stopping" in text_lower:
                severity = 6.8
            else:
                severity = 6.5
        else:
            severity = 6.5
    elif any(word in text_lower for word in ["bad", "moderate"]):
        severity = max(severity, 6.5)
    elif any(word in text_lower for word in ["mild", "slight", "minor", "little"]):
        severity = min(severity, 4.0)
    
    if ("bleeding" in text_lower or "blood" in text_lower) and ("won't stop" in text_lower or "wont stop" in text_lower or "not stopping" in text_lower or "continuing" in text_lower or "persistent" in text_lower or "heavy" in text_lower):
        if "significant" not in text_lower and "severe" not in text_lower:
            severity = max(severity, 7.0)
    
    if len(injuries) >= 2:
        severity = max(severity, 8.5)
    
    if "broken" in text_lower and ("arm" in text_lower or "leg" in text_lower or "foot" in text_lower):
        severity = max(severity, 8.5)
    
    if any(phrase in text_lower for phrase in ["wrong way", "facing wrong", "out of place", "dislocated"]):
        severity = max(severity, 8.0)
    
    return min(severity, 10.0)


def _mock_parse(raw_text: str) -> Dict[str, Any]:
    """
    Mock parser for development/testing when API is unavailable.
    
    Version 4.3: Fixed severity detection for "significant bleeding that won't stop" = 6.8
    """
    if not raw_text:
        raw_text = ""
    
    text_lower = raw_text.lower().strip()
    
    injuries, injury_severity = _detect_injuries(text_lower)
    
    symptom_categories = []
    
    if any(word in text_lower for word in ["chest", "heart", "cardiac"]):
        symptom_categories.append("chest_pain")
    if any(word in text_lower for word in ["breath", "breathing", "shortness", "short of breath"]):
        symptom_categories.append("shortness_of_breath")
    if any(word in text_lower for word in ["fever", "temperature", "hot"]):
        symptom_categories.append("fever")
    if any(word in text_lower for word in ["head", "headache"]):
        symptom_categories.append("headache")
    if any(word in text_lower for word in ["stomach", "abdominal", "belly", "abdomen"]):
        symptom_categories.append("abdominal_pain")
    if any(word in text_lower for word in ["bleeding", "blood", "hemorrhage"]):
        symptom_categories.append("bleeding")
    if injuries:
        symptom_categories.extend(injuries)
        symptom_categories.append("trauma")
    
    if not symptom_categories:
        symptom_categories.append("general_discomfort")
    
    severity = _calculate_severity_spectrum(text_lower, injuries, injury_severity)
    
    if "significant" in text_lower and ("bleeding" in text_lower or "blood" in text_lower):
        if "won't stop" in text_lower or "wont stop" in text_lower or "not stopping" in text_lower:
            severity = 6.8
        elif severity < 6.5:
            severity = 6.5
    
    red_flags = []
    
    if any(phrase in text_lower for phrase in ["severe chest", "crushing", "pressure", "heart pain"]):
        red_flags.append("severe_chest_pain")
    if any(word in text_lower for word in ["unconscious", "passed out", "fainted"]):
        red_flags.append("loss_of_consciousness")
    if any(phrase in text_lower for phrase in ["can't breathe", "struggling to breathe", "shortness of breath"]):
        red_flags.append("difficulty_breathing")
    if any(word in text_lower for word in ["bleeding", "blood", "hemorrhage"]):
        red_flags.append("active_bleeding")
    if injuries:
        red_flags.append("traumatic_injury")
        if len(injuries) >= 2:
            red_flags.append("multiple_injuries")
    if "broken" in text_lower:
        red_flags.append("fracture")
    if any(phrase in text_lower for phrase in ["wrong way", "facing wrong", "out of place", "dislocated"]):
        red_flags.append("dislocation")
    
    if severity >= 9.0:
        red_flags.append("critical_severity")
    
    return {
        "symptom_categories": symptom_categories,
        "severity": severity,
        "duration_days": 1 if injuries else 3,
        "pattern": "acute" if injuries else ("progressive" if "worse" in text_lower else "constant"),
        "red_flags": red_flags
    }


def generate_explanation(risk_score: float, triage_label: str, parsed_symptoms: Dict, red_flags: list) -> str:
    """
    Generate explanation for triage decision using LLM.
    
    Args:
        risk_score: Computed risk score (0-1)
        triage_label: Triage category
        parsed_symptoms: Parsed symptom dictionary
        red_flags: List of red flags
        
    Returns:
        Explanation text
    """
    try:
        if LLM_PROVIDER == "openai":
            return _explain_with_openai(risk_score, triage_label, parsed_symptoms, red_flags)
        elif LLM_PROVIDER == "anthropic":
            return _explain_with_anthropic(risk_score, triage_label, parsed_symptoms, red_flags)
        else:
            return _mock_explanation(risk_score, triage_label, parsed_symptoms, red_flags)
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return _mock_explanation(risk_score, triage_label, parsed_symptoms, red_flags)


def _explain_with_openai(risk_score: float, triage_label: str, parsed_symptoms: Dict, red_flags: list) -> str:
    """Generate explanation using OpenAI."""
    try:
        from openai import OpenAI
        
        if not OPENAI_API_KEY:
            return _mock_explanation(risk_score, triage_label, parsed_symptoms, red_flags)
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = EXPLANATION_PROMPT.format(
            risk_score=risk_score,
            triage_label=triage_label,
            parsed_symptoms=json.dumps(parsed_symptoms),
            red_flags=json.dumps(red_flags)
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    except Exception:
        return _mock_explanation(risk_score, triage_label, parsed_symptoms, red_flags)


def _explain_with_anthropic(risk_score: float, triage_label: str, parsed_symptoms: Dict, red_flags: list) -> str:
    """Generate explanation using Anthropic."""
    try:
        from anthropic import Anthropic
        
        if not ANTHROPIC_API_KEY:
            return _mock_explanation(risk_score, triage_label, parsed_symptoms, red_flags)
        
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = EXPLANATION_PROMPT.format(
            risk_score=risk_score,
            triage_label=triage_label,
            parsed_symptoms=json.dumps(parsed_symptoms),
            red_flags=json.dumps(red_flags)
        )
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    except Exception:
        return _mock_explanation(risk_score, triage_label, parsed_symptoms, red_flags)


def _mock_explanation(risk_score: float, triage_label: str, parsed_symptoms: Dict, red_flags: list) -> str:
    """Mock explanation for development/testing."""
    base = f"This case is classified as {triage_label} based on a risk score of {risk_score:.2%}."
    
    if red_flags:
        base += f" Red flags detected: {', '.join(red_flags)}."
    
    severity = parsed_symptoms.get("severity", 0)
    if severity >= 8.0:
        base += f" High symptom severity ({severity:.1f}/10) indicates urgent medical evaluation may be needed."
    elif severity >= 6.0:
        base += f" Moderate to high severity ({severity:.1f}/10) suggests consultation with a healthcare provider."
    
    return base
