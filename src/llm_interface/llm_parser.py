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
    try:
        if LLM_PROVIDER == "openai":
            return _parse_with_openai(raw_text)
        elif LLM_PROVIDER == "anthropic":
            return _parse_with_anthropic(raw_text)
        else:
            logger.warning(f"Unknown LLM provider: {LLM_PROVIDER}, using mock parser")
            return _mock_parse(raw_text)
    except Exception as e:
        logger.error(f"Error parsing symptoms: {e}")
        return _mock_parse(raw_text)


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


def _mock_parse(raw_text: str) -> Dict[str, Any]:
    """Mock parser for development/testing when API is unavailable."""
    text_lower = raw_text.lower()
    
    symptom_categories = []
    if any(word in text_lower for word in ["chest", "heart", "cardiac", "aching pain in my heart", "heart is", "heart hurting", "heart pain"]):
        symptom_categories.append("chest_pain")
    if any(word in text_lower for word in ["breath", "breathing", "shortness", "short of breath"]):
        symptom_categories.append("shortness_of_breath")
    if any(word in text_lower for word in ["fever", "temperature", "hot"]):
        symptom_categories.append("fever")
    if any(word in text_lower for word in ["head", "headache"]):
        symptom_categories.append("headache")
    if any(word in text_lower for word in ["bleeding", "blood", "hemorrhage"]):
        symptom_categories.append("bleeding")
    if not symptom_categories:
        symptom_categories.append("general_discomfort")
    
    severity = 5.0
    if any(phrase in text_lower for phrase in ["dying", "death", "dead", "kill me", "im dying", "i'm dying", "i am dying"]):
        severity = 10.0
    elif any(word in text_lower for word in ["severe", "extreme", "intense", "unbearable", "critical"]):
        severity = 9.0
    elif any(word in text_lower for word in ["moderate", "bad"]):
        severity = 6.0
    elif any(word in text_lower for word in ["mild", "slight", "minor"]):
        severity = 3.0
    
    red_flags = []
    if any(phrase in text_lower for phrase in ["severe chest", "crushing", "pressure", "heart pain", "aching pain in my heart", "heart is hurting", "heart hurting", "heart hurts"]):
        red_flags.append("severe_chest_pain")
    if any(word in text_lower for word in ["unconscious", "passed out", "fainted"]):
        red_flags.append("loss_of_consciousness")
    if any(phrase in text_lower for phrase in ["can't breathe", "struggling to breathe", "shortness of breath", "short of breath", "cant breathe"]):
        red_flags.append("difficulty_breathing")
    if any(word in text_lower for word in ["bleeding", "blood", "hemorrhage"]):
        red_flags.append("active_bleeding")
    if ("chest" in text_lower or "heart" in text_lower) and "breath" in text_lower:
        red_flags.append("cardiac_concern")
    if "heart" in text_lower and ("hurt" in text_lower or "pain" in text_lower or "hurting" in text_lower):
        red_flags.append("severe_chest_pain")
    
    if severity >= 9.0:
        red_flags.append("critical_severity")
    
    if "dying" in text_lower:
        red_flags.append("critical_severity")
        severity = max(severity, 10.0)
    
    return {
        "symptom_categories": symptom_categories,
        "severity": severity,
        "duration_days": 3,
        "pattern": "progressive" if "worse" in text_lower or "increasing" in text_lower else "constant",
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
    base = f"This case is classified as {triage_label} based on a risk score of {risk_score:.2f}."
    
    if red_flags:
        base += f" Red flags detected: {', '.join(red_flags)}."
    
    if parsed_symptoms.get("severity", 0) > 7:
        base += " High symptom severity contributes to the risk assessment."
    
    return base


