"""Prompt templates for LLM symptom parsing."""

SYMPTOM_PARSING_PROMPT = """You are a medical assistant that parses patient symptom descriptions into structured data.

Given the following symptom description, extract and return a JSON object with the following structure:
{{
    "symptom_categories": ["category1", "category2", ...],
    "severity": <number between 0-10>,
    "duration_days": <number>,
    "pattern": "intermittent" | "progressive" | "constant" | "acute",
    "red_flags": ["flag1", "flag2", ...]
}}

Symptom categories should be standardized medical terms (e.g., "chest_pain", "shortness_of_breath", "fever", "headache").
Red flags are concerning symptoms that may indicate serious conditions (e.g., "severe_chest_pain", "loss_of_consciousness", "difficulty_breathing").

Symptom description:
{symptom_text}

Return ONLY valid JSON, no additional text."""

EXPLANATION_PROMPT = """You are a medical assistant providing a brief, clear explanation of a triage decision.

Given:
- Risk score: {risk_score} (0-1 scale, where 1 is highest risk)
- Triage category: {triage_label}
- Parsed symptoms: {parsed_symptoms}
- Red flags detected: {red_flags}

Provide a concise explanation (2-3 sentences) explaining why this case was classified as {triage_label}.
Focus on the key factors: symptom severity, red flags, and risk level.

Explanation:"""

