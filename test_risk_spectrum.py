"""Test script to check risk score spectrum across different symptoms."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.llm_interface.llm_parser import parse_symptom_text
from src.inference.triage_engine import run_triage

test_cases = [
    ("mild headache", "Should be low risk"),
    ("stomach ache after eating", "Should be low-moderate risk"),
    ("slight fever, feeling tired", "Should be moderate risk"),
    ("moderate chest discomfort", "Should be moderate-high risk"),
    ("my stomach hurts and my arm is broken and my foot is facing the wrong way", "Should be high risk"),
    ("broken arm", "Should be high risk"),
    ("heart is hurting", "Should be very high risk"),
    ("im dying", "Should be critical"),
    ("mild cough, runny nose", "Should be low risk"),
    ("severe abdominal pain", "Should be high risk"),
    ("minor cut on finger", "Should be low risk"),
    ("chest pain and shortness of breath", "Should be critical"),
    ("dislocated shoulder", "Should be high risk"),
    ("fever and body aches", "Should be moderate risk"),
    ("bleeding from wound", "Should be moderate-high risk"),
]

print("=" * 80)
print("RISK SPECTRUM TEST RESULTS")
print("=" * 80)
print()

results = []

for symptom_text, expected in test_cases:
    try:
        parsed = parse_symptom_text(symptom_text)
        parsed["raw_text"] = symptom_text
        
        risk_score, triage_label, explanation = run_triage(
            parsed_symptoms=parsed,
            age=30,
            sex="M",
            raw_text=symptom_text
        )
        
        results.append({
            "symptom": symptom_text[:50],
            "risk": risk_score,
            "triage": triage_label,
            "severity": parsed.get("severity", 0),
            "red_flags": len(parsed.get("red_flags", []))
        })
        
        print(f"Symptom: {symptom_text}")
        print(f"  Risk Score: {risk_score:.2%}")
        print(f"  Triage: {triage_label}")
        print(f"  Severity: {parsed.get('severity', 0):.1f}/10")
        print(f"  Red Flags: {len(parsed.get('red_flags', []))}")
        print(f"  Expected: {expected}")
        print()
    except Exception as e:
        print(f"ERROR with '{symptom_text}': {e}")
        print()

print("=" * 80)
print("RISK SCORE DISTRIBUTION")
print("=" * 80)

risk_scores = [r["risk"] for r in results]
unique_scores = sorted(set(risk_scores))

print(f"Unique risk scores found: {len(unique_scores)}")
print(f"Risk score range: {min(risk_scores):.2%} to {max(risk_scores):.2%}")
print()
print("Risk score distribution:")
for score in unique_scores:
    count = risk_scores.count(score)
    print(f"  {score:.2%}: {count} case(s)")

print()
print("=" * 80)
print("ANALYSIS")
print("=" * 80)

if len(unique_scores) <= 3:
    print("⚠️  WARNING: System appears to be binary/ternary - only a few distinct risk scores!")
    print("   Need to adjust the model to provide more spectrum.")
else:
    print("✓ System provides spectrum of risk scores")

print()
print("Cases by risk level:")
low_risk = [r for r in results if r["risk"] < 0.4]
moderate_risk = [r for r in results if 0.4 <= r["risk"] < 0.7]
high_risk = [r for r in results if 0.7 <= r["risk"] < 0.9]
critical_risk = [r for r in results if r["risk"] >= 0.9]

print(f"  Low risk (<40%): {len(low_risk)} cases")
print(f"  Moderate risk (40-70%): {len(moderate_risk)} cases")
print(f"  High risk (70-90%): {len(high_risk)} cases")
print(f"  Critical risk (≥90%): {len(critical_risk)} cases")

