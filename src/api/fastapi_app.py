"""FastAPI backend for triage system."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from src.database.db_utils import (
    get_db_session, insert_patient, insert_symptom_report,
    insert_clinical_features, insert_triage_prediction, get_patient_history
)
from src.llm_interface.llm_parser import parse_symptom_text
from src.data_preprocessing.create_clinical_features import create_feature_vector
from src.inference.triage_engine import run_triage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="clinix.ai API", version="1.0.0")


class TriageRequest(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    symptom_text: str
    patient_id: Optional[int] = None


class TriageResponse(BaseModel):
    patient_id: int
    symptom_report_id: int
    risk_score: float
    triage_label: str
    triage_category: str
    explanation: str
    parsed_symptoms: dict


@app.post("/triage", response_model=TriageResponse)
async def triage_endpoint(request: TriageRequest):
    """
    Process triage request.
    
    Accepts patient demographics and symptom text, returns triage decision.
    """
    try:
        with get_db_session() as session:
            if request.patient_id:
                patient_id = request.patient_id
            else:
                patient_id = insert_patient(
                    session, age=request.age, sex=request.sex
                )
            
            parsed_symptoms = parse_symptom_text(request.symptom_text)
            
            symptom_report_id = insert_symptom_report(
                session,
                patient_id=patient_id,
                raw_text=request.symptom_text,
                parsed_symptoms_json=parsed_symptoms,
                parsed_severity=parsed_symptoms.get("severity"),
                red_flags_json=parsed_symptoms.get("red_flags", [])
            )
            
            feature_vector = create_feature_vector(
                parsed_symptoms, age=request.age, sex=request.sex
            )
            
            insert_clinical_features(
                session,
                patient_id=patient_id,
                symptom_report_id=symptom_report_id,
                feature_vector=feature_vector
            )
            
            risk_score, triage_label, explanation = run_triage(
                parsed_symptoms, age=request.age, sex=request.sex
            )
            
            insert_triage_prediction(
                session,
                patient_id=patient_id,
                symptom_report_id=symptom_report_id,
                risk_score=risk_score,
                triage_label=triage_label,
                explanation=explanation
            )
            
            from src.config import TRIAGE_LABELS
            triage_category = TRIAGE_LABELS.get(triage_label, triage_label)
            
            return TriageResponse(
                patient_id=patient_id,
                symptom_report_id=symptom_report_id,
                risk_score=risk_score,
                triage_label=triage_label,
                triage_category=triage_category,
                explanation=explanation,
                parsed_symptoms=parsed_symptoms
            )
    
    except Exception as e:
        logger.error(f"Error in triage endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patient/{patient_id}/history")
async def get_history(patient_id: int):
    """
    Get patient history of symptom reports and triage decisions.
    """
    try:
        with get_db_session() as session:
            history = get_patient_history(session, patient_id)
            
            if not history:
                raise HTTPException(status_code=404, detail="Patient not found")
            
            import json
            results = []
            for row in history:
                results.append({
                    "report_id": row.report_id,
                    "raw_text": row.raw_text,
                    "parsed_symptoms": json.loads(row.parsed_symptoms_json) if row.parsed_symptoms_json else None,
                    "severity": row.parsed_severity,
                    "red_flags": json.loads(row.red_flags_json) if row.red_flags_json else None,
                    "risk_score": row.risk_score,
                    "triage_label": row.triage_label,
                    "explanation": row.explanation,
                    "report_timestamp": row.report_timestamp,
                    "prediction_timestamp": row.prediction_timestamp
                })
            
            return {"patient_id": patient_id, "history": results}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

