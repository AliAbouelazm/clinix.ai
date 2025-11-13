"""Database utilities for connection and operations."""

import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from pathlib import Path

from src.config import DB_URL, DB_PATH, PROJECT_ROOT

engine = None
SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global engine
    if engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    return engine


def get_session() -> Session:
    """Get database session."""
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(bind=get_engine())
    return SessionLocal()


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema():
    """Initialize database schema from schema.sql."""
    schema_path = PROJECT_ROOT / "src" / "database" / "schema.sql"
    engine = get_engine()
    
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    
    with engine.connect() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()


def insert_patient(session: Session, age: int = None, sex: str = None, other_demographics: str = None) -> int:
    """Insert a new patient and return patient_id."""
    result = session.execute(
        text("""
            INSERT INTO patients (age, sex, other_demographics)
            VALUES (:age, :sex, :other_demographics)
        """),
        {"age": age, "sex": sex, "other_demographics": other_demographics}
    )
    session.commit()
    return result.lastrowid


def insert_symptom_report(
    session: Session,
    patient_id: int,
    raw_text: str,
    parsed_symptoms_json: dict = None,
    parsed_severity: float = None,
    red_flags_json: dict = None
) -> int:
    """Insert symptom report and return report_id."""
    result = session.execute(
        text("""
            INSERT INTO symptom_reports 
            (patient_id, raw_text, parsed_symptoms_json, parsed_severity, red_flags_json)
            VALUES (:patient_id, :raw_text, :parsed_symptoms_json, :parsed_severity, :red_flags_json)
        """),
        {
            "patient_id": patient_id,
            "raw_text": raw_text,
            "parsed_symptoms_json": json.dumps(parsed_symptoms_json) if parsed_symptoms_json else None,
            "parsed_severity": parsed_severity,
            "red_flags_json": json.dumps(red_flags_json) if red_flags_json else None
        }
    )
    session.commit()
    return result.lastrowid


def insert_clinical_features(
    session: Session,
    patient_id: int,
    symptom_report_id: int,
    feature_vector: dict
) -> int:
    """Insert clinical features and return feature_id."""
    result = session.execute(
        text("""
            INSERT INTO clinical_features (patient_id, symptom_report_id, feature_vector_json)
            VALUES (:patient_id, :symptom_report_id, :feature_vector_json)
        """),
        {
            "patient_id": patient_id,
            "symptom_report_id": symptom_report_id,
            "feature_vector_json": json.dumps(feature_vector)
        }
    )
    session.commit()
    return result.lastrowid


def insert_triage_prediction(
    session: Session,
    patient_id: int,
    symptom_report_id: int,
    risk_score: float,
    triage_label: str,
    explanation: str = None
) -> int:
    """Insert triage prediction and return prediction_id."""
    result = session.execute(
        text("""
            INSERT INTO triage_predictions 
            (patient_id, symptom_report_id, risk_score, triage_label, explanation)
            VALUES (:patient_id, :symptom_report_id, :risk_score, :triage_label, :explanation)
        """),
        {
            "patient_id": patient_id,
            "symptom_report_id": symptom_report_id,
            "risk_score": risk_score,
            "triage_label": triage_label,
            "explanation": explanation
        }
    )
    session.commit()
    return result.lastrowid


def get_patient_history(session: Session, patient_id: int):
    """Get all symptom reports and triage predictions for a patient."""
    result = session.execute(
        text("""
            SELECT 
                sr.id as report_id,
                sr.raw_text,
                sr.parsed_symptoms_json,
                sr.parsed_severity,
                sr.red_flags_json,
                sr.timestamp as report_timestamp,
                tp.risk_score,
                tp.triage_label,
                tp.explanation,
                tp.timestamp as prediction_timestamp
            FROM symptom_reports sr
            LEFT JOIN triage_predictions tp ON sr.id = tp.symptom_report_id
            WHERE sr.patient_id = :patient_id
            ORDER BY sr.timestamp DESC
        """),
        {"patient_id": patient_id}
    )
    return result.fetchall()

