"""Streamlit dashboard for triage system."""

import streamlit as st
import pandas as pd
import json
import sys
import os
from pathlib import Path
from datetime import datetime

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    sys.path.append(str(project_root))

try:
    os.chdir(project_root)
except:
    pass

from src.database.db_utils import get_db_session, insert_patient, insert_symptom_report, insert_clinical_features, insert_triage_prediction, get_patient_history
from src.database.db_utils import init_schema
from src.llm_interface.llm_parser import parse_symptom_text
from src.data_preprocessing.create_clinical_features import create_feature_vector
from src.inference.triage_engine import run_triage
from src.config import TRIAGE_LABELS
from src.visualization.plot_triage_distribution import plot_triage_distribution, plot_severity_vs_risk

st.set_page_config(page_title="clinix.ai", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Playfair+Display:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Crimson Text', 'Times New Roman', serif;
    }
    
    .main {
        background-color: #f5f3f0;
    }
    
    .stApp {
        background-color: #f5f3f0;
    }
    
    .logo-header {
        text-align: center;
        padding: 2rem 0 1rem;
        border-bottom: 2px solid #8b6f47;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #3d2f1f;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .logo-subtitle {
        font-family: 'Crimson Text', serif;
        font-size: 0.9rem;
        color: #6b5d4f;
        font-style: italic;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #3d2f1f;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #6b5237;
        color: #f5f3f0 !important;
        border: none;
        font-family: 'Crimson Text', serif;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #5a4229;
        color: #f5f3f0 !important;
    }
    
    .stButton>button:focus {
        color: #f5f3f0 !important;
    }
    
    .stButton>button:active {
        color: #f5f3f0 !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stTextArea label, .stRadio label {
        font-family: 'Crimson Text', serif;
        color: #3d2f1f;
        font-weight: 600;
    }
    
    .stSelectbox>div>div, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
        font-family: 'Crimson Text', serif;
        color: #3d2f1f !important;
        background-color: #fefefe !important;
        border: 1px solid #c9b8a3;
    }
    
    .stNumberInput>div>div>input {
        color: #3d2f1f !important;
        background-color: #fefefe !important;
    }
    
    input[type="number"] {
        color: #3d2f1f !important;
        background-color: #fefefe !important;
    }
    
    .stTextArea>div>div>textarea {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .stRadio>div {
        font-family: 'Crimson Text', serif;
        color: #3d2f1f;
    }
    
    .stExpander {
        background-color: #fefefe;
        border: 1px solid #c9b8a3;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .stExpander label {
        font-family: 'Playfair Display', serif;
        color: #3d2f1f;
        font-weight: 600;
    }
    
    p, div, span, li {
        font-family: 'Crimson Text', serif;
        color: #3d2f1f;
    }
    
    .control-section {
        background-color: #fefefe;
        border: 1px solid #c9b8a3;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        color: #3d2f1f;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #8b6f47;
        padding-bottom: 0.5rem;
    }
    
    .stMetric {
        background-color: #fefefe;
        border: 1px solid #c9b8a3;
        padding: 1rem;
        border-radius: 4px;
    }
    
    .stMetric label {
        font-family: 'Crimson Text', serif;
        color: #6b5d4f;
        font-size: 0.9rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif;
        color: #3d2f1f;
    }
    
    .disclaimer {
        background-color: #f5ede0;
        border-left: 4px solid #8b6f47;
        padding: 1rem;
        margin-bottom: 2rem;
        font-family: 'Crimson Text', serif;
        color: #5a4a3a;
        font-size: 0.95rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f5f3f0;
    }
    
    .sidebar h1, .sidebar h2, .sidebar h3 {
        font-family: 'Playfair Display', serif;
        color: #3d2f1f;
    }
    
    .stAlert {
        font-family: 'Crimson Text', serif;
    }
    
    .stInfo {
        background-color: #ede8e0;
        border-left: 4px solid #8b6f47;
    }
    
    .stSuccess {
        background-color: #ede8e0;
        border-left: 4px solid #8b6f47;
    }
    
    .stWarning {
        background-color: #f5ede0;
        border-left: 4px solid #8b6f47;
    }
    
    .stError {
        background-color: #f0e5e5;
        border-left: 4px solid #a67c7c;
    }
    
    code {
        background-color: #f5f3f0 !important;
        color: #3d2f1f !important;
        border: 1px solid #c9b8a3 !important;
    }
    
    pre {
        background-color: #f5f3f0 !important;
        border: 1px solid #c9b8a3 !important;
        color: #3d2f1f !important;
    }
    
    .stCode {
        background-color: #f5f3f0 !important;
        border: 1px solid #c9b8a3 !important;
    }
    
    [data-testid="stCodeBlock"] {
        background-color: #f5f3f0 !important;
        border: 1px solid #c9b8a3 !important;
    }
    
    [data-testid="stCodeBlock"] pre {
        background-color: #f5f3f0 !important;
        color: #3d2f1f !important;
    }
    
    .stJson {
        background-color: #f5f3f0 !important;
        border: 1px solid #c9b8a3 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo-header">
    <div class="logo-text">clinix.ai</div>
    <div class="logo-subtitle">medical triage assessment</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<strong>⚠️ Important Disclaimer:</strong> This system is for educational and demonstration purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult qualified healthcare providers for medical concerns.
</div>
""", unsafe_allow_html=True)

init_schema()

from src.config import MODEL_PATH
import os

if not MODEL_PATH.exists():
    st.markdown("""
    <div class="control-section" style="background-color: #f5ede0; border-left: 4px solid #8b6f47;">
        <div style="font-family: 'Playfair Display', serif; color: #3d2f1f; font-weight: 600; margin-bottom: 0.5rem;">
            Model Not Trained
        </div>
        <p style="font-family: 'Crimson Text', serif; color: #5a4a3a; margin-bottom: 1rem;">
            The risk classification model needs to be trained before you can use the triage assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Train Model Now", type="primary"):
        with st.spinner("Training model... This may take a minute."):
            try:
                import subprocess
                import sys
                from pathlib import Path
                
                project_root = Path(__file__).parent.parent.parent
                env = os.environ.copy()
                env['PYTHONPATH'] = str(project_root)
                
                result = subprocess.run(
                    [sys.executable, "src/models/train_baseline_model.py"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success("Model trained successfully! You can now use the triage assessment.")
                    st.rerun()
                else:
                    st.error(f"Error training model: {result.stderr}")
                    st.code(result.stdout)
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown("---")

st.markdown("""
<div class="control-section">
    <div class="section-title">Navigation</div>
</div>
""", unsafe_allow_html=True)

page = st.radio("", ["Triage Assessment", "Patient History", "Analytics"], horizontal=True, label_visibility="collapsed")

st.markdown("---")

if page == "Triage Assessment":
    st.markdown("""
    <div class="control-section">
        <div class="section-title">Patient Information</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    with col2:
        sex = st.selectbox("Sex", options=["M", "F"], index=0)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="section-title">Symptom Assessment</div>
    """, unsafe_allow_html=True)
    
    symptom_text = st.text_area(
        "Describe your symptoms:",
        height=150,
        placeholder="Example: I've been experiencing chest pain for the past 2 days. It's getting worse and I feel short of breath."
    )
    
    if st.button("Run Triage", type="primary"):
        if not symptom_text.strip():
            st.error("Please enter symptom description")
        else:
            with st.spinner("Processing symptoms and computing risk..."):
                try:
                    with get_db_session() as session:
                        patient_id = insert_patient(session, age=age, sex=sex)
                        
                        parsed_symptoms = parse_symptom_text(symptom_text)
                        
                        symptom_report_id = insert_symptom_report(
                            session,
                            patient_id=patient_id,
                            raw_text=symptom_text,
                            parsed_symptoms_json=parsed_symptoms,
                            parsed_severity=parsed_symptoms.get("severity"),
                            red_flags_json=parsed_symptoms.get("red_flags", [])
                        )
                        
                        feature_vector = create_feature_vector(parsed_symptoms, age, sex)
                        
                        insert_clinical_features(
                            session,
                            patient_id=patient_id,
                            symptom_report_id=symptom_report_id,
                            feature_vector=feature_vector
                        )
                        
                        risk_score, triage_label, explanation = run_triage(parsed_symptoms, age, sex)
                        
                        insert_triage_prediction(
                            session,
                            patient_id=patient_id,
                            symptom_report_id=symptom_report_id,
                            risk_score=risk_score,
                            triage_label=triage_label,
                            explanation=explanation
                        )
                        
                        st.session_state["last_patient_id"] = patient_id
                        st.session_state["last_triage"] = {
                            "risk_score": risk_score,
                            "triage_label": triage_label,
                            "explanation": explanation,
                            "parsed_symptoms": parsed_symptoms
                        }
                
                except Exception as e:
                    st.error(f"Error processing triage: {e}")
                    if "Model not found" in str(e):
                        st.info("Please train the model first by running: `python3 src/models/train_baseline_model.py`")
                    else:
                        import traceback
                        st.code(traceback.format_exc())
    
    if "last_triage" in st.session_state:
        triage_data = st.session_state["last_triage"]
        
        st.markdown("---")
        st.markdown("""
        <div class="section-title">Triage Results</div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{triage_data['risk_score']:.2%}")
        
        with col2:
            label_colors = {
                "urgent": "🔴",
                "consult": "🟡",
                "self_care": "🟢"
            }
            label_color = label_colors.get(triage_data["triage_label"], "⚪")
            st.metric("Triage Category", f"{label_color} {TRIAGE_LABELS.get(triage_data['triage_label'], triage_data['triage_label'])}")
        
        with col3:
            st.metric("Severity", f"{triage_data['parsed_symptoms'].get('severity', 0):.1f}/10")
        
        st.markdown("""
        <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #3d2f1f; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Explanation
        </div>
        """, unsafe_allow_html=True)
        st.info(triage_data["explanation"])
        
        st.markdown("""
        <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #3d2f1f; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Detected Symptoms
        </div>
        """, unsafe_allow_html=True)
        parsed = triage_data["parsed_symptoms"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Symptom Categories:**")
            for symptom in parsed.get("symptom_categories", []):
                st.markdown(f"- {symptom}")
            
            st.markdown("**Pattern:**")
            st.markdown(parsed.get("pattern", "N/A"))
        
        with col2:
            st.markdown("**Red Flags:**")
            red_flags = parsed.get("red_flags", [])
            if red_flags:
                for flag in red_flags:
                    st.markdown(f"- ⚠️ {flag}")
            else:
                st.markdown("None detected")
            
            st.markdown("**Duration:**")
            st.markdown(f"{parsed.get('duration_days', 0)} days")

elif page == "Patient History":
    st.markdown("""
    <div class="section-title">Patient History</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="control-section">
    """, unsafe_allow_html=True)
    patient_id = st.number_input("Patient ID", min_value=1, value=st.session_state.get("last_patient_id", 1), step=1)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Load History"):
        try:
            with get_db_session() as session:
                history = get_patient_history(session, patient_id)
                
                if not history:
                    st.warning("No history found for this patient")
                else:
                    st.markdown(f"""
                    <div class="section-title">History for Patient {patient_id}</div>
                    """, unsafe_allow_html=True)
                    
                    for row in history:
                        with st.expander(f"Report {row.report_id} - {row.report_timestamp}"):
                            st.markdown("**Symptom Description:**")
                            st.markdown(row.raw_text)
                            
                            if row.parsed_symptoms_json:
                                parsed = json.loads(row.parsed_symptoms_json)
                                st.markdown("**Parsed Symptoms:**")
                                st.json(parsed)
                            
                            if row.risk_score is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Risk Score", f"{row.risk_score:.2%}")
                                with col2:
                                    st.metric("Triage", TRIAGE_LABELS.get(row.triage_label, row.triage_label))
                                
                                if row.explanation:
                                    st.markdown("**Explanation:**")
                                    st.info(row.explanation)
        
        except Exception as e:
            st.error(f"Error loading history: {e}")

elif page == "Analytics":
    st.markdown("""
    <div class="section-title">Triage Analytics</div>
    """, unsafe_allow_html=True)
    
    try:
        st.markdown("""
        <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #3d2f1f; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Triage Distribution
        </div>
        """, unsafe_allow_html=True)
        fig1 = plot_triage_distribution()
        st.pyplot(fig1)
        
        st.markdown("""
        <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #3d2f1f; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Severity vs Risk Score
        </div>
        """, unsafe_allow_html=True)
        fig2 = plot_severity_vs_risk()
        st.pyplot(fig2)
    
    except Exception as e:
        st.error(f"Error generating analytics: {e}")

