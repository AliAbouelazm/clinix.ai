"""Streamlit dashboard for triage system."""

import streamlit as st
import pandas as pd
import json
import sys
import os
import importlib
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

CACHE_BUST_VERSION = "4.3.3"
if 'cache_bust' not in st.session_state or st.session_state.cache_bust != CACHE_BUST_VERSION:
    st.session_state.cache_bust = CACHE_BUST_VERSION
    importlib.invalidate_caches()
    
    modules_to_reload = [
        'src.llm_interface.llm_parser',
        'src.inference.triage_engine',
        'src.database.db_utils'
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception:
                pass

from src.database.db_utils import get_db_session, insert_patient, insert_symptom_report, insert_clinical_features, insert_triage_prediction, get_patient_history
from src.database.db_utils import init_schema
from src.llm_interface.llm_parser import parse_symptom_text
from src.data_preprocessing.create_clinical_features import create_feature_vector
from src.inference.triage_engine import run_triage
from src.config import TRIAGE_LABELS
from src.visualization.plot_triage_distribution import plot_triage_distribution, plot_severity_vs_risk
import uuid

st.set_page_config(page_title="clinix.ai", layout="wide", initial_sidebar_state="collapsed")

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background-color: #ffffff;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    .logo-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .logo-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #666666;
        font-weight: 400;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #1a1a1a;
        color: #ffffff !important;
        border: none;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.65rem 1.8rem;
        border-radius: 6px;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff !important;
    }
    
    .stButton>button:focus,
    .stButton>button:active {
        color: #ffffff !important;
    }
    
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        color: #ffffff !important;
    }
    
    .stButton button p,
    .stButton button span,
    .stButton button div {
        color: #ffffff !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stTextArea label, .stRadio label {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stSelectbox>div>div, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
    }
    
    .stSelectbox [data-baseweb="select"] {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #1a1a1a !important;
    }
    
    .stSelectbox [data-baseweb="select"] [aria-selected="true"] {
        color: #1a1a1a !important;
    }
    
    [data-baseweb="popover"] [role="option"] {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #f8f8f8 !important;
        color: #1a1a1a !important;
    }
    
    [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background-color: #f8f8f8 !important;
        color: #1a1a1a !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .stSelectbox [data-baseweb="select"] {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .stSelectbox [data-baseweb="select"] > div {
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] [data-baseweb="popover"] {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] [data-baseweb="popover"] [role="option"] {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] [data-baseweb="popover"] [role="option"]:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    .stNumberInput>div>div>input {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    input[type="number"] {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stTextArea>div>div>textarea {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .stRadio>div {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }
    
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    .stExpander label {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-weight: 600;
    }
    
    p, div, span, li {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }
    
    .control-section {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 0.75rem;
    }
    
    .stMetric {
        background-color: #f8f8f8;
        border: 1px solid #e5e5e5;
        padding: 1rem;
        border-radius: 6px;
    }
    
    .stMetric label {
        font-family: 'Inter', sans-serif;
        color: #666666;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-weight: 600;
    }
    
    .disclaimer {
        background-color: #fff9e6;
        border-left: 3px solid #ffa500;
        padding: 1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-size: 0.9rem;
        border-radius: 0 6px 6px 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .sidebar h1, .sidebar h2, .sidebar h3 {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }
    
    .stAlert {
        font-family: 'Inter', sans-serif;
    }
    
    .stInfo {
        background-color: #e8f4f8;
        border-left: 3px solid #1a1a1a;
    }
    
    .stSuccess {
        background-color: #e8f4f8;
        border-left: 3px solid #1a1a1a;
    }
    
    .stWarning {
        background-color: #fff9e6;
        border-left: 3px solid #ffa500;
    }
    
    .stError {
        background-color: #ffe8e8;
        border-left: 3px solid #cc0000;
    }
    
    code {
        background-color: #f8f8f8 !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    pre {
        background-color: #f8f8f8 !important;
        border: 1px solid #e5e5e5 !important;
        color: #1a1a1a !important;
    }
    
    .stCode {
        background-color: #f8f8f8 !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    [data-testid="stCodeBlock"] {
        background-color: #f8f8f8 !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    [data-testid="stCodeBlock"] pre {
        background-color: #f8f8f8 !important;
        color: #1a1a1a !important;
    }
    
    .stJson {
        background-color: #f8f8f8 !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        color: #666666;
        padding: 0.75rem 1.5rem;
        border-radius: 6px 6px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1a1a1a;
        font-weight: 600;
        background-color: transparent;
        border-bottom: 2px solid #1a1a1a;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #1a1a1a;
        background-color: #f8f8f8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo-header">
    <div class="logo-text">clinix.ai</div>
    <div class="logo-subtitle">medical triage assessment</div>
    <div style="margin-top: 1rem; font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #666666; font-weight: 400; text-transform: uppercase; letter-spacing: 0.5px;">
        Version 4.3.3 - Schema Fix
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<strong>Important Disclaimer:</strong> This system is for educational and demonstration purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult qualified healthcare providers for medical concerns.
</div>
""", unsafe_allow_html=True)

try:
    init_schema()
except Exception as e:
    st.warning(f"Database initialization warning: {str(e)[:100]}")

from src.config import MODEL_PATH
import os

if not MODEL_PATH.exists():
    st.markdown("""
    <div class="control-section" style="background-color: #fff9e6; border-left: 3px solid #ffa500;">
        <div style="font-family: 'Inter', sans-serif; color: #1a1a1a; font-weight: 600; margin-bottom: 0.5rem;">
            Model Not Trained
        </div>
        <p style="font-family: 'Inter', sans-serif; color: #1a1a1a; margin-bottom: 1rem;">
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

tab1, tab2, tab3 = st.tabs(["Triage Assessment", "Patient History", "Analytics"])

with tab1:
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
                        patient_id = insert_patient(session, user_id=st.session_state.user_id, age=age, sex=sex)
                        
                        parsed_symptoms = parse_symptom_text(symptom_text)
                        parsed_symptoms["raw_text"] = symptom_text
                        
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
                        
                        try:
                            risk_score, triage_label, explanation = run_triage(
                                parsed_symptoms=parsed_symptoms,
                                age=age,
                                sex=sex,
                                raw_text=symptom_text
                            )
                        except TypeError:
                            risk_score, triage_label, explanation = run_triage(
                                parsed_symptoms=parsed_symptoms,
                                age=age,
                                sex=sex
                            )
                        
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
                            "parsed_symptoms": parsed_symptoms,
                            "raw_text": symptom_text
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
        
        if triage_data["triage_label"] == "urgent":
            st.markdown("""
            <div style="background-color: #ffe8e8; border-left: 4px solid #cc0000; padding: 1.5rem; margin-bottom: 2rem; border-radius: 6px;">
                <h3 style="color: #cc0000; font-family: 'Inter', sans-serif; margin: 0 0 0.5rem 0; font-weight: 700; font-size: 1.2rem;">URGENT: SEEK IMMEDIATE MEDICAL CARE</h3>
                <p style="color: #1a1a1a; font-family: 'Inter', sans-serif; margin: 0; font-size: 1rem; font-weight: 500;">
                    Critical symptoms detected. This requires immediate medical attention. Please go to the nearest emergency room or call emergency services.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{triage_data['risk_score']:.2%}")
        
        with col2:
            st.metric("Triage Category", TRIAGE_LABELS.get(triage_data['triage_label'], triage_data['triage_label']))
        
        with col3:
            st.metric("Severity", f"{triage_data['parsed_symptoms'].get('severity', 0):.1f}/10")
        
        st.markdown("""
        <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #1a1a1a; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Explanation
        </div>
        """, unsafe_allow_html=True)
        st.info(triage_data["explanation"])
        
        st.markdown("""
        <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #1a1a1a; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
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
                    st.markdown(f"- {flag}")
            else:
                st.markdown("None detected")
            
            st.markdown("**Duration:**")
            st.markdown(f"{parsed.get('duration_days', 0)} days")

with tab2:
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
                history = get_patient_history(session, patient_id, user_id=st.session_state.user_id)
                
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

with tab3:
    st.markdown("""
    <div class="section-title">Triage Analytics</div>
    """, unsafe_allow_html=True)
    
    try:
        st.markdown("""
        <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #1a1a1a; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Triage Distribution
        </div>
        """, unsafe_allow_html=True)
        fig1 = plot_triage_distribution()
        st.pyplot(fig1)
        
        st.markdown("""
        <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #1a1a1a; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Severity vs Risk Score
        </div>
        """, unsafe_allow_html=True)
        fig2 = plot_severity_vs_risk()
        st.pyplot(fig2)
    
    except Exception as e:
        st.error(f"Error generating analytics: {e}")

