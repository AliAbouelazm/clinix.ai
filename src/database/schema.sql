CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    other_demographics TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_user ON patients(user_id);

CREATE TABLE IF NOT EXISTS symptom_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    parsed_symptoms_json TEXT,
    parsed_severity REAL,
    red_flags_json TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS clinical_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    symptom_report_id INTEGER NOT NULL,
    feature_vector_json TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id),
    FOREIGN KEY (symptom_report_id) REFERENCES symptom_reports(id)
);

CREATE TABLE IF NOT EXISTS triage_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    symptom_report_id INTEGER NOT NULL,
    risk_score REAL NOT NULL,
    triage_label TEXT NOT NULL,
    explanation TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id),
    FOREIGN KEY (symptom_report_id) REFERENCES symptom_reports(id)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_path TEXT NOT NULL,
    training_date TIMESTAMP,
    accuracy REAL,
    other_metrics TEXT
);

CREATE INDEX IF NOT EXISTS idx_symptom_reports_patient ON symptom_reports(patient_id);
CREATE INDEX IF NOT EXISTS idx_clinical_features_patient ON clinical_features(patient_id);
CREATE INDEX IF NOT EXISTS idx_triage_predictions_patient ON triage_predictions(patient_id);


