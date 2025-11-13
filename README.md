# clinix.ai

A medical triage and symptom-to-risk assessment system that uses LLM semantic interpretation, classical ML models, SQL storage, and clinical feature engineering to provide triage recommendations.

## Live Demo

Try the interactive Streamlit app: **[https://clinixai.streamlit.app/](https://clinixai.streamlit.app/)**

## ⚠️ IMPORTANT DISCLAIMER

**This system is for EDUCATIONAL and DEMONSTRATION purposes ONLY.**

- **NOT a substitute for professional medical advice, diagnosis, or treatment**
- **NOT intended for actual clinical use**
- **Always consult qualified healthcare providers for medical concerns**
- **Do not use this system for real medical decisions**

## Overview

The AI Clinic Layer system:

1. Accepts free-text symptom descriptions from users
2. Uses an LLM (OpenAI or Anthropic) to parse symptoms into structured medical features
3. Applies a classical ML model (Logistic Regression or Random Forest) to compute risk scores
4. Classifies cases into triage categories: "Seek care now", "Consult GP", or "Monitor at home"
5. Stores all data in SQLite database
6. Provides a Streamlit dashboard and FastAPI backend

## Architecture

```
User Input (Symptoms)
    ↓
LLM Parser (OpenAI/Anthropic)
    ↓
Structured Features
    ↓
Feature Engineering
    ↓
ML Model (Logistic Regression/Random Forest)
    ↓
Risk Score + Triage Decision
    ↓
SQL Database Storage
    ↓
Dashboard Visualization
```

## Tech Stack

- **Python** - Core language
- **pandas, numpy** - Data processing
- **scikit-learn** - Classical ML models
- **SQLAlchemy** - Database ORM
- **FastAPI** - Backend API
- **Streamlit** - Dashboard UI
- **OpenAI/Anthropic** - LLM for symptom parsing
- **SQLite** - Database storage
- **matplotlib/plotly** - Visualizations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_clinic_layer
```

**Note:** The project is named `ai_clinic_layer` in the repository, but the application is branded as **clinix.ai**.

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

## Setup

1. Initialize the database:
```python
from src.database.db_utils import init_schema
init_schema()
```

2. Train the model:
```bash
python src/models/train_baseline_model.py
```

This will:
- Load/create sample medical dataset
- Clean and preprocess data
- Train a Logistic Regression model
- Save model to `models/risk_classifier.pkl`

## Usage

### Streamlit Dashboard

Launch the interactive dashboard:
```bash
streamlit run src/app/streamlit_app.py
```

The dashboard provides:
- **Triage Assessment**: Enter symptoms and get risk assessment
- **Patient History**: View past symptom reports and triage decisions
- **Analytics**: Visualize triage distribution and severity vs risk

### FastAPI Backend

Start the API server:
```bash
python src/api/fastapi_app.py
```

Or using uvicorn:
```bash
uvicorn src.api.fastapi_app:app --reload
```

API endpoints:
- `POST /triage` - Submit symptom text and get triage decision
- `GET /patient/{id}/history` - Get patient history
- `GET /health` - Health check

Example API request:
```bash
curl -X POST "http://localhost:8000/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "sex": "M",
    "symptom_text": "I have chest pain and shortness of breath"
  }'
```

## Project Structure

```
ai_clinic_layer/
├── data/
│   ├── raw/              # Raw medical datasets
│   ├── interim/          # Intermediate processed data
│   ├── processed/        # Final processed data
│   └── clinic.db         # SQLite database
├── models/               # Trained ML models
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── config.py         # Configuration constants
│   ├── database/         # Database schema and utilities
│   ├── llm_interface/    # LLM parsing layer
│   ├── data_preprocessing/  # Data cleaning and feature engineering
│   ├── models/           # ML model training and risk scoring
│   ├── inference/        # Triage engine
│   ├── api/              # FastAPI backend
│   ├── visualization/    # Plotting utilities
│   └── app/              # Streamlit dashboard
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md
```

## Pipeline Steps

1. **Symptom Input**: User provides free-text symptom description
2. **LLM Parsing**: LLM converts text to structured features:
   - Symptom categories
   - Severity (0-10)
   - Duration
   - Pattern (intermittent/progressive/constant/acute)
   - Red flags
3. **Feature Engineering**: Create numeric feature vector:
   - Symptom count
   - Normalized severity
   - Red flag indicators
   - Demographic features (age, sex)
   - Encoded symptom categories
4. **Risk Scoring**: ML model computes risk score (0-1)
5. **Triage Classification**:
   - Risk ≥ 0.8 → "urgent" (Seek care now)
   - 0.4 ≤ Risk < 0.8 → "consult" (Consult GP)
   - Risk < 0.4 → "self_care" (Monitor at home)
6. **Explanation Generation**: LLM generates explanation for decision
7. **Storage**: All data stored in SQL database

## Database Schema

- **patients**: Patient demographics
- **symptom_reports**: Raw and parsed symptom data
- **clinical_features**: Engineered feature vectors
- **triage_predictions**: Risk scores and triage decisions
- **model_metadata**: Model training information

## Model Details

- **Algorithm**: Logistic Regression (default) or Random Forest
- **Features**: 20+ clinical features from parsed symptoms and demographics
- **Target**: Binary risk classification (high/low risk)
- **Training**: Uses sample medical dataset (can be replaced with real data)

## Testing

Run tests:
```bash
pytest tests/
```

Test coverage:
- LLM parser output structure
- Feature engineering correctness
- Risk scoring ranges
- API endpoint functionality

## Limitations

- **Educational only**: Not for real medical use
- **Sample data**: Uses synthetic training data
- **Limited symptoms**: May not cover all medical conditions
- **No validation**: Not validated against clinical standards
- **API dependency**: Requires LLM API access for full functionality

## Future Improvements

- Integration with real medical datasets
- More sophisticated feature engineering
- Additional ML models for comparison
- Real-time model retraining
- Multi-language support
- Integration with electronic health records
- Clinical validation studies

## License

This project is for educational purposes only.

## Author

**AliAbouelazm**

---

For questions or issues, please open an issue on GitHub.

