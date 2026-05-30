# WindSense AI - Predictive Maintenance & Digital Twin

An advanced, Agentic AI-powered predictive maintenance system for wind turbine fleets. WindSense AI combines high-frequency telemetry, Machine Learning (Anomaly Detection, Fault Classification, Remaining Useful Life), and a LangGraph-based AI Orchestrator to autonomously detect, diagnose, and report mechanical faults before they lead to catastrophic failure.

## 🏗️ System Architecture

```mermaid
flowchart TD
    %% Data Layer
    subgraph Data Layer
        A[Simulated SCADA Telemetry] --> B(Data Preprocessing & Feature Engineering)
        B --> C[(Parquet Data Store)]
    end

    %% ML Pipeline
    subgraph ML Pipeline
        C --> D{Digital Twin Engine}
        C --> E[Anomaly Detection<br/>Isolation Forest]
        D --> E
        E --> F[Fault Classification<br/>XGBoost]
        E --> G[RUL Prediction<br/>Random Forest]
        F --> H((Risk Scoring Engine))
        G --> H
        H --> I[(Risk Scores Parquet)]
    end

    %% AI Agents Layer
    subgraph AI Agents (LangGraph)
        I -.->|Triggers on HIGH risk| J[Orchestrator Agent]
        J --> K[RCA Agent<br/>Groq Llama-3]
        J --> L[RAG Agent<br/>ChromaDB]
        K --> M[Report Agent<br/>Executive Summary]
        L --> M
    end

    %% Application Layer
    subgraph Application Layer
        I --> N[FastAPI Backend]
        M --> N
        N <-->|REST & WebSockets| O[React Dashboard<br/>Vite + Tailwind]
    end
```

## 📊 Data Pipeline & Feature Engineering

### 1. The Dataset
The project uses high-frequency simulated SCADA telemetry for a fleet of 5 wind turbines (WTG-001 to WTG-005) recorded at 10-minute intervals over a 3-year period (over 7.7 million rows).
**Core Sensors Tracked**:
- `wind_speed`, `rotor_speed`, `generator_speed`
- `active_power`, `pitch_angle`
- `gearbox_temperature`, `bearing_temperature`
- `vibration_nacelle`, `vibration_generator`
- `oil_pressure`, `oil_temperature`

### 2. Feature Engineering
Raw data is processed using heavily optimized PyArrow and Pandas pipelines. Engineered features include:
- **Rolling Windows**: 3-hour, 24-hour, and 7-day rolling means and standard deviations to capture short-term spikes and long-term degradation.
- **Exponentially Weighted Moving Averages (EWMA)**: To smooth noise while retaining sensitivity to sudden shifts.
- **Differentials & Gradients**: Rate of change for temperatures (`gearbox_temp_diff`, `bearing_temp_diff`).
- **Digital Twin Residuals**: A physics-informed digital twin models the *expected* behavior (e.g., expected active power given current wind speed). The residuals (Actual - Expected) are fed as features to the ML models.

## 🧠 Machine Learning Models

### Anomaly Detection (Isolation Forest)
Detects when a turbine deviates from normal operational boundaries.
- **Target Features**: Digital twin residuals, current temperatures, vibration metrics, and 3-hour rolling averages.
- **Output**: Anomaly Score (-1 to 1) and binary Anomaly Flag.

### Fault Classification (XGBoost)
When an anomaly is detected, this model classifies the specific type of failure.
- **Target Features**: Temperature gradients, cross-sensor ratios (e.g., generator speed vs. active power), vibration standard deviations, and oil pressure anomalies.
- **Fault Classes**: `gearbox_fault`, `generator_fault`, `bearing_fault`, `pitch_fault`, `normal`.

### Remaining Useful Life (RUL) Prediction (Random Forest)
Estimates the number of days until a component reaches critical failure.
- **Target Features**: Cumulative degradation signatures (running sum of anomaly scores), 7-day rolling standard deviations, and distance from historical failure thresholds.
- **Output**: Estimated RUL in days.

### Risk Scoring Engine
A heuristic engine that combines the Anomaly Score, Fault Confidence, and RUL into a normalized `overall_risk_score` (0.0 to 1.0). This maps directly to risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

## 🤖 Agentic AI Workflow (LangGraph)

The system features an autonomous diagnostic workflow powered by LangGraph, Groq (Llama-3), and ChromaDB.

1. **Orchestrator Agent**: Monitors the Risk Scores. If a turbine enters `HIGH` or `CRITICAL` status, it triggers the diagnostic pipeline.
2. **RCA (Root Cause Analysis) Agent**: Ingests the 50 most recent SCADA telemetry rows for the anomalous turbine. Uses LLM reasoning to hypothesize the physical cause of the fault based on sensor trends (e.g., "rapid temperature rise combined with oil pressure drop indicates bearing lubrication failure").
3. **RAG Agent**: Takes the predicted fault class and searches a ChromaDB vector store containing OEM maintenance manuals, SOPs, and historical repair logs to retrieve the exact repair procedure.
4. **Report Agent**: Synthesizes the RCA hypothesis, RAG repair procedures, and live metrics into a final markdown executive summary, delivered instantly to the React dashboard.

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- A Groq API Key (for LLM inference)

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The application will be accessible at `http://localhost:3000`.

## 📄 License
MIT License