# 🌪️ WindSense AI
### Agentic AI-Powered Predictive Maintenance & Digital Twin Platform for Wind Turbines

WindSense AI is an end-to-end Industrial AI platform designed to predict equipment failures before they occur.

The system continuously monitors wind turbine telemetry, detects micro-anomalies, predicts component failures, estimates Remaining Useful Life (RUL), and autonomously generates maintenance recommendations using Agentic AI.

By combining:

- Digital Twins
- Machine Learning
- Time-Series Analytics
- Retrieval-Augmented Generation (RAG)
- Multi-Agent Systems
- Real-Time Monitoring

WindSense AI enables maintenance teams to transition from:

**Reactive Maintenance → Preventive Maintenance → Predictive Maintenance → Autonomous Maintenance**

---

# 🎯 Business Problem

Unexpected turbine failures can result in:

- Expensive emergency repairs
- Unplanned downtime
- Loss of energy production
- Safety risks
- Reduced turbine lifespan

Traditional monitoring systems generate thousands of alerts daily but provide little insight into:

- Why the issue occurred
- Which component is failing
- How urgently maintenance is required
- What actions should be taken

WindSense AI addresses these challenges through intelligent diagnostics and autonomous decision support.

---

# 🚀 Key Features

## Digital Twin Simulation

Creates a virtual representation of turbine behavior and continuously compares:

Expected Behavior vs Actual Behavior

This enables early detection of hidden degradation patterns.

---

## Anomaly Detection

Isolation Forest identifies abnormal operating conditions before visible failures occur.

Examples:

- Bearing temperature spikes
- Vibration abnormalities
- Oil pressure drops
- Generator overheating

---

## Fault Classification

XGBoost predicts the most probable fault category.

Supported Fault Types:

- Bearing Failure
- Gearbox Failure
- Generator Overheating
- Blade Imbalance
- Oil Leakage
- Electrical Fault

---

## Remaining Useful Life (RUL)

Predicts:

> How many days remain before a component reaches critical failure.

Maintenance teams can schedule repairs proactively.

---

## Risk Scoring Engine

Combines:

- Anomaly Score
- Fault Confidence
- RUL Prediction
- Digital Twin Deviation

to produce:

| Risk Score | Tier |
|------------|------|
| 0.0–0.25 | LOW |
| 0.25–0.50 | MEDIUM |
| 0.50–0.75 | HIGH |
| 0.75–1.00 | CRITICAL |

---

# 🤖 Agentic AI Layer

Unlike traditional predictive maintenance systems, WindSense AI includes autonomous AI agents.

## Orchestrator Agent

Coordinates all agents using LangGraph.

Responsibilities:

- Monitor risk levels
- Trigger investigations
- Route tasks

---

## RCA Agent

Performs Root Cause Analysis.

Example:

"Rapid bearing temperature increase combined with declining oil pressure suggests lubrication failure."

---

## RAG Agent

Retrieves:

- OEM Manuals
- SOPs
- Maintenance Procedures
- Historical Repair Logs

from ChromaDB.

---

## Parts Planning Agent

Determines:

- Required replacement parts
- Inventory needs
- Maintenance preparation

---

## Report Agent

Generates executive maintenance reports automatically.

Output includes:

- Fault diagnosis
- Risk level
- RUL estimate
- Maintenance recommendations
- Parts list

---

# 🏗 System Architecture

```mermaid
flowchart TD

A[SCADA Sensor Data]
--> B[Data Preprocessing]

B --> C[Digital Twin Engine]

B --> D[Isolation Forest]

C --> D

D --> E[Fault Classifier]

D --> F[RUL Predictor]

E --> G[Risk Scoring Engine]
F --> G

G --> H[LangGraph Orchestrator]

H --> I[RCA Agent]
H --> J[RAG Agent]
H --> K[Parts Agent]

I --> L[Report Agent]
J --> L
K --> L

L --> M[FastAPI Backend]

M --> N[React Dashboard]
````

---

# 📊 Dataset Overview

### Fleet

* 5 Wind Turbines
* WTG-001 → WTG-005

### Telemetry Volume

* 7.7+ Million Records
* Multi-Year Operational History

### Sensor Streams

| Mechanical          | Electrical      |
| ------------------- | --------------- |
| Rotor Speed         | Active Power    |
| Bearing Temperature | Generator Speed |
| Gearbox Temperature | Power Output    |
| Oil Pressure        | Voltage Metrics |

Additional:

* Vibration Sensors
* Pitch Angle
* Wind Speed
* Oil Temperature

---

# 🔬 Feature Engineering

### Statistical Features

* Rolling Mean
* Rolling Std
* EWMA

### Trend Features

* Rate of Change
* Temperature Gradient
* Vibration Growth

### Digital Twin Features

* Residual Error
* Normalized Residual
* Twin Deviation Score

### Health Indicators

* Fault Signatures
* Degradation Scores
* Historical Failure Distance

---

# 🧠 Machine Learning Stack

| Task                 | Model            |
| -------------------- | ---------------- |
| Anomaly Detection    | Isolation Forest |
| Fault Classification | XGBoost          |
| RUL Prediction       | Random Forest    |
| Root Cause Analysis  | Groq Llama       |
| Knowledge Retrieval  | ChromaDB         |
| Agent Orchestration  | LangGraph        |

---

# 🌐 Technology Stack

## AI / ML

* Scikit-Learn
* XGBoost
* LangGraph
* ChromaDB
* Groq Llama

## Backend

* FastAPI
* WebSockets

## Frontend

* React
* Vite
* TailwindCSS

## Data Processing

* Pandas
* PyArrow
* NumPy

---

# 📁 Project Structure

```text
wind-turbine-predictive-maintenance/

├── src/
│   ├── agents/
│   ├── api/
│   ├── models/
│   ├── data/
│   ├── twin/
│   └── rag/
│
├── frontend/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── configs/
│
└── outputs/
```

---

# 🚀 Running the Project

## Backend

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn src.api.main:app --reload
```

## Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# 🔮 Future Enhancements

* Kafka Real-Time Streaming
* Live Digital Twin Dashboard
* Transformer-Based Anomaly Detection
* Multi-Agent Maintenance Scheduling
* Inventory Optimization Agent
* Azure Cloud Deployment
* Predictive Maintenance Chatbot
* Autonomous Work Order Generation

---

# Author 
 HIMKAR VASHISTHA
 