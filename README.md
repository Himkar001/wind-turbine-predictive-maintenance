# WindSense AI - Predictive Maintenance & Digital Twin

An advanced, Agentic AI-powered predictive maintenance system for wind turbine fleets. WindSense AI combines high-frequency telemetry, Machine Learning (Anomaly Detection, Fault Classification, Remaining Useful Life), and a LangGraph-based AI Orchestrator to autonomously detect, diagnose, and report mechanical faults before they lead to catastrophic failure.

## 🌟 Key Features

* **Real-time Fleet Dashboard**: A React + Vite frontend providing live telemetry, risk score gauges, and fleet-wide health status mapping.
* **Agentic AI Diagnostic Pipeline**: An orchestrator agent delegates tasks to specialized sub-agents:
  * **RCA Agent**: Performs Root Cause Analysis on sensor data using LLMs.
  * **RAG Agent**: Retrieves relevant SOPs and maintenance manuals via ChromaDB.
  * **Report Agent**: Synthesizes a comprehensive, actionable executive summary.
* **ML Predictive Pipeline**: Custom models for Anomaly Detection (Isolation Forest), Fault Classification (XGBoost), and RUL Prediction (Random Forest).
* **High-Performance API**: A FastAPI backend heavily optimized with PyArrow and in-memory caching to instantly process gigabytes of historical parquet data.

## 🏗️ Architecture

- **Backend**: Python 3.10, FastAPI, Uvicorn, PyArrow, Pandas
- **Frontend**: React, Vite, Tailwind CSS (via PostCSS), Recharts
- **AI & ML**: LangGraph, LangChain, Groq/OpenAI, SentenceTransformers, ChromaDB, Scikit-learn
- **Data Storage**: Parquet, CSV, JSON

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- A Groq API Key (for LLM inference)

### 1. Backend Setup

```bash
# Clone the repository
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

## 📁 Project Structure

```text
.
├── frontend/                # React Dashboard UI
│   ├── src/
│   │   ├── components/      # UI components (RiskGauge, AlertFeed, Charts)
│   │   ├── pages/           # Dashboard layout
│   │   └── index.css        # Core styling and theme tokens
├── src/
│   ├── agents/              # LangGraph AI agents (Orchestrator, RAG, RCA)
│   ├── api/                 # FastAPI routes and WebSocket handlers
│   ├── models/              # ML training and inference pipelines
│   ├── rag/                 # ChromaDB ingestion and vector search
│   └── data/                # Data simulators and processors
├── outputs/                 # Compiled parquet data files
└── data/models/             # Pickled ML models
```

## 🧠 AI Agent Workflow (Phase 6)

When a turbine enters a `HIGH` or `CRITICAL` risk tier, the backend invokes the AI pipeline:
1. **Orchestrator**: Evaluates the risk score and determines if deep analysis is required.
2. **RCA**: Ingests the 50 most recent telemetry rows and outputs a technical hypothesis.
3. **RAG**: Searches the embedded ChromaDB knowledge base for relevant manuals matching the fault.
4. **Report**: Combines the RCA hypothesis, the RAG procedures, and live metrics into a final markdown summary.

## 📄 License
MIT License