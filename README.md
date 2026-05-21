# 🌬️ Wind Turbine Predictive Maintenance & Digital Twin

An Agentic AI-powered predictive maintenance system for wind turbines and power plant generators.

## 🏗️ Architecture
- **Digital Twin Engine** — expected vs actual behavior modeling
- **ML Models** — anomaly detection, fault classification, RUL prediction
- **Agentic AI Layer** — RCA, RAG, scheduler, parts planner, report agents
- **React Dashboard** — real-time alerts, charts, chatbot

## 🛠️ Tech Stack
Python · FastAPI · LangGraph · LangChain · XGBoost · PyTorch · ChromaDB · PostgreSQL · React · Azure

## 📁 Project Structure
\`\`\`
src/
├── data/       # preprocessing & data loaders
├── models/     # ML models
├── twin/       # digital twin engine
├── risk/       # risk scoring engine
├── agents/     # LangGraph agents
└── api/        # FastAPI backend
\`\`\`

## 🚀 Setup
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env
\`\`\`

## 📌 Status
🔨 In active development — Phase 1 (Data Generation)