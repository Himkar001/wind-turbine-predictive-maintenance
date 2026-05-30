"""
Phase 6 — FastAPI Application
==============================
Orchestrates all routers and services. Exposes:

  /health             → liveness + model readiness
  /fleet/*            → fleet status & KPI summary  (fleet router)
  /reports/*          → report listing              (fleet router)
  /predict            → live ML inference           (predictions router)
  /risk/{turbine_id}  → per-turbine risk breakdown  (predictions router)
  /turbines           → fleet turbine list          (predictions router)
  /analyze/*          → LangGraph agentic pipeline  (agents router)
  /fleet/analyze      → bulk fleet analysis         (agents router)
  /report/*           → maintenance reports         (agents router)
  /chat               → Groq LLM chatbot            (agents router)
  /alerts             → paginated alert list        (alerts router)
  /ws/alerts          → WebSocket broadcast stream  (alerts router)
  /ws/sensors/{id}    → per-turbine sensor stream   (alerts router)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Routers ───────────────────────────────────────────────────────────────────
from src.api.routes.fleet import router as fleet_router, reports_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.agents import router as agents_router
from src.api.routes.alerts import router as alerts_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RISK_SCORES_PATH   = Path("outputs/risk_scores.parquet")
FLEET_SUMMARY_PATH = Path("outputs/fleet_risk_summary.csv")
REPORTS_DIR        = Path("outputs/reports")


# ── WebSocket connection manager (shared across routers) ───────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"[WS] Connected — total active: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"[WS] Disconnected — total active: {len(self.active)}")

    async def broadcast(self, message: dict):
        data = json.dumps(message, default=str)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


manager = ConnectionManager()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="WindSense AI — Phase 6 API",
    description=(
        "Full-stack predictive maintenance platform: "
        "ML inference · LangGraph agents · Real-time WebSocket streams"
    ),
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(fleet_router)
app.include_router(reports_router)
app.include_router(predictions_router)
app.include_router(agents_router)
app.include_router(alerts_router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("WindSense AI — Phase 6 API starting …")
    logger.info("=" * 60)

    # 1. Ingest knowledge base (non-blocking — skip if it fails)
    logger.info("[Startup] Ensuring ChromaDB knowledge base is ingested …")
    try:
        from src.rag.ingest import ingest_knowledge_base
        ingest_knowledge_base(reset=False)
        logger.info("[Startup] Knowledge base ready")
    except Exception as exc:
        logger.warning(f"[Startup] KB ingestion skipped: {exc}")

    # 2. Pre-compile LangGraph
    logger.info("[Startup] Pre-compiling LangGraph …")
    try:
        from src.agents.orchestrator_agent import get_compiled_graph
        get_compiled_graph()
        logger.info("[Startup] LangGraph compiled")
    except Exception as exc:
        logger.warning(f"[Startup] LangGraph compile skipped: {exc}")

    # 3. Load ML models (async-friendly — runs in thread pool)
    logger.info("[Startup] Loading ML models (background) …")
    try:
        from src.api.services.pipeline_service import initialise_pipeline
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, initialise_pipeline)
    except Exception as exc:
        logger.warning(f"[Startup] Model load deferred: {exc}")

    logger.info("=" * 60)
    logger.info("Phase 6 API ready → http://0.0.0.0:8000/docs")
    logger.info("=" * 60)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    """Liveness + readiness probe."""
    from src.api.services.pipeline_service import (
        is_models_ready,
        get_load_error,
        get_loaded_at,
    )
    return {
        "status":                "ok",
        "version":               "6.0.0",
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "models_ready":          is_models_ready(),
        "model_load_error":      get_load_error(),
        "models_loaded_at":      get_loaded_at(),
        "risk_scores_available": RISK_SCORES_PATH.exists(),
        "fleet_summary_available": FLEET_SUMMARY_PATH.exists(),
        "reports_dir":           str(REPORTS_DIR),
        "ws_connections":        len(manager.active),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)