"""
Phase 5 — FastAPI Backend
Exposes the agentic pipeline via REST endpoints + WebSocket.

Endpoints:
  POST /analyze/{turbine_id}     → run full pipeline for one turbine
  POST /fleet/analyze            → run pipeline for all HIGH/CRITICAL turbines
  GET  /report/{turbine_id}      → fetch latest saved report for a turbine
  GET  /fleet/status             → fleet risk summary from Phase 4 outputs
  GET  /health                   → health check
  WS   /ws/alerts                → WebSocket stream for live alert notifications
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_pipeline, run_fleet_pipeline, get_compiled_graph
from src.rag.ingest import ingest_knowledge_base

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RISK_SCORES_PATH   = Path("outputs/risk_scores.parquet")
PREDICTIONS_PATH   = Path("outputs/pipeline_predictions.parquet")
FLEET_SUMMARY_PATH = Path("outputs/fleet_risk_summary.csv")
REPORTS_DIR        = Path("outputs/reports")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wind Turbine Predictive Maintenance — Phase 5 API",
    description="Agentic AI pipeline: RCA + RAG + Scheduler + Parts + Reports",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        data = json.dumps(message, default=str)
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.active.remove(ws)


manager = ConnectionManager()


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    risk_score:       float = Field(..., ge=0, le=1)
    risk_tier:        str   = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    anomaly_score:    float = Field(..., ge=0, le=1)
    fault_type:       str
    fault_confidence: float = Field(..., ge=0, le=1)
    rul_days:         float = Field(..., ge=0)
    rul_urgency:      str   = Field(..., pattern="^(low|medium|high|critical)$")
    sensor_snapshot:  dict  = Field(default_factory=dict)
    timestamp:        Optional[str] = None


class FleetAnalyzeRequest(BaseModel):
    min_risk_tier:    str = Field(default="HIGH", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    max_turbines:     int = Field(default=5, ge=1, le=20)


# ── Helpers ───────────────────────────────────────────────────────────────────

TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _load_risk_scores() -> pd.DataFrame:
    if not RISK_SCORES_PATH.exists():
        raise HTTPException(status_code=503, detail="risk_scores.parquet not found — run Phase 4 first")
    return pd.read_parquet(RISK_SCORES_PATH)


def _get_latest_turbine_row(df: pd.DataFrame, turbine_id: str) -> dict:
    subset = df[df["turbine_id"] == turbine_id]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"Turbine {turbine_id} not found in risk scores")
    return subset.sort_values("timestamp").iloc[-1].to_dict()


def _row_to_alert(row: dict) -> dict:
    sensor_cols = [c for c in row if c.startswith(("bearing_", "generator_", "vibration",
                                                      "oil_", "rotor_", "wind_", "power_",
                                                      "residual_", "norm_"))]
    return {
        "turbine_id":       str(row["turbine_id"]),
        "risk_score":       float(row.get("overall_risk_score", 0)),
        "risk_tier":        str(row.get("risk_tier", "LOW")),
        "anomaly_score":    float(row.get("anomaly_score", 0)),
        "fault_type":       str(row.get("predicted_fault", "normal")),
        "fault_confidence": float(row.get("fault_confidence", 0)),
        "rul_days":         float(row.get("rul_days", 90)),
        "rul_urgency":      str(row.get("rul_urgency", "low")),
        "sensor_snapshot":  {k: float(row[k]) for k in sensor_cols if pd.notna(row.get(k))},
        "timestamp":        str(row.get("timestamp", datetime.now(timezone.utc).isoformat())),
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Phase 5 API...")
    logger.info("Ensuring ChromaDB knowledge base is ingested...")
    try:
        ingest_knowledge_base(reset=False)
    except Exception as exc:
        logger.warning(f"KB ingestion skipped: {exc}")
    logger.info("Pre-compiling LangGraph...")
    get_compiled_graph()
    logger.info("Phase 5 API ready.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "5.0.0",
        "risk_scores_available": RISK_SCORES_PATH.exists(),
        "reports_dir": str(REPORTS_DIR),
        "ws_connections": len(manager.active),
    }


@app.post("/analyze/{turbine_id}")
async def analyze_turbine(
    turbine_id: str,
    request: Optional[AnalyzeRequest] = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Run the full agentic pipeline for a single turbine.
    If request body is omitted, loads latest row from risk_scores.parquet.
    """
    if request is None:
        # Load from Phase 4 outputs
        df = _load_risk_scores()
        row = _get_latest_turbine_row(df, turbine_id)
        alert = _row_to_alert(row)
    else:
        alert = {
            "turbine_id":       turbine_id,
            "risk_score":       request.risk_score,
            "risk_tier":        request.risk_tier,
            "anomaly_score":    request.anomaly_score,
            "fault_type":       request.fault_type,
            "fault_confidence": request.fault_confidence,
            "rul_days":         request.rul_days,
            "rul_urgency":      request.rul_urgency,
            "sensor_snapshot":  request.sensor_snapshot,
            "timestamp":        request.timestamp or datetime.now(timezone.utc).isoformat(),
        }

    logger.info(f"[API] /analyze/{turbine_id} → tier={alert['risk_tier']}")

    try:
        final_state = run_pipeline(**alert)
    except Exception as exc:
        logger.error(f"[API] Pipeline error for {turbine_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    report = final_state.get("report_output", {})
    json_summary = report.get("json_summary", {})

    # Broadcast to WebSocket listeners
    await manager.broadcast({
        "event":       "analysis_complete",
        "turbine_id":  turbine_id,
        "risk_tier":   alert["risk_tier"],
        "fault_type":  alert["fault_type"],
        "report_path": report.get("report_path"),
        "timestamp":   report.get("generated_at"),
    })

    return {
        "success": True,
        "turbine_id": turbine_id,
        "pipeline_errors": final_state.get("pipeline_errors", []),
        "report": json_summary,
    }


@app.post("/fleet/analyze")
async def analyze_fleet(request: FleetAnalyzeRequest):
    """
    Automatically load HIGH/CRITICAL turbines from risk_scores.parquet
    and run the agentic pipeline for each.
    """
    df = _load_risk_scores()

    # Get latest row per turbine
    latest = df.sort_values("timestamp").groupby("turbine_id").last().reset_index()

    # Filter by minimum risk tier
    min_order = TIER_ORDER[request.min_risk_tier]
    filtered = latest[
        latest["risk_tier"].map(lambda t: TIER_ORDER.get(str(t).upper(), 0)) >= min_order
    ].head(request.max_turbines)

    if filtered.empty:
        return {"success": True, "message": f"No turbines at or above {request.min_risk_tier}", "results": []}

    alerts = [_row_to_alert(row) for _, row in filtered.iterrows()]
    logger.info(f"[API] /fleet/analyze → {len(alerts)} turbines to process")

    results = run_fleet_pipeline(alerts)

    summaries = []
    for state in results:
        report = state.get("report_output", {}) or {}
        summaries.append({
            "turbine_id":      state.get("turbine_id"),
            "risk_tier":       state.get("risk_tier"),
            "fault_type":      state.get("fault_type"),
            "rul_days":        state.get("rul_days"),
            "report_path":     report.get("report_path"),
            "pipeline_errors": state.get("pipeline_errors", []),
        })

    await manager.broadcast({
        "event":           "fleet_analysis_complete",
        "turbines_processed": len(summaries),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    })

    return {"success": True, "turbines_processed": len(summaries), "results": summaries}


@app.get("/report/{turbine_id}")
async def get_latest_report(turbine_id: str, format: str = "json"):
    """
    Fetch the most recent report for a turbine.
    format=json (default) or format=markdown
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(REPORTS_DIR.glob(f"{turbine_id}_*.md"), reverse=True)

    if not reports:
        raise HTTPException(status_code=404, detail=f"No reports found for {turbine_id}")

    latest = reports[0]

    if format == "markdown":
        return {"turbine_id": turbine_id, "report_path": str(latest), "markdown": latest.read_text()}

    json_path = latest.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    return {"turbine_id": turbine_id, "report_path": str(latest), "markdown": latest.read_text()}


@app.get("/fleet/status")
async def fleet_status():
    """Return fleet risk summary from Phase 4 outputs."""
    if FLEET_SUMMARY_PATH.exists():
        df = pd.read_csv(FLEET_SUMMARY_PATH)
        return {"source": "fleet_risk_summary.csv", "data": df.to_dict(orient="records")}

    if RISK_SCORES_PATH.exists():
        df = pd.read_parquet(RISK_SCORES_PATH)
        latest = df.sort_values("timestamp").groupby("turbine_id").last().reset_index()
        cols = ["turbine_id", "risk_tier", "overall_risk_score", "predicted_fault",
                "rul_days", "rul_urgency", "timestamp"]
        cols = [c for c in cols if c in latest.columns]
        return {"source": "risk_scores.parquet", "data": latest[cols].to_dict(orient="records")}

    raise HTTPException(status_code=503, detail="No Phase 4 outputs found")


@app.get("/reports/list")
async def list_reports(turbine_id: Optional[str] = None, limit: int = 20):
    """List recent reports, optionally filtered by turbine_id."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"{turbine_id}_*.json" if turbine_id else "*.json"
    files = sorted(REPORTS_DIR.glob(pattern), reverse=True)[:limit]

    results = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append({
                "turbine_id":   data.get("turbine_id"),
                "generated_at": data.get("generated_at"),
                "risk_tier":    data.get("alert", {}).get("risk_tier"),
                "fault_type":   data.get("alert", {}).get("fault_type"),
                "report_path":  data.get("report_path"),
                "filename":     f.name,
            })
        except Exception:
            continue

    return {"count": len(results), "reports": results}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint — broadcasts pipeline completion events."""
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "event": "connected",
            "message": "Wind Turbine Alert Stream connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        while True:
            # Keep connection alive; server broadcasts on analysis completion
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"event": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)