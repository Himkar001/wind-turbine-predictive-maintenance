"""
Phase 6 — Agents Router
=========================
Endpoints:
  POST /analyze/{turbine_id}   → full LangGraph agentic pipeline for one turbine
  POST /fleet/analyze          → pipeline for all HIGH/CRITICAL turbines
  GET  /report/{turbine_id}    → fetch latest saved report
  POST /chat                   → single-turn chatbot Q&A (Groq LLM, no pipeline)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])

RISK_SCORES_PATH   = Path("outputs/risk_scores.parquet")
FLEET_SUMMARY_PATH = Path("outputs/fleet_risk_summary.csv")
REPORTS_DIR        = Path("outputs/reports")

TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# ── Request models ────────────────────────────────────────────────────────────

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
    min_risk_tier: str = Field(default="HIGH", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    max_turbines:  int = Field(default=5, ge=1, le=20)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    turbine_id: Optional[str] = None
    context: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    sensor_cols = [
        c for c in row
        if c.startswith(("bearing_", "generator_", "vibration",
                          "oil_", "rotor_", "wind_", "power_",
                          "residual_", "norm_"))
    ]
    return {
        "turbine_id":       str(row["turbine_id"]),
        "risk_score":       float(row.get("overall_risk_score", 0)),
        "risk_tier":        str(row.get("risk_tier", "LOW")),
        "anomaly_score":    float(row.get("anomaly_score", 0)),
        "fault_type":       str(row.get("predicted_fault_type", "normal")),
        "fault_confidence": float(row.get("fault_confidence", 0)),
        "rul_days":         float(row.get("rul_days", 90)),
        "rul_urgency":      str(row.get("rul_urgency", "low")),
        "sensor_snapshot":  {k: float(row[k]) for k in sensor_cols if pd.notna(row.get(k))},
        "timestamp":        str(row.get("timestamp", datetime.now(timezone.utc).isoformat())),
    }


# ── POST /analyze/{turbine_id} ────────────────────────────────────────────────

@router.post("/analyze/{turbine_id}")
async def analyze_turbine(
    turbine_id: str,
    request: Optional[AnalyzeRequest] = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Run the full LangGraph agentic pipeline for a single turbine.
    If request body is omitted, loads latest row from risk_scores.parquet.
    """
    from src.agents.orchestrator_agent import run_pipeline
    from src.api.main import manager  # WebSocket broadcast

    if request is None:
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

    logger.info(f"[Agents] /analyze/{turbine_id} tier={alert['risk_tier']}")

    try:
        final_state = run_pipeline(**alert)
    except Exception as exc:
        logger.error(f"[Agents] Pipeline error for {turbine_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    report = final_state.get("report_output", {}) or {}
    json_summary = report.get("json_summary", {})

    await manager.broadcast({
        "event":       "analysis_complete",
        "turbine_id":  turbine_id,
        "risk_tier":   alert["risk_tier"],
        "fault_type":  alert["fault_type"],
        "report_path": report.get("report_path"),
        "timestamp":   report.get("generated_at"),
    })

    return {
        "success":         True,
        "turbine_id":      turbine_id,
        "pipeline_errors": final_state.get("pipeline_errors", []),
        "report":          json_summary,
    }


# ── POST /fleet/analyze ───────────────────────────────────────────────────────

@router.post("/fleet/analyze")
async def analyze_fleet(request: FleetAnalyzeRequest):
    """
    Auto-load HIGH/CRITICAL turbines from risk_scores.parquet and run the
    agentic pipeline for each.
    """
    from src.agents.orchestrator_agent import run_fleet_pipeline
    from src.api.main import manager

    df = _load_risk_scores()
    latest = df.sort_values("timestamp").groupby("turbine_id").last().reset_index()

    min_order = TIER_ORDER[request.min_risk_tier]
    filtered = latest[
        latest["risk_tier"].map(lambda t: TIER_ORDER.get(str(t).upper(), 0)) >= min_order
    ].head(request.max_turbines)

    if filtered.empty:
        return {
            "success": True,
            "message": f"No turbines at or above {request.min_risk_tier}",
            "results": [],
        }

    alerts = [_row_to_alert(row) for _, row in filtered.iterrows()]
    logger.info(f"[Agents] /fleet/analyze → {len(alerts)} turbines")

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
        "event":              "fleet_analysis_complete",
        "turbines_processed": len(summaries),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    })

    return {"success": True, "turbines_processed": len(summaries), "results": summaries}


# ── GET /report/{turbine_id} ──────────────────────────────────────────────────

@router.get("/report/{turbine_id}")
async def get_latest_report(turbine_id: str, format: str = "json"):
    """Fetch the most recent maintenance report for a turbine."""
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


# ── POST /chat ────────────────────────────────────────────────────────────────

_CHAT_SYSTEM = """You are WindSense AI, an expert assistant for wind turbine predictive maintenance.
You help operations engineers interpret sensor data, understand fault predictions, plan maintenance schedules,
and explain what ML model outputs mean in plain English.

When given a turbine_id and context (risk score, fault type, RUL, sensor readings), you answer concisely
and practically — like a knowledgeable colleague, not a textbook. Keep answers under 200 words unless
the user asks for detail. Use bullet points for lists. Be direct about urgency when risk is HIGH or CRITICAL."""


def _build_fleet_context(turbine_id: str = None) -> str:
    """
    Build a grounded context string from real parquet/CSV data.
    Always includes full fleet summary + selected turbine detail if provided.
    """
    lines = []

    # ── Full fleet summary ────────────────────────────────────────────────────
    try:
        if FLEET_SUMMARY_PATH.exists():
            import pandas as pd
            fleet_df = pd.read_csv(FLEET_SUMMARY_PATH)
            lines.append("=== FLEET DATA (from ML pipeline outputs) ===")
            lines.append(f"Total turbines in fleet: {len(fleet_df)}")
            lines.append("Turbine rankings (rank 1 = most at risk):\n")

            for _, row in fleet_df.sort_values("fleet_rank").iterrows():
                tid          = str(row.get("turbine_id", "?"))
                mean_risk    = float(row.get("mean_risk_score", 0))
                max_risk     = float(row.get("max_risk_score", 0))
                p95_risk     = float(row.get("p95_risk_score", 0))
                crit_count   = int(row.get("critical_count", 0))
                high_count   = int(row.get("high_count", 0))
                min_rul      = float(row.get("min_rul_days", 0))
                dom_fault    = str(row.get("dominant_fault", "none"))
                rank         = int(row.get("fleet_rank", 0))

                lines.append(
                    f"  Rank #{rank}: {tid}\n"
                    f"    - Mean Risk Score : {mean_risk:.3f}\n"
                    f"    - Max Risk Score  : {max_risk:.3f}\n"
                    f"    - P95 Risk Score  : {p95_risk:.3f}\n"
                    f"    - Critical Alerts : {crit_count}\n"
                    f"    - High Alerts     : {high_count}\n"
                    f"    - Min RUL (days)  : {min_rul:.1f}\n"
                    f"    - Dominant Fault  : {dom_fault}"
                )
        else:
            lines.append("=== FLEET DATA ===")
            lines.append("fleet_risk_summary.csv not found — run Phase 4 first.")
    except Exception as exc:
        lines.append(f"=== FLEET DATA ===\nError loading fleet data: {exc}")

    # ── Selected turbine detail ─────────────────────────────────────────────────────────
    if turbine_id and RISK_SCORES_PATH.exists():
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(RISK_SCORES_PATH)
            available = set(schema.names)
            want_cols = [
                "turbine_id", "timestamp", "overall_risk_score", "risk_tier",
                "anomaly_score", "predicted_fault_type", "fault_confidence",
                "rul_days", "rul_urgency", "dominant_signal"
            ]
            safe_cols = [c for c in want_cols if c in available]

            table = pq.read_table(
                RISK_SCORES_PATH,
                columns=safe_cols,
                filters=[("turbine_id", "==", turbine_id)],
            )
            df = table.to_pandas()
            if not df.empty:
                # Use peak risk row for fault context, latest row for current state
                row = df.sort_values("overall_risk_score", ascending=False).iloc[0]
                latest_row = df.sort_values("timestamp").iloc[-1]
                lines.append(f"\n=== SELECTED TURBINE DETAIL: {turbine_id} ===")
                lines.append(f"  Current Risk Score  : {float(latest_row.get('overall_risk_score', 0)):.4f} (latest reading)")
                lines.append(f"  Peak Risk Score     : {float(row.get('overall_risk_score', 0)):.4f} (worst recorded)")
                lines.append(f"  Current Risk Tier   : {latest_row.get('risk_tier', 'N/A')}")
                lines.append(f"  Current Anomaly     : {float(latest_row.get('anomaly_score', 0)):.4f}")
                lines.append(f"  Peak Fault Detected : {row.get('predicted_fault_type', 'N/A')} ({float(row.get('fault_confidence', 0)):.0%} confidence)")
                lines.append(f"  Current RUL (days)  : {float(latest_row.get('rul_days', 0)):.1f}")
                lines.append(f"  RUL Urgency         : {latest_row.get('rul_urgency', 'N/A')}")
                lines.append(f"  Dominant Signal     : {row.get('dominant_signal', 'N/A')}")
                lines.append(f"  Last Reading        : {latest_row.get('timestamp', 'N/A')}")
            else:
                lines.append(f"\n=== SELECTED TURBINE: {turbine_id} ===\nNo data found for this turbine ID.")
        except Exception as exc:
            lines.append(f"\n=== SELECTED TURBINE: {turbine_id} ===\nError loading turbine data: {exc}")

    return "\n".join(lines)


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Single-turn chatbot Q&A powered by Groq LLM.
    Optionally enriched with turbine context from risk_scores.parquet.
    Fast (~1-2s) — no LangGraph pipeline overhead.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from dotenv import load_dotenv
    load_dotenv()

    # Build fully grounded context — always includes full fleet data
    fleet_context = _build_fleet_context(turbine_id=request.turbine_id)

    # Append any extra context the frontend passed
    extra = ""
    if request.context:
        extra = f"\n=== ADDITIONAL CONTEXT ===\n{json.dumps(request.context, indent=2)}"

    user_message = (
        f"{fleet_context}{extra}\n\n"
        f"=== USER QUESTION ===\n{request.message}"
    )

    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
        messages = [
            SystemMessage(content=_CHAT_SYSTEM),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        answer = response.content

    except Exception as exc:
        logger.error(f"[Chat] LLM error: {exc}")
        raise HTTPException(status_code=500, detail=f"Chat LLM error: {exc}")

    return {
        "turbine_id": request.turbine_id,
        "question":   request.message,
        "answer":     answer,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
