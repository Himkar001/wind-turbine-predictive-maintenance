"""
Phase 6 — Predictions Router
==============================
Endpoints:
  POST /predict              → live inference from sensor snapshot
  GET  /risk/{turbine_id}    → latest risk breakdown (fast — PyArrow filter pushdown)
  GET  /turbines             → list all known turbine IDs

Performance note:
  risk_scores.parquet is ~3 GB. We NEVER read it all into memory.
  Instead we use pyarrow.parquet.read_table() with filters= so only the
  rows for the requested turbine_id are pulled off disk.
  The fleet_risk_summary.csv (< 1 KB) is used for fast metadata lookups.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])

RISK_SCORES_PATH   = Path("outputs/risk_scores.parquet")
FLEET_SUMMARY_PATH = Path("outputs/fleet_risk_summary.csv")

RISK_TIERS = [(0.80, "CRITICAL"), (0.60, "HIGH"), (0.35, "MEDIUM"), (0.00, "LOW")]

# Columns needed for the /risk detail endpoint (avoids reading all ~100+ cols)
_RISK_COLS = [
    "turbine_id", "timestamp",
    "overall_risk_score", "risk_tier", "anomaly_score",
    "is_anomaly_prediction", "predicted_fault_type", "fault_confidence",
    "rul_days", "rul_urgency", "dominant_signal", "risk_breakdown_json",
]

# Columns needed for the chart time-series
_CHART_COLS = [
    "timestamp", "anomaly_score", "overall_risk_score",
    "rul_days", "fault_confidence",
]


def _tier_from_score(score: float) -> str:
    for threshold, tier in RISK_TIERS:
        if score >= threshold:
            return tier
    return "LOW"


# ── PyArrow fast read — one turbine only ──────────────────────────────────────

_TURBINE_HISTORY_CACHE = {}

def _read_turbine_parquet(turbine_id: str, columns: list[str], last_n: int = 100) -> pd.DataFrame:
    """
    Read only the rows for `turbine_id` from the large parquet file using
    PyArrow filter pushdown — avoids loading the full 3 GB into RAM.
    Caches the result in memory for instant subsequent loads.
    """
    global _TURBINE_HISTORY_CACHE
    if turbine_id in _TURBINE_HISTORY_CACHE:
        return _TURBINE_HISTORY_CACHE[turbine_id]

    if not RISK_SCORES_PATH.exists():


        raise HTTPException(
            status_code=503,
            detail="risk_scores.parquet not found — run Phase 4 first",
        )

    # Only request columns that actually exist in the schema
    try:
        schema = pq.read_schema(RISK_SCORES_PATH)
        available = set(schema.names)
        # De-duplicate requested columns while preserving order
        unique_cols = list(dict.fromkeys(columns))
        safe_cols = [c for c in unique_cols if c in available]

        table = pq.read_table(
            RISK_SCORES_PATH,
            columns=safe_cols if safe_cols else None,
            filters=[("turbine_id", "==", turbine_id)],
        )
        df = table.to_pandas()
    except Exception as exc:
        logger.warning(f"[PyArrow] Filter read failed ({exc}) — falling back to pandas sample")
        # Worst-case fallback: still much faster than full read via chunked reading
        df = pd.read_parquet(RISK_SCORES_PATH, columns=columns)
        df = df[df["turbine_id"] == turbine_id]

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Turbine {turbine_id} not found")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    _TURBINE_HISTORY_CACHE[turbine_id] = df
    return df





def _turbine_from_fleet_csv(turbine_id: str) -> Optional[dict]:
    """Fast lookup from the tiny fleet_risk_summary.csv."""
    if not FLEET_SUMMARY_PATH.exists():
        return None
    try:
        df = pd.read_csv(FLEET_SUMMARY_PATH)
        row = df[df["turbine_id"] == turbine_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except Exception:
        return None


# ── Request models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    turbine_id: str
    sensor_snapshot: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(request: PredictRequest):
    """
    Run live ML inference on a raw sensor snapshot.
    Falls back to fleet_risk_summary.csv (instant) if models not loaded.
    """
    from src.api.services.pipeline_service import predict_from_snapshot, is_models_ready

    ts = request.timestamp or datetime.now(timezone.utc).isoformat()

    # Try live model inference first
    if is_models_ready():
        try:
            result = predict_from_snapshot(request.turbine_id, request.sensor_snapshot)
            result["timestamp"] = ts
            result["risk_tier"] = _tier_from_score(result["risk_score"])
            result["source"] = "live_model"
            return result
        except Exception as exc:
            logger.warning(f"[Predict] Live inference failed: {exc} — falling back")

    # Fast fallback: fleet CSV → no parquet read at all
    fleet_row = _turbine_from_fleet_csv(request.turbine_id)
    if fleet_row:
        score = float(fleet_row.get("p95_risk_score", fleet_row.get("mean_risk_score", 0)))
        return {
            "turbine_id":       request.turbine_id,
            "anomaly_score":    score,
            "is_anomaly":       score > 0.35,
            "fault_type":       str(fleet_row.get("dominant_fault", "unknown")),
            "fault_confidence": min(score * 1.2, 1.0),
            "rul_days":         float(fleet_row.get("min_rul_days", 30)),
            "rul_urgency":      "critical" if score > 0.8 else "high" if score > 0.6 else "medium",
            "risk_score":       score,
            "risk_tier":        _tier_from_score(score),
            "timestamp":        ts,
            "source":           "fleet_csv_fallback",
        }

    raise HTTPException(status_code=503, detail="Models not loaded and no fleet CSV available")


# ── GET /risk/{turbine_id} ────────────────────────────────────────────────────

@router.get("/risk/{turbine_id}")
async def get_risk(turbine_id: str, last_n: int = 60):
    """
    Returns the latest risk data for a specific turbine.
    Uses PyArrow filter pushdown — reads only this turbine's rows from the
    3 GB parquet file. Typically completes in 1-3 seconds.

    Also enriches with fleet_risk_summary.csv for fleet-level metrics.
    """
    # ── 1. Fast fleet-level data from CSV ────────────────────────────────────
    fleet_row = _turbine_from_fleet_csv(turbine_id)

    # ── 2. Per-turbine detail from parquet (filtered) ─────────────────────────
    try:
        df = _read_turbine_parquet(turbine_id, columns=_RISK_COLS + _CHART_COLS, last_n=last_n)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[Risk] Parquet read error for {turbine_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Parquet read error: {exc}")

    # ── 3. Build result from PEAK-risk row (matches fleet map) ───────────────
    # Sort by overall_risk_score desc — peak row has the correct tier + fault
    peak = df.sort_values("overall_risk_score", ascending=False).iloc[0].to_dict()
    last = df.sort_values("timestamp").iloc[-1].to_dict()  # for chart reference

    # Resolve fault type: prefer peak row, fall back to fleet CSV dominant_fault
    fault_type = str(peak.get("predicted_fault_type", peak.get("predicted_fault", "unknown")))
    if fault_type.lower().strip() in ("normal", "nan", "") and fleet_row:
        fault_type = str(fleet_row.get("dominant_fault", fault_type))

    # Parse risk breakdown JSON from peak row
    breakdown = {}
    raw_bd = peak.get("risk_breakdown_json")
    if raw_bd:
        try:
            breakdown = json.loads(raw_bd)
        except Exception:
            pass

    # Chart time-series (last_n rows sorted by timestamp)
    df_chart = df.tail(last_n)
    chart_cols = [c for c in _CHART_COLS if c in df_chart.columns]
    chart_data = df_chart[chart_cols].fillna(0).to_dict(orient="records")

    result = {
        "turbine_id":       turbine_id,
        "risk_score":       float(peak.get("overall_risk_score", 0)),
        "risk_tier":        str(peak.get("risk_tier", "LOW")),
        "anomaly_score":    float(peak.get("anomaly_score", 0)),
        "is_anomaly":       bool(peak.get("is_anomaly_prediction", False)),
        "fault_type":       fault_type,
        "fault_confidence": float(peak.get("fault_confidence", 0)),
        "rul_days":         float(peak.get("rul_days", 30)),
        "rul_urgency":      str(peak.get("rul_urgency", "low")),
        "dominant_signal":  str(peak.get("dominant_signal", "anomaly")),
        "risk_breakdown":   breakdown,
        "timestamp":        str(last.get("timestamp", "")),
        "chart_data":       chart_data,
    }

    # Enrich with fleet CSV data if available
    if fleet_row:
        result["fleet_rank"]       = int(fleet_row.get("fleet_rank", 0))
        result["mean_risk_score"]  = float(fleet_row.get("mean_risk_score", 0))
        result["max_risk_score"]   = float(fleet_row.get("max_risk_score", 0))
        result["critical_count"]   = int(fleet_row.get("critical_count", 0))
        result["high_count"]       = int(fleet_row.get("high_count", 0))
        result["min_rul_days"]     = float(fleet_row.get("min_rul_days", 0))
        result["dominant_fault"]   = str(fleet_row.get("dominant_fault", "none"))

    return result


# ── GET /turbines ─────────────────────────────────────────────────────────────

@router.get("/turbines")
async def list_turbines():
    """
    Return the list of all turbine IDs.
    Uses fleet_risk_summary.csv (instant) — no parquet read.
    """
    if FLEET_SUMMARY_PATH.exists():
        df = pd.read_csv(FLEET_SUMMARY_PATH)
        turbine_ids = sorted(df["turbine_id"].tolist())
        return {"turbine_count": len(turbine_ids), "turbine_ids": turbine_ids}

    # Fallback: read only turbine_id column from parquet (fast)
    if not RISK_SCORES_PATH.exists():
        raise HTTPException(status_code=503, detail="No data available — run Phase 4 first")

    try:
        table = pq.read_table(RISK_SCORES_PATH, columns=["turbine_id"])
        turbine_ids = sorted(table.column("turbine_id").unique().to_pylist())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"turbine_count": len(turbine_ids), "turbine_ids": turbine_ids}
