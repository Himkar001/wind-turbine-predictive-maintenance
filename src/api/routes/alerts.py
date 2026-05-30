"""
Phase 6 — Alerts Router
=========================
Endpoints:
  GET /alerts                      → paginated list of recent alerts
  WS  /ws/alerts                   → broadcast stream (analysis events + periodic pings)
  WS  /ws/sensors/{turbine_id}     → live sensor reading stream (replays parquet data)

Performance note:
  risk_scores.parquet is ~3 GB. All reads here use PyArrow with column
  selection and filter pushdown — never the full file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

# Columns needed for the alerts list — avoids reading all ~100 cols from 3 GB
_ALERT_COLS = [
    "turbine_id", "timestamp", "overall_risk_score", "risk_tier",
    "anomaly_score", "predicted_fault_type", "fault_confidence",
    "rul_days", "rul_urgency",
]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["alerts"])

RISK_SCORES_PATH = Path("outputs/risk_scores.parquet")

TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
TIER_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#eab308",
    "LOW":      "#22c55e",
}

# Sensor columns to stream in the live feed
_STREAM_SENSORS = [
    "bearing_temp_1", "bearing_temp_2",
    "generator_temp", "gearbox_temp",
    "vibration_nacelle_x", "vibration_nacelle_y",
    "rotor_speed", "power_output",
    "oil_pressure", "wind_speed",
    "overall_risk_score", "anomaly_score",
]


# ── GET /alerts ───────────────────────────────────────────────────────────────

_ALERT_CACHE = None

@router.get("/alerts")
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    min_tier: str = Query("MEDIUM", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    turbine_id: Optional[str] = Query(None),
):
    """
    Return the most recent HIGH/CRITICAL anomaly alerts.
    Cached in memory to prevent blocking reads of the 3GB parquet file.
    """
    global _ALERT_CACHE
    if not RISK_SCORES_PATH.exists():
        raise HTTPException(status_code=503, detail="risk_scores.parquet not found — run Phase 4 first")

    # Load dominant fault per turbine from fleet CSV (instant, < 1KB file)
    fleet_fault_map: dict = {}
    fleet_csv = Path("outputs/fleet_risk_summary.csv")
    if fleet_csv.exists():
        try:
            import pandas as _pd
            _csv = _pd.read_csv(fleet_csv)
            if "dominant_fault" in _csv.columns:
                fleet_fault_map = _csv.set_index("turbine_id")["dominant_fault"].to_dict()
        except Exception:
            pass

    if _ALERT_CACHE is None:
        try:
            # Build pyarrow filters
            pa_filters = []
            # We don't filter turbine_id here so we can cache the whole fleet
            
            schema = pq.read_schema(RISK_SCORES_PATH)
            available = set(schema.names)
            safe_cols = [c for c in _ALERT_COLS if c in available]

            table = pq.read_table(
                RISK_SCORES_PATH,
                columns=safe_cols,
            )
            df = table.to_pandas()
            
            # Sort newest-first globally once
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp", ascending=False)
            
            _ALERT_CACHE = df
            
        except Exception as exc:
            logger.error(f"[Alerts] PyArrow read error: {exc}")
            raise HTTPException(status_code=500, detail=f"Parquet read error: {exc}")

    df = _ALERT_CACHE

    # Apply filters on the cached dataframe
    if turbine_id:
        df = df[df["turbine_id"] == turbine_id]

    # Filter by minimum tier
    min_order = TIER_ORDER.get(min_tier.upper(), 0)
    if "risk_tier" in df.columns:
        df = df[
            df["risk_tier"].map(lambda t: TIER_ORDER.get(str(t).upper(), 0)) >= min_order
        ]

    df = df.head(limit)


    alerts = []
    for _, row in df.iterrows():
        tid = str(row.get("turbine_id", "?"))
        tier = str(row.get("risk_tier", "LOW")).upper()
        raw_fault = str(row.get("predicted_fault_type", row.get("predicted_fault", "normal")))
        # Enrich 'normal' label using fleet dominant fault so alerts are informative
        if raw_fault.lower().strip() in ("normal", "nan", ""):
            raw_fault = fleet_fault_map.get(tid, "anomaly_detected")
        alerts.append({
            "turbine_id":       tid,
            "timestamp":        str(row.get("timestamp", "")),
            "risk_score":       round(float(row.get("overall_risk_score", 0)), 4),
            "risk_tier":        tier,
            "tier_color":       TIER_COLORS.get(tier, "#6b7280"),
            "anomaly_score":    round(float(row.get("anomaly_score", 0)), 4),
            "fault_type":       raw_fault,
            "fault_confidence": round(float(row.get("fault_confidence", 0)), 4),
            "rul_days":         round(float(row.get("rul_days", 30)), 2),
            "rul_urgency":      str(row.get("rul_urgency", "low")),
        })

    return {
        "count":      len(alerts),
        "min_tier":   min_tier,
        "turbine_id": turbine_id,
        "alerts":     alerts,
    }


# ── WS /ws/alerts ─────────────────────────────────────────────────────────────

@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket alert stream.
    - Broadcasts pipeline completion events from the manager (set in main.py)
    - Sends periodic fleet-status pings every 15 seconds
    """
    from src.api.main import manager

    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "event":     "connected",
            "message":   "WindSense AI Alert Stream connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

        tick = 0
        while True:
            await asyncio.sleep(15)
            tick += 1

            # Send a fleet heartbeat every 15s — read only 2 cols via PyArrow
            try:
                if RISK_SCORES_PATH.exists():
                    table = pq.read_table(
                        RISK_SCORES_PATH,
                        columns=["turbine_id", "overall_risk_score"],
                    )
                    df_hb = table.to_pandas()
                    latest = df_hb.groupby("turbine_id")["overall_risk_score"].max().to_dict()
                    await websocket.send_text(json.dumps({
                        "event":       "fleet_heartbeat",
                        "tick":        tick,
                        "fleet_risks": {k: round(float(v), 4) for k, v in latest.items()},
                        "timestamp":   datetime.now(timezone.utc).isoformat(),
                    }))
                else:
                    await websocket.send_text(json.dumps({"event": "ping", "tick": tick}))
            except Exception as exc:
                logger.warning(f"[WS/alerts] Heartbeat error: {exc}")
                await websocket.send_text(json.dumps({"event": "ping", "tick": tick}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("[WS/alerts] Client disconnected")
    except Exception as exc:
        logger.error(f"[WS/alerts] Unexpected error: {exc}")
        manager.disconnect(websocket)


# ── WS /ws/sensors/{turbine_id} ───────────────────────────────────────────────

@router.websocket("/ws/sensors/{turbine_id}")
async def websocket_sensor_stream(websocket: WebSocket, turbine_id: str):
    """
    Live sensor stream for a single turbine.
    Replays rows from risk_scores.parquet with small gaussian jitter
    to simulate a real-time SCADA data feed. Emits one reading per second.
    """
    await websocket.accept()
    logger.info(f"[WS/sensors] Client connected for {turbine_id}")

    try:
        # Load turbine data once — PyArrow filtered read (fast even on 3 GB)
        if not RISK_SCORES_PATH.exists():
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "risk_scores.parquet not found",
            }))
            await websocket.close()
            return

        try:
            schema    = pq.read_schema(RISK_SCORES_PATH)
            available = set(schema.names)
            want_cols = ["turbine_id"] + [c for c in _STREAM_SENSORS if c in available]
            table = pq.read_table(
                RISK_SCORES_PATH,
                columns=want_cols,
                filters=[("turbine_id", "==", turbine_id)],
            )
            turbine_df = table.to_pandas()
        except Exception as exc:
            await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))
            await websocket.close()
            return

        if turbine_df.empty:
            await websocket.send_text(json.dumps({
                "event":   "error",
                "message": f"Turbine {turbine_id} not found",
            }))
            await websocket.close()
            return

        # Keep only sensor columns + risk columns
        stream_cols = [c for c in _STREAM_SENSORS if c in turbine_df.columns]
        turbine_df = turbine_df[stream_cols].fillna(0)

        # Precompute std per column for jitter
        stds = {col: float(turbine_df[col].std()) * 0.02 for col in stream_cols}

        # Sample row pool — cycle through the last 500 rows endlessly
        pool = turbine_df.tail(500).reset_index(drop=True)
        n = len(pool)
        idx = 0

        await websocket.send_text(json.dumps({
            "event":      "stream_started",
            "turbine_id": turbine_id,
            "channels":   stream_cols,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }))

        while True:
            row = pool.iloc[idx % n]
            payload = {col: round(float(row[col]) + random.gauss(0, stds[col]), 4)
                       for col in stream_cols}
            payload["turbine_id"] = turbine_id
            payload["timestamp"]  = datetime.now(timezone.utc).isoformat()
            payload["event"]      = "sensor_reading"

            await websocket.send_text(json.dumps(payload))
            idx += 1
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info(f"[WS/sensors] {turbine_id} client disconnected")
    except Exception as exc:
        logger.error(f"[WS/sensors] {turbine_id} error: {exc}")
        try:
            await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))
        except Exception:
            pass
