"""
Phase 6 — Fleet Router
=======================
Endpoints:
  GET /fleet/status   → full turbine list with latest risk data
  GET /fleet/summary  → dashboard KPI cards (counts, averages, tier breakdown)
  GET /reports/list   → recent reports index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fleet", tags=["fleet"])
reports_router = APIRouter(prefix="/reports", tags=["reports"])

# ── Paths ─────────────────────────────────────────────────────────────────────
RISK_SCORES_PATH   = Path("outputs/risk_scores.parquet")
REPORTS_DIR        = Path("outputs/reports")

TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
TIER_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#eab308",
    "LOW":      "#22c55e",
}

# Path to the pre-computed fleet summary CSV
FLEET_SUMMARY_PATH = Path("outputs/fleet_risk_summary.csv")

# ── Global Cache ──────────────────────────────────────────────────────────────
_LATEST_SNAPSHOT_CACHE: Optional[pd.DataFrame] = None


def _get_latest_snapshot() -> pd.DataFrame:
    """
    Build per-turbine fleet snapshot:
    - Risk score / tier  → from the chronologically latest row (most current).
    - Fault type         → from the most recent row where fault != 'normal',
                           so we always display an informative fault label.
    - RUL               → from the chronologically latest row.
    Cached in memory for instant repeat access.
    """
    global _LATEST_SNAPSHOT_CACHE
    if _LATEST_SNAPSHOT_CACHE is not None:
        return _LATEST_SNAPSHOT_CACHE

    if not RISK_SCORES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="risk_scores.parquet not found — run Phase 4 first"
        )

    logger.info("[Fleet] Building fleet snapshot (PyArrow optimised)...")
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(RISK_SCORES_PATH)
        want_cols = [
            "turbine_id", "timestamp", "risk_tier", "overall_risk_score",
            "predicted_fault_type", "rul_days", "rul_urgency"
        ]
        safe_cols = [c for c in want_cols if c in schema.names]

        df = pq.read_table(RISK_SCORES_PATH, columns=safe_cols).to_pandas()

        # ── Peak-risk row per turbine ──────────────────────────────────────────
        # Use idxmax() for O(N) performance (~10x faster than sort_values on 7M rows)
        idx = df.groupby("turbine_id", sort=False)["overall_risk_score"].idxmax()
        latest = df.loc[idx].reset_index(drop=True)


        # ── Override 'normal' fault labels with dominant fault from CSV ────────
        # If the peak-risk row still says 'normal', use the fleet CSV's
        # dominant_fault which was computed by Phase 4 across all anomaly windows.
        fault_col = "predicted_fault_type"
        if FLEET_SUMMARY_PATH.exists() and fault_col in latest.columns:
            try:
                csv_df = pd.read_csv(FLEET_SUMMARY_PATH)
                if "dominant_fault" in csv_df.columns:
                    fault_map = csv_df.set_index("turbine_id")["dominant_fault"].to_dict()
                    mask = latest[fault_col].str.lower().str.strip().isin(["normal", "nan", ""])
                    latest.loc[mask, fault_col] = latest.loc[mask, "turbine_id"].map(fault_map)
            except Exception:
                pass  # CSV fallback failed — keep as-is

        _LATEST_SNAPSHOT_CACHE = latest
        logger.info(f"[Fleet] Snapshot ready — {len(latest)} turbines.")
        return _LATEST_SNAPSHOT_CACHE
    except Exception as exc:
        logger.error(f"[Fleet] Failed to build snapshot: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to read parquet: {exc}")


# ── /fleet/status ─────────────────────────────────────────────────────────────

@router.get("/status")
async def fleet_status():
    """
    Returns the exact latest risk snapshot for every turbine in the fleet.
    Cached in memory for instant response.
    """
    latest = _get_latest_snapshot()

    want_cols = [
        "turbine_id", "risk_tier", "overall_risk_score",
        "predicted_fault_type", "predicted_fault", "rul_days", "rul_urgency", "timestamp"
    ]
    cols = [c for c in want_cols if c in latest.columns]
    records = latest[cols].fillna(0).to_dict(orient="records")

    for r in records:
        r["risk_score"] = r.pop("overall_risk_score", 0)
        # Handle both old and new column names
        r["fault_type"] = r.pop("predicted_fault_type", r.pop("predicted_fault", "normal"))
        r["tier_color"] = TIER_COLORS.get(str(r.get("risk_tier", "LOW")).upper(), "#6b7280")

    return {
        "source":        "risk_scores.parquet (cached)",
        "turbine_count": len(records),
        "data":          records,
    }


# ── /fleet/summary ────────────────────────────────────────────────────────────

@router.get("/summary")
async def fleet_summary():
    """
    Dashboard KPI cards based on the true real-time snapshot:
    - total turbines
    - tier breakdown (CRITICAL / HIGH / MEDIUM / LOW counts)
    - average risk score
    - turbine most at risk
    - fleet-wide dominant fault
    """
    df = _get_latest_snapshot()
    score_col = "overall_risk_score" if "overall_risk_score" in df.columns else None

    total = len(df)

    # Tier breakdown
    tier_col = "risk_tier" if "risk_tier" in df.columns else None
    tier_breakdown = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    if tier_col:
        for tier, count in df[tier_col].str.upper().value_counts().items():
            if tier in tier_breakdown:
                tier_breakdown[tier] = int(count)

    # Average risk
    avg_risk = float(df[score_col].mean()) if score_col and score_col in df.columns else 0.0

    # Most at-risk turbine
    most_at_risk = None
    if score_col and score_col in df.columns:
        idx = df[score_col].idxmax()
        row = df.loc[idx]
        most_at_risk = {
            "turbine_id": str(row.get("turbine_id", "?")),
            "risk_score":  round(float(row.get(score_col, 0)), 4),
            "risk_tier":   str(row.get("risk_tier", "?")),
        }

    # Dominant fleet fault
    fault_col = "predicted_fault_type" if "predicted_fault_type" in df.columns else ("predicted_fault" if "predicted_fault" in df.columns else None)
    dominant_fault = "unknown"
    if fault_col:
        excl = {"none", "normal", "maintenance", "nan"}
        faults = df[fault_col].dropna().astype(str).str.lower()
        faults = faults[~faults.isin(excl)]
        if not faults.empty:
            dominant_fault = faults.mode().iloc[0]

    return {
        "total_turbines":  total,
        "tier_breakdown":  tier_breakdown,
        "average_risk":    round(avg_risk, 4),
        "most_at_risk":    most_at_risk,
        "dominant_fault":  dominant_fault,
        "tier_colors":     TIER_COLORS,
    }


# ── /reports/list ─────────────────────────────────────────────────────────────

@reports_router.get("/list")
async def list_reports(
    turbine_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent maintenance reports, optionally filtered by turbine_id."""
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
