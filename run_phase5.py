"""
Phase 5 — Entry Point
Runs the full Phase 5 agentic pipeline:
  1. Ingests maintenance knowledge base into ChromaDB (once)
  2. Loads Phase 4 risk_scores.parquet
  3. Filters HIGH + CRITICAL turbines
  4. Runs LangGraph pipeline for each alert
  5. Prints fleet summary
  6. (Optional) Starts FastAPI server

Usage:
    # Run pipeline only (no server)
    python run_phase5.py

    # Run pipeline then start API server
    python run_phase5.py --serve

    # Only start the API server
    python run_phase5.py --api-only

    # Force re-ingest ChromaDB
    python run_phase5.py --reingest
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import sys
import io
# Fix Windows cp1252 console encoding — allows arrow/checkmark symbols in logs
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/phase5_run.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RISK_SCORES_PATH   = Path("outputs/risk_scores.parquet")
FLEET_SUMMARY_PATH = Path("outputs/fleet_risk_summary.csv")
REPORTS_DIR        = Path("outputs/reports")

TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def parse_args():
    parser = argparse.ArgumentParser(description="Wind Turbine Phase 5 — Agentic Pipeline")
    parser.add_argument("--serve",      action="store_true", help="Start FastAPI after pipeline run")
    parser.add_argument("--api-only",   action="store_true", help="Skip pipeline, start API only")
    parser.add_argument("--reingest",   action="store_true", help="Force re-ingest ChromaDB knowledge base")
    parser.add_argument("--min-tier",   default="HIGH",      help="Minimum risk tier to process (default: HIGH)")
    parser.add_argument("--max",        type=int, default=5, help="Max turbines to process (default: 5)")
    parser.add_argument("--port",       type=int, default=8000, help="API port (default: 8000)")
    return parser.parse_args()


def step_ingest_kb(reingest: bool = False):
    logger.info("=" * 60)
    logger.info("STEP 1 — Knowledge Base Ingestion (ChromaDB)")
    logger.info("=" * 60)
    from src.rag.ingest import ingest_knowledge_base
    ingest_knowledge_base(reset=reingest)


def step_load_alerts(min_tier: str, max_turbines: int) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 2 — Loading Phase 4 Risk Scores")
    logger.info("=" * 60)

    if not RISK_SCORES_PATH.exists():
        logger.error(f"risk_scores.parquet not found at {RISK_SCORES_PATH}")
        logger.error("Run Phase 4 first: python run_phase4.py")
        sys.exit(1)

    df = pd.read_parquet(RISK_SCORES_PATH)
    logger.info(f"Loaded {len(df):,} rows from risk_scores.parquet")

    # Latest row per turbine
    # Use peak risk window per turbine, not latest timestamp
    # This finds the highest-risk reading for each turbine across the full timeline
    latest = (
        df.sort_values("overall_risk_score", ascending=False)
        .groupby("turbine_id")
        .first()
        .reset_index()
    )
    logger.info(f"Unique turbines: {len(latest)}")

    # Filter by tier
    min_order = TIER_ORDER.get(min_tier.upper(), 2)
    filtered = latest[
        latest["risk_tier"].map(lambda t: TIER_ORDER.get(str(t).upper(), 0)) >= min_order
    ].sort_values("overall_risk_score", ascending=False).head(max_turbines)

    logger.info(f"Turbines at or above {min_tier}: {len(filtered)}")
    for _, row in filtered.iterrows():
        logger.info(
            f"  {row['turbine_id']} | tier={row.get('risk_tier')} | "
            f"score={row.get('overall_risk_score', 0):.3f} | "
            f"fault={row.get('predicted_fault_type') or row.get('predicted_fault') or 'unknown'} | "
            f"rul={row.get('predicted_rul_days') or row.get('rul_days') or 0:.1f}d"
        )

    if filtered.empty:
        logger.warning(f"No turbines at or above {min_tier} tier — nothing to process")
        return []

    # Build alert dicts
    alerts = []
    for _, row in filtered.iterrows():
        sensor_cols = [
            c for c in row.index
            if c.startswith(("bearing_", "generator_", "vibration", "oil_",
                             "rotor_", "wind_", "power_", "residual_", "norm_"))
        ]
        alerts.append({
            "turbine_id":       str(row["turbine_id"]),
            "risk_score":       float(row.get("overall_risk_score", 0)),
            "risk_tier":        str(row.get("risk_tier", "LOW")),
            "anomaly_score":    float(row.get("anomaly_score", 0)),
            "fault_type": str(
                row.get("predicted_fault_type") or
                row.get("predicted_fault") or
                "normal"
            ),
            "fault_confidence": float(row.get("fault_confidence", 0)),
            "rul_days": float(
                row.get("predicted_rul_days") or
                row.get("rul_days") or
                30.0
            ),
            "rul_urgency":      str(row.get("rul_urgency", "low")),
            "sensor_snapshot":  {k: float(row[k]) for k in sensor_cols if pd.notna(row.get(k))},
            "timestamp":        str(row.get("timestamp", "")),
        })

    return alerts


def step_run_pipeline(alerts: list[dict]) -> list:
    logger.info("=" * 60)
    logger.info(f"STEP 3 — Running Agentic Pipeline ({len(alerts)} turbines)")
    logger.info("=" * 60)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    from src.agents.orchestrator_agent import run_fleet_pipeline
    return run_fleet_pipeline(alerts)


def step_print_summary(results: list):
    logger.info("=" * 60)
    logger.info("STEP 4 — Phase 5 Fleet Summary")
    logger.info("=" * 60)

    total = len(results)
    succeeded = sum(1 for r in results if not r.get("pipeline_errors"))
    reports = [r.get("report_output", {}) or {} for r in results]

    print("\n" + "=" * 70)
    print("  WIND TURBINE PREDICTIVE MAINTENANCE — PHASE 5 RESULTS")
    print("=" * 70)
    print(f"  Turbines processed : {total}")
    print(f"  Successful         : {succeeded}")
    print(f"  With errors        : {total - succeeded}")
    print("-" * 70)

    for r in results:
        turbine = r.get("turbine_id", "UNKNOWN")
        tier    = r.get("risk_tier", "N/A")
        fault   = r.get("fault_type", "N/A")
        rul     = r.get("rul_days", 0)
        report  = r.get("report_output", {}) or {}
        path    = report.get("report_path", "N/A")
        errs    = r.get("pipeline_errors", [])

        tier_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(tier, "⚪")
        err_str = f" | ⚠️ {len(errs)} error(s)" if errs else ""

        print(f"  {tier_icon} {turbine:10s} | {tier:8s} | fault={fault:25s} | rul={rul:6.1f}d{err_str}")
        if path != "N/A":
            print(f"             └─ Report: {path}")

    print("=" * 70)
    print(f"  Reports saved to: {REPORTS_DIR.resolve()}")
    print("=" * 70 + "\n")


def step_start_api(port: int):
    logger.info("=" * 60)
    logger.info(f"STEP 5 — Starting FastAPI Server on port {port}")
    logger.info("=" * 60)
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)


def main():
    args = parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("  WIND TURBINE PREDICTIVE MAINTENANCE — PHASE 5")
    logger.info("  Agentic AI Pipeline (LangGraph + Ollama llama3.1:8b)")
    logger.info("=" * 60 + "\n")

    if args.api_only:
        step_start_api(args.port)
        return

    # Step 1 — ChromaDB ingestion
    step_ingest_kb(reingest=args.reingest)

    # Step 2 — Load alerts
    alerts = step_load_alerts(min_tier=args.min_tier, max_turbines=args.max)

    if alerts:
        # Step 3 — Run pipeline
        results = step_run_pipeline(alerts)
        # Step 4 — Summary
        step_print_summary(results)
    else:
        logger.info("No alerts to process — all turbines below threshold")

    # Step 5 — Optional API server
    if args.serve:
        step_start_api(args.port)


if __name__ == "__main__":
    main()