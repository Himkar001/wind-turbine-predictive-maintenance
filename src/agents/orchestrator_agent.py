"""
Phase 5 — LangGraph Orchestrator
Builds and compiles the agentic pipeline graph.
Routes LOW/MEDIUM alerts to report directly (skip scheduler + parts).
HIGH/CRITICAL alerts go through the full pipeline.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.state import TurbineAlertState
from src.agents.rca_agent import rca_agent
from src.agents.rag_agent import rag_agent
from src.agents.scheduler_agent import scheduler_agent
from src.agents.parts_agent import parts_agent
from src.agents.report_agent import report_agent
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_rag(state: TurbineAlertState) -> Literal["scheduler_agent", "report_agent"]:
    """
    Conditional edge after RAG:
    - HIGH / CRITICAL → full pipeline (scheduler → parts → report)
    - LOW / MEDIUM    → skip to report (no scheduling needed yet)
    """
    risk_tier = state.get("risk_tier", "MEDIUM").upper()
    if risk_tier in ("HIGH", "CRITICAL"):
        logger.info(f"[ORCHESTRATOR] {state.get('turbine_id')} tier={risk_tier} → routing to scheduler")
        return "scheduler_agent"
    else:
        logger.info(f"[ORCHESTRATOR] {state.get('turbine_id')} tier={risk_tier} → skipping to report")
        return "report_agent"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(TurbineAlertState)

    # Register nodes
    graph.add_node("rca_agent",       rca_agent)
    graph.add_node("rag_agent",       rag_agent)
    graph.add_node("scheduler_agent", scheduler_agent)
    graph.add_node("parts_agent",     parts_agent)
    graph.add_node("report_agent",    report_agent)

    # Entry point
    graph.set_entry_point("rca_agent")

    # Edges
    graph.add_edge("rca_agent", "rag_agent")

    # Conditional routing after RAG
    graph.add_conditional_edges(
        "rag_agent",
        route_after_rag,
        {
            "scheduler_agent": "scheduler_agent",
            "report_agent":    "report_agent",
        },
    )

    graph.add_edge("scheduler_agent", "parts_agent")
    graph.add_edge("parts_agent",     "report_agent")
    graph.add_edge("report_agent",    END)

    return graph


# Compile once at module level for reuse
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
        logger.info("[ORCHESTRATOR] LangGraph compiled successfully")
    return _compiled_graph


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run_pipeline(
    turbine_id: str,
    risk_score: float,
    risk_tier: str,
    anomaly_score: float,
    fault_type: str,
    fault_confidence: float,
    rul_days: float,
    rul_urgency: str,
    sensor_snapshot: dict,
    timestamp: str = None,
) -> TurbineAlertState:
    """
    Run the full agentic pipeline for a single turbine alert.
    Returns the final TurbineAlertState with all agent outputs populated.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    initial_state: TurbineAlertState = {
        "turbine_id":       turbine_id,
        "timestamp":        timestamp,
        "risk_score":       risk_score,
        "risk_tier":        risk_tier,
        "anomaly_score":    anomaly_score,
        "fault_type":       fault_type,
        "fault_confidence": fault_confidence,
        "rul_days":         rul_days,
        "rul_urgency":      rul_urgency,
        "sensor_snapshot":  sensor_snapshot,
        "rca_output":       None,
        "rag_output":       None,
        "schedule_output":  None,
        "parts_output":     None,
        "report_output":    None,
        "pipeline_errors":  [],
        "pipeline_version": "phase5-v1",
    }

    logger.info(
        f"\n{'='*60}\n"
        f"[PIPELINE] Starting: {turbine_id}\n"
        f"  Risk: {risk_score:.3f} ({risk_tier}) | Fault: {fault_type} | RUL: {rul_days:.1f}d\n"
        f"{'='*60}"
    )

    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)

    errors = final_state.get("pipeline_errors", [])
    logger.info(
        f"[PIPELINE] Completed: {turbine_id} | "
        f"errors={len(errors)} | "
        f"report={final_state.get('report_output', {}).get('report_path', 'N/A')}"
    )

    return final_state


def run_fleet_pipeline(alerts: list[dict]) -> list[TurbineAlertState]:
    """
    Run the pipeline for a list of turbine alert dicts.
    Each dict must have the same keys as run_pipeline's parameters.
    Returns list of final states in the same order.
    """
    results = []
    logger.info(f"[FLEET] Running pipeline for {len(alerts)} turbine alerts")

    for i, alert in enumerate(alerts):
        logger.info(f"[FLEET] Processing {i+1}/{len(alerts)}: {alert.get('turbine_id')}")
        try:
            result = run_pipeline(**alert)
            results.append(result)
        except Exception as exc:
            logger.error(f"[FLEET] Failed for {alert.get('turbine_id')}: {exc}")
            results.append({
                "turbine_id": alert.get("turbine_id"),
                "pipeline_errors": [str(exc)],
            })

    logger.info(f"[FLEET] Completed {len(results)}/{len(alerts)} alerts")
    return results