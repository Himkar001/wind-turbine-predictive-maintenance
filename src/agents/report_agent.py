"""
Phase 5 — Report Agent
Synthesises all upstream agent outputs into:
  1. A structured Markdown maintenance report (saved to disk)
  2. A JSON summary dict (returned via FastAPI)
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import TurbineAlertState, ReportOutput
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

REPORTS_DIR = Path("outputs/reports")

_SYSTEM = """You are a technical writer producing a professional wind turbine maintenance report.
Write a concise executive summary (3-5 sentences) for the maintenance manager.
Cover: what the fault is, how serious it is, what action is needed, and when.
Be clear, factual, and avoid jargon. Do not use bullet points in this summary."""


def _build_exec_summary_prompt(state: TurbineAlertState) -> str:
    rca = state.get("rca_output", {})
    schedule = state.get("schedule_output", {})
    rag = state.get("rag_output", {})
    return f"""Turbine {state['turbine_id']} has triggered a {state.get('risk_tier', 'UNKNOWN')} risk alert.

Fault: {state.get('fault_type', 'unknown')} (confidence: {state.get('fault_confidence', 0):.0%})
Root Cause: {rca.get('primary_cause', 'N/A')}
Remaining Useful Life: {state.get('rul_days', 0):.1f} days
Recommended Action: {schedule.get('recommended_action', 'N/A')}
Deadline: {schedule.get('deadline_days', 14)} days
Maintenance Procedure: {rag.get('maintenance_procedure', 'N/A')[:300]}

Write a 3-5 sentence executive summary for the maintenance manager."""


def _build_markdown(state: TurbineAlertState, exec_summary: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rca = state.get("rca_output", {})
    rag = state.get("rag_output", {})
    schedule = state.get("schedule_output", {})
    parts = state.get("parts_output", {})
    errors = state.get("pipeline_errors", [])

    # Risk tier → emoji
    tier_icon = {
        "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
    }.get(state.get("risk_tier", "UNKNOWN"), "⚪")

    # Parts table
    parts_table = "| Part Name | Part Number | Qty | Lead Time | Est. Cost |\n"
    parts_table += "|-----------|-------------|-----|-----------|----------|\n"
    for p in parts.get("required_parts", []):
        parts_table += (
            f"| {p['part_name']} | {p['part_number']} | {p['qty']} | "
            f"{p['lead_days']}d | ${p['unit_cost_usd']:,} |\n"
        )

    # Causal chain
    chain = rca.get("causal_chain", [])
    chain_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(chain)) if chain else "_Not available_"

    # Safety warnings
    warnings = rag.get("safety_warnings", [])
    warnings_md = "\n".join(f"- ⚠️ {w}" for w in warnings) if warnings else "_None identified_"

    # Pipeline errors
    errors_md = "\n".join(f"- `{e}`" for e in errors) if errors else "_None_"

    report = f"""# Wind Turbine Maintenance Report
**Generated:** {now}
**Pipeline Version:** {state.get('pipeline_version', 'phase5-v1')}

---

## {tier_icon} Alert Summary — {state['turbine_id']}

| Field | Value |
|-------|-------|
| **Turbine ID** | {state['turbine_id']} |
| **Timestamp** | {state.get('timestamp', 'N/A')} |
| **Risk Score** | {state.get('risk_score', 0):.3f} |
| **Risk Tier** | {tier_icon} **{state.get('risk_tier', 'UNKNOWN')}** |
| **Anomaly Score** | {state.get('anomaly_score', 0):.3f} |
| **Fault Type** | `{state.get('fault_type', 'unknown')}` |
| **Fault Confidence** | {state.get('fault_confidence', 0):.1%} |
| **RUL (days)** | {state.get('rul_days', 0):.1f} days |
| **RUL Urgency** | {state.get('rul_urgency', 'unknown').upper()} |

---

## Executive Summary

{exec_summary}

---

## Root Cause Analysis

**Primary Cause:** {rca.get('primary_cause', '_Not available_')}

**Confidence:** {rca.get('confidence', 'N/A')}

**Affected Components:** {', '.join(rca.get('affected_components', [])) or '_Not identified_'}

**Causal Chain:**

{chain_md}

---

## Maintenance Recommendation

{rag.get('maintenance_procedure', '_Not available_')}

**Safety Warnings:**

{warnings_md}

**Estimated Downtime:** {rag.get('estimated_downtime', 'Unknown')}

---

## Maintenance Schedule

| Field | Value |
|-------|-------|
| **Priority Rank** | #{schedule.get('priority_rank', 'N/A')} |
| **Action** | {schedule.get('recommended_action', 'N/A')} |
| **Type** | {schedule.get('maintenance_type', 'corrective').upper()} |
| **Deadline** | {schedule.get('deadline_days', 'N/A')} days from today |
| **Assigned Team** | {schedule.get('assigned_team', 'N/A')} |
| **Est. Downtime** | {schedule.get('estimated_downtime_hours', 'N/A')} hours |

{schedule.get('scheduling_notes', '')}

---

## Spare Parts Order

**Order Urgency:** `{parts.get('order_urgency', 'standard').upper()}`

**Estimated Total Cost:** ${parts.get('estimated_cost_usd', 0):,.2f}

{parts_table}

**Supplier Notes:** {parts.get('supplier_notes', 'N/A')}

---

## Pipeline Diagnostics

**Errors / Warnings:** {errors_md}

---
_Report generated by Wind Turbine Predictive Maintenance System — Phase 5 Agentic Pipeline_
"""
    return report


def _build_json_summary(state: TurbineAlertState, exec_summary: str, report_path: str) -> dict:
    rca = state.get("rca_output", {})
    rag = state.get("rag_output", {})
    schedule = state.get("schedule_output", {})
    parts = state.get("parts_output", {})

    return {
        "turbine_id": state.get("turbine_id"),
        "timestamp": state.get("timestamp"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_path": report_path,
        "alert": {
            "risk_score": state.get("risk_score"),
            "risk_tier": state.get("risk_tier"),
            "anomaly_score": state.get("anomaly_score"),
            "fault_type": state.get("fault_type"),
            "fault_confidence": state.get("fault_confidence"),
            "rul_days": state.get("rul_days"),
            "rul_urgency": state.get("rul_urgency"),
        },
        "rca": {
            "primary_cause": rca.get("primary_cause"),
            "causal_chain": rca.get("causal_chain", []),
            "affected_components": rca.get("affected_components", []),
            "confidence": rca.get("confidence"),
        },
        "recommendation": {
            "procedure": rag.get("maintenance_procedure"),
            "safety_warnings": rag.get("safety_warnings", []),
            "estimated_downtime": rag.get("estimated_downtime"),
        },
        "schedule": {
            "priority_rank": schedule.get("priority_rank"),
            "recommended_action": schedule.get("recommended_action"),
            "maintenance_type": schedule.get("maintenance_type"),
            "deadline_days": schedule.get("deadline_days"),
            "assigned_team": schedule.get("assigned_team"),
            "estimated_downtime_hours": schedule.get("estimated_downtime_hours"),
        },
        "parts": {
            "order_urgency": parts.get("order_urgency"),
            "estimated_cost_usd": parts.get("estimated_cost_usd"),
            "required_parts": parts.get("required_parts", []),
            "supplier_notes": parts.get("supplier_notes"),
        },
        "executive_summary": exec_summary,
        "pipeline_errors": state.get("pipeline_errors", []),
    }


def report_agent(state: TurbineAlertState) -> TurbineAlertState:
    """LangGraph node — generates and saves the final maintenance report."""
    turbine_id = state.get("turbine_id", "UNKNOWN")
    logger.info(f"[REPORT] Generating report for {turbine_id}")

    errors = list(state.get("pipeline_errors", []))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Generate executive summary via LLM ────────────────────────────────────
    try:
        prompt = _build_exec_summary_prompt(state)
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        response = _llm.invoke(messages)
        exec_summary = response.content.strip()
    except Exception as exc:
        logger.error(f"[REPORT] LLM exec summary failed: {exc}")
        errors.append(f"report_agent_llm: {exc}")
        rca = state.get("rca_output", {})
        exec_summary = (
            f"Turbine {turbine_id} has a {state.get('risk_tier', 'UNKNOWN')} risk alert. "
            f"Fault: {state.get('fault_type', 'unknown')}. "
            f"Root cause: {rca.get('primary_cause', 'See RCA section')}. "
            f"RUL: {state.get('rul_days', 0):.1f} days. "
            f"Immediate inspection recommended."
        )

    # ── Build report ──────────────────────────────────────────────────────────
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{turbine_id}_{state.get('risk_tier', 'UNKNOWN')}_{timestamp_str}.md"
    report_path = str(REPORTS_DIR / filename)

    markdown = _build_markdown(state, exec_summary)
    json_summary = _build_json_summary(state, exec_summary, report_path)

    # ── Save Markdown ─────────────────────────────────────────────────────────
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info(f"[REPORT] Saved → {report_path}")
    except Exception as exc:
        logger.error(f"[REPORT] Failed to save markdown: {exc}")
        errors.append(f"report_agent_save: {exc}")

    # ── Save JSON sidecar ─────────────────────────────────────────────────────
    json_path = report_path.replace(".md", ".json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2, default=str)
        logger.info(f"[REPORT] JSON saved → {json_path}")
    except Exception as exc:
        logger.error(f"[REPORT] Failed to save JSON: {exc}")
        errors.append(f"report_agent_json: {exc}")

    report_output = ReportOutput(
        markdown=markdown,
        json_summary=json_summary,
        report_path=report_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(f"[REPORT] ✓ {turbine_id} report complete | tier={state.get('risk_tier')}")
    return {**state, "report_output": report_output, "pipeline_errors": errors}