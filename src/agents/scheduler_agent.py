"""
Phase 5 — Scheduler Agent
Takes turbine alert + RCA + RAG outputs → produces a prioritised
maintenance schedule with deadlines and team assignment.
"""

from __future__ import annotations
import logging

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import TurbineAlertState, ScheduleOutput
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

_SYSTEM = """You are a wind farm operations manager responsible for scheduling maintenance work.
You receive turbine fault alerts and must create a practical maintenance schedule.

Consider:
- RUL (Remaining Useful Life) determines urgency: <7 days = emergency, <30 = urgent, <90 = planned
- Risk tiers: CRITICAL = same day, HIGH = within 3 days, MEDIUM = within 2 weeks, LOW = next PM window
- Maintenance types: emergency | corrective | preventive
- Teams: Mechanical Team A, Mechanical Team B, Electrical Team, Specialist Contractor

Respond in this EXACT format — no extra text:
PRIORITY_RANK: <1-5, where 1=most urgent>
RECOMMENDED_ACTION: <specific action in one sentence>
DEADLINE_DAYS: <integer number of days from today>
MAINTENANCE_TYPE: <emergency|corrective|preventive>
ASSIGNED_TEAM: <team name>
ESTIMATED_DOWNTIME_HOURS: <number>
SCHEDULING_NOTES: <any important scheduling considerations>"""


def _build_prompt(state: TurbineAlertState) -> str:
    rca = state.get("rca_output", {})
    rag = state.get("rag_output", {})

    return f"""TURBINE ALERT FOR SCHEDULING:
  Turbine ID: {state['turbine_id']}
  Risk Score: {state.get('risk_score', 0):.3f}
  Risk Tier: {state.get('risk_tier', 'UNKNOWN')}
  Fault Type: {state.get('fault_type', 'unknown')}
  Fault Confidence: {state.get('fault_confidence', 0):.2f}
  RUL (days): {state.get('rul_days', 0):.1f}
  RUL Urgency: {state.get('rul_urgency', 'unknown')}
  Anomaly Score: {state.get('anomaly_score', 0):.3f}

ROOT CAUSE ANALYSIS:
  Primary Cause: {rca.get('primary_cause', 'N/A')}
  Affected Components: {', '.join(rca.get('affected_components', []))}
  RCA Confidence: {rca.get('confidence', 'N/A')}

RECOMMENDED MAINTENANCE PROCEDURE:
  {rag.get('maintenance_procedure', 'N/A')}
  Estimated Downtime from KB: {rag.get('estimated_downtime', 'unknown')}

Create a maintenance schedule entry for this turbine."""


def _parse_schedule(text: str) -> ScheduleOutput:
    lines = text.strip().splitlines()
    result: ScheduleOutput = {
        "priority_rank": 3,
        "recommended_action": "",
        "deadline_days": 14,
        "maintenance_type": "corrective",
        "assigned_team": "Mechanical Team A",
        "estimated_downtime_hours": 8.0,
        "raw_llm_response": text,
    }

    for line in lines:
        line = line.strip()
        if line.startswith("PRIORITY_RANK:"):
            try:
                result["priority_rank"] = int(line.replace("PRIORITY_RANK:", "").strip())
            except ValueError:
                pass
        elif line.startswith("RECOMMENDED_ACTION:"):
            result["recommended_action"] = line.replace("RECOMMENDED_ACTION:", "").strip()
        elif line.startswith("DEADLINE_DAYS:"):
            try:
                result["deadline_days"] = int(line.replace("DEADLINE_DAYS:", "").strip())
            except ValueError:
                pass
        elif line.startswith("MAINTENANCE_TYPE:"):
            result["maintenance_type"] = line.replace("MAINTENANCE_TYPE:", "").strip().lower()
        elif line.startswith("ASSIGNED_TEAM:"):
            result["assigned_team"] = line.replace("ASSIGNED_TEAM:", "").strip()
        elif line.startswith("ESTIMATED_DOWNTIME_HOURS:"):
            try:
                result["estimated_downtime_hours"] = float(
                    line.replace("ESTIMATED_DOWNTIME_HOURS:", "").strip()
                )
            except ValueError:
                pass
        elif line.startswith("SCHEDULING_NOTES:"):
            result["scheduling_notes"] = line.replace("SCHEDULING_NOTES:", "").strip()

    if not result["recommended_action"]:
        result["recommended_action"] = f"Inspect and repair {state_fault_type_fallback(text)}"

    return result


def state_fault_type_fallback(text: str) -> str:
    """Extract fault reference from raw text as fallback."""
    for keyword in ["bearing", "gearbox", "blade", "generator", "electrical", "oil"]:
        if keyword in text.lower():
            return keyword
    return "affected component"


def scheduler_agent(state: TurbineAlertState) -> TurbineAlertState:
    """LangGraph node — produces a maintenance schedule entry."""
    turbine_id = state.get("turbine_id", "UNKNOWN")
    logger.info(
        f"[SCHEDULER] Scheduling {turbine_id} | tier={state.get('risk_tier')} | "
        f"rul={state.get('rul_days', 0):.1f}d"
    )

    errors = list(state.get("pipeline_errors", []))

    try:
        prompt = _build_prompt(state)
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        response = _llm.invoke(messages)
        schedule = _parse_schedule(response.content)
        logger.info(
            f"[SCHEDULER] {turbine_id} → {schedule['maintenance_type'].upper()} | "
            f"deadline={schedule['deadline_days']}d | team={schedule['assigned_team']}"
        )

    except Exception as exc:
        logger.error(f"[SCHEDULER] {turbine_id} failed: {exc}")
        errors.append(f"scheduler_agent: {exc}")

        # Risk-based fallback schedule
        risk_tier = state.get("risk_tier", "MEDIUM")
        deadline_map = {"CRITICAL": 1, "HIGH": 3, "MEDIUM": 14, "LOW": 30}
        schedule = ScheduleOutput(
            priority_rank=1 if risk_tier == "CRITICAL" else 2,
            recommended_action=f"Inspect {state.get('fault_type', 'turbine')} — auto-scheduled",
            deadline_days=deadline_map.get(risk_tier, 14),
            maintenance_type="corrective",
            assigned_team="Mechanical Team A",
            estimated_downtime_hours=8.0,
            raw_llm_response="",
        )

    return {**state, "schedule_output": schedule, "pipeline_errors": errors}