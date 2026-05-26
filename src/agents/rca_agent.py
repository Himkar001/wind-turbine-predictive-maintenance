"""
Phase 5 — RCA Agent
Analyses sensor snapshot + Phase 4 outputs to generate a structured
root cause chain using llama3.1:8b via Ollama.
"""

from __future__ import annotations
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import TurbineAlertState, RCAOutput
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM ──────────────────────────────────────────────────────────────────────
_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """You are an expert wind turbine maintenance engineer with 20 years of experience
diagnosing faults in industrial turbines. You reason systematically from sensor data to root causes.

When given turbine sensor readings and ML model outputs, you:
1. Identify the primary root cause in one clear sentence
2. Build a causal chain (what sensor deviated → what component was affected → what fault resulted)
3. List the affected physical components
4. Assign a confidence level: LOW / MEDIUM / HIGH

Always respond in this EXACT format — no extra text:
PRIMARY_CAUSE: <one sentence>
CAUSAL_CHAIN:
- <step 1>
- <step 2>
- <step 3>
AFFECTED_COMPONENTS: <comma-separated list>
CONFIDENCE: <LOW|MEDIUM|HIGH>"""


def _build_prompt(state: TurbineAlertState) -> str:
    sensor = state.get("sensor_snapshot", {})
    sensor_lines = "\n".join(
        f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
        for k, v in sensor.items()
    )
    return f"""Turbine: {state['turbine_id']}
Timestamp: {state.get('timestamp', 'unknown')}

ML MODEL OUTPUTS:
  Risk Score: {state.get('risk_score', 0):.3f} ({state.get('risk_tier', 'UNKNOWN')})
  Anomaly Score: {state.get('anomaly_score', 0):.3f}
  Predicted Fault: {state.get('fault_type', 'unknown')} (confidence={state.get('fault_confidence', 0):.2f})
  Remaining Useful Life: {state.get('rul_days', 0):.1f} days ({state.get('rul_urgency', 'unknown')} urgency)

SENSOR SNAPSHOT (normalised values, 0=healthy baseline):
{sensor_lines}

Perform root cause analysis for this turbine alert."""


def _parse_rca(text: str) -> RCAOutput:
    """Parse structured LLM response into RCAOutput dict."""
    lines = text.strip().splitlines()
    result: RCAOutput = {
        "primary_cause": "",
        "causal_chain": [],
        "confidence": "LOW",
        "affected_components": [],
        "raw_llm_response": text,
    }

    mode = None
    for line in lines:
        line = line.strip()
        if line.startswith("PRIMARY_CAUSE:"):
            result["primary_cause"] = line.replace("PRIMARY_CAUSE:", "").strip()
        elif line.startswith("CAUSAL_CHAIN:"):
            mode = "chain"
        elif line.startswith("AFFECTED_COMPONENTS:"):
            mode = None
            components = line.replace("AFFECTED_COMPONENTS:", "").strip()
            result["affected_components"] = [c.strip() for c in components.split(",")]
        elif line.startswith("CONFIDENCE:"):
            mode = None
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().upper()
        elif mode == "chain" and line.startswith("-"):
            result["causal_chain"].append(line.lstrip("- ").strip())

    # Fallback if parsing failed
    if not result["primary_cause"]:
        result["primary_cause"] = f"Unstructured fault detected: {text[:200]}"

    return result


def rca_agent(state: TurbineAlertState) -> TurbineAlertState:
    """LangGraph node — performs root cause analysis."""
    turbine_id = state.get("turbine_id", "UNKNOWN")
    logger.info(f"[RCA] Analysing {turbine_id} | fault={state.get('fault_type')} | tier={state.get('risk_tier')}")

    errors = list(state.get("pipeline_errors", []))

    try:
        prompt = _build_prompt(state)
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        response = _llm.invoke(messages)
        rca = _parse_rca(response.content)
        logger.info(f"[RCA] {turbine_id} → {rca['primary_cause'][:80]} (conf={rca['confidence']})")

    except Exception as exc:
        logger.error(f"[RCA] {turbine_id} failed: {exc}")
        errors.append(f"rca_agent: {exc}")
        rca = RCAOutput(
            primary_cause=f"RCA failed: {exc}",
            causal_chain=[],
            confidence="LOW",
            affected_components=[],
            raw_llm_response="",
        )

    return {**state, "rca_output": rca, "pipeline_errors": errors}