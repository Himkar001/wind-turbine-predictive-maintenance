"""
Phase 5 — Parts Agent
Maps fault type + urgency → required spare parts list with
quantities, part numbers, lead times, and ordering priority.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import TurbineAlertState, PartsOutput
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
# ── Static parts catalogue (fallback + LLM grounding) ────────────────────────
PARTS_CATALOGUE: Dict[str, List[Dict[str, Any]]] = {
    "bearing_failure": [
        {"part_name": "Main Shaft Bearing",       "part_number": "BRG-MSB-001", "qty": 1, "lead_days": 14, "unit_cost_usd": 8500},
        {"part_name": "Bearing Seals Kit",         "part_number": "SEAL-BRG-KIT", "qty": 1, "lead_days": 3,  "unit_cost_usd": 180},
        {"part_name": "ISO VG 320 Gear Oil (20L)", "part_number": "OIL-VG320-20", "qty": 2, "lead_days": 1,  "unit_cost_usd": 95},
        {"part_name": "Fastener Set",              "part_number": "FAST-GEN-001", "qty": 1, "lead_days": 1,  "unit_cost_usd": 45},
    ],
    "blade_imbalance": [
        {"part_name": "Leading Edge Tape (10m)",   "part_number": "BLD-LET-10M",  "qty": 3, "lead_days": 5,  "unit_cost_usd": 220},
        {"part_name": "Pitch Encoder",             "part_number": "ENC-PCH-003",  "qty": 1, "lead_days": 7,  "unit_cost_usd": 650},
        {"part_name": "Balance Weights Set",       "part_number": "BLD-BWS-001",  "qty": 1, "lead_days": 3,  "unit_cost_usd": 120},
        {"part_name": "Composite Repair Kit",      "part_number": "BLD-CRK-001",  "qty": 1, "lead_days": 7,  "unit_cost_usd": 890},
    ],
    "electrical_fault": [
        {"part_name": "IGBT Module",               "part_number": "IGBT-CNV-001", "qty": 2, "lead_days": 10, "unit_cost_usd": 1200},
        {"part_name": "Converter Fuse Set",        "part_number": "FUSE-CNV-KIT", "qty": 1, "lead_days": 2,  "unit_cost_usd": 85},
        {"part_name": "Slip Ring Brush Set",       "part_number": "SRB-GEN-001",  "qty": 1, "lead_days": 5,  "unit_cost_usd": 340},
        {"part_name": "Insulation Tape Roll",      "part_number": "INS-TAPE-001", "qty": 3, "lead_days": 1,  "unit_cost_usd": 25},
    ],
    "gearbox_fault": [
        {"part_name": "Gearbox Oil Filter",        "part_number": "FILT-GBX-001", "qty": 2, "lead_days": 2,  "unit_cost_usd": 95},
        {"part_name": "Magnetic Chip Detector",    "part_number": "DET-MAG-001",  "qty": 1, "lead_days": 3,  "unit_cost_usd": 180},
        {"part_name": "ISO VG 320 Gear Oil (20L)", "part_number": "OIL-VG320-20", "qty": 4, "lead_days": 1,  "unit_cost_usd": 95},
        {"part_name": "Gearbox Seal Kit",          "part_number": "SEAL-GBX-KIT", "qty": 1, "lead_days": 5,  "unit_cost_usd": 420},
    ],
    "generator_overheating": [
        {"part_name": "Nacelle Air Filter",        "part_number": "FILT-AIR-NCL", "qty": 2, "lead_days": 2,  "unit_cost_usd": 65},
        {"part_name": "Cooling Fan Belt",          "part_number": "BELT-FAN-001", "qty": 1, "lead_days": 3,  "unit_cost_usd": 85},
        {"part_name": "Cooling Fan Motor",         "part_number": "MOT-FAN-001",  "qty": 1, "lead_days": 7,  "unit_cost_usd": 450},
        {"part_name": "Temperature Sensor PT100",  "part_number": "SEN-TMP-PT100","qty": 2, "lead_days": 3,  "unit_cost_usd": 75},
    ],
    "oil_leak": [
        {"part_name": "Main Shaft Seal",           "part_number": "SEAL-MSH-001", "qty": 1, "lead_days": 5,  "unit_cost_usd": 280},
        {"part_name": "Hydraulic Hose (2m)",       "part_number": "HYD-HOSE-2M",  "qty": 2, "lead_days": 2,  "unit_cost_usd": 65},
        {"part_name": "O-Ring Kit",                "part_number": "ORING-KIT-001","qty": 1, "lead_days": 1,  "unit_cost_usd": 35},
        {"part_name": "Oil Absorbent Kit",         "part_number": "ABS-KIT-001",  "qty": 2, "lead_days": 1,  "unit_cost_usd": 28},
    ],
    "maintenance": [
        {"part_name": "Nacelle Air Filter",        "part_number": "FILT-AIR-NCL", "qty": 1, "lead_days": 2,  "unit_cost_usd": 65},
        {"part_name": "Gearbox Oil Filter",        "part_number": "FILT-GBX-001", "qty": 1, "lead_days": 2,  "unit_cost_usd": 95},
        {"part_name": "Grease Cartridge Set",      "part_number": "GRS-CART-SET", "qty": 2, "lead_days": 1,  "unit_cost_usd": 45},
        {"part_name": "General Fastener Kit",      "part_number": "FAST-GEN-001", "qty": 1, "lead_days": 1,  "unit_cost_usd": 45},
    ],
    "normal": [
        {"part_name": "Consumables Kit (monitor)", "part_number": "CONS-MON-001", "qty": 1, "lead_days": 7,  "unit_cost_usd": 50},
    ],
}

_SYSTEM = """You are a wind turbine spare parts specialist. Based on the fault details and
scheduled maintenance, determine if the standard parts list needs adjustment and provide
an ordering recommendation.

Respond in this EXACT format — no extra text:
ORDER_URGENCY: <immediate|standard|monitor>
SUPPLIER_NOTES: <any special sourcing notes in one sentence>
ADDITIONAL_PARTS: <any parts not in the standard list, or 'none'>
COST_ESTIMATE_USD: <estimated total cost as integer, or 'unknown'>"""


def _build_prompt(state: TurbineAlertState, catalogue_parts: list) -> str:
    schedule = state.get("schedule_output", {})
    parts_text = "\n".join(
        f"  - {p['part_name']} (P/N: {p['part_number']}, qty={p['qty']}, "
        f"lead={p['lead_days']}d, ~${p['unit_cost_usd']})"
        for p in catalogue_parts
    )
    return f"""TURBINE FAULT DETAILS:
  Turbine: {state['turbine_id']}
  Fault: {state.get('fault_type', 'unknown')}
  Risk Tier: {state.get('risk_tier', 'UNKNOWN')}
  RUL: {state.get('rul_days', 0):.1f} days ({state.get('rul_urgency', 'unknown')})
  Maintenance Type: {schedule.get('maintenance_type', 'corrective')}
  Deadline: {schedule.get('deadline_days', 14)} days
  Recommended Action: {schedule.get('recommended_action', 'N/A')}

STANDARD PARTS FROM CATALOGUE:
{parts_text}

Review the standard parts list for this fault and provide ordering guidance."""


def _calculate_cost(parts: list) -> float:
    return sum(p.get("unit_cost_usd", 0) * p.get("qty", 1) for p in parts)


def parts_agent(state: TurbineAlertState) -> TurbineAlertState:
    """LangGraph node — determines required spare parts and ordering urgency."""
    turbine_id = state.get("turbine_id", "UNKNOWN")
    fault_type = state.get("fault_type", "normal")
    logger.info(f"[PARTS] Planning parts for {turbine_id} | fault={fault_type}")

    errors = list(state.get("pipeline_errors", []))

    # Fetch from catalogue (fallback to maintenance kit if fault not found)
    catalogue_parts = PARTS_CATALOGUE.get(fault_type, PARTS_CATALOGUE["maintenance"])
    base_cost = _calculate_cost(catalogue_parts)

    try:
        prompt = _build_prompt(state, catalogue_parts)
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        response = _llm.invoke(messages)

        # Parse LLM response
        text = response.content
        order_urgency = "standard"
        supplier_notes = ""
        estimated_cost = base_cost

        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("ORDER_URGENCY:"):
                order_urgency = line.replace("ORDER_URGENCY:", "").strip().lower()
            elif line.startswith("SUPPLIER_NOTES:"):
                supplier_notes = line.replace("SUPPLIER_NOTES:", "").strip()
            elif line.startswith("COST_ESTIMATE_USD:"):
                try:
                    val = line.replace("COST_ESTIMATE_USD:", "").strip()
                    if val.lower() != "unknown":
                        estimated_cost = float(val.replace(",", ""))
                except ValueError:
                    pass

        # Override urgency based on risk tier if LLM is conservative
        risk_tier = state.get("risk_tier", "MEDIUM")
        if risk_tier == "CRITICAL" and order_urgency != "immediate":
            order_urgency = "immediate"
            logger.info(f"[PARTS] Upgraded order urgency to immediate (CRITICAL tier)")

        parts_output = PartsOutput(
            required_parts=catalogue_parts,
            order_urgency=order_urgency,
            estimated_cost_usd=round(estimated_cost, 2),
            supplier_notes=supplier_notes or "Order through standard procurement channel.",
            raw_llm_response=text,
        )

        logger.info(
            f"[PARTS] {turbine_id} → {len(catalogue_parts)} parts | "
            f"urgency={order_urgency} | est. cost=${estimated_cost:,.0f}"
        )

    except Exception as exc:
        logger.error(f"[PARTS] {turbine_id} failed: {exc}")
        errors.append(f"parts_agent: {exc}")

        # Urgency fallback based on risk tier
        urgency_map = {"CRITICAL": "immediate", "HIGH": "immediate", "MEDIUM": "standard", "LOW": "monitor"}
        parts_output = PartsOutput(
            required_parts=catalogue_parts,
            order_urgency=urgency_map.get(state.get("risk_tier", "MEDIUM"), "standard"),
            estimated_cost_usd=round(base_cost, 2),
            supplier_notes="Auto-generated from catalogue — verify with procurement.",
            raw_llm_response="",
        )

    return {**state, "parts_output": parts_output, "pipeline_errors": errors}