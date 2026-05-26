"""
Phase 5 — Shared LangGraph State Schema
All agents read from and write to this TypedDict.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List


class TurbineAlertState(TypedDict, total=False):
    # ── Input identifiers ────────────────────────────────────────────────────
    turbine_id: str                        # e.g. "WTG-002"
    timestamp: str                         # ISO string of the alert reading

    # ── Risk scorer outputs (Phase 4) ────────────────────────────────────────
    risk_score: float                      # 0.0 – 1.0
    risk_tier: str                         # LOW | MEDIUM | HIGH | CRITICAL
    anomaly_score: float                   # 0.0 – 1.0
    fault_type: str                        # e.g. "bearing_failure"
    fault_confidence: float                # 0.0 – 1.0
    rul_days: float                        # predicted remaining useful life
    rul_urgency: str                       # low | medium | high | critical

    # ── Live sensor snapshot (dict of sensor → value) ────────────────────────
    sensor_snapshot: Dict[str, float]

    # ── Agent outputs (filled progressively as graph runs) ───────────────────
    rca_output: Optional[RCAOutput]
    rag_output: Optional[RAGOutput]
    schedule_output: Optional[ScheduleOutput]
    parts_output: Optional[PartsOutput]
    report_output: Optional[ReportOutput]

    # ── Pipeline metadata ────────────────────────────────────────────────────
    pipeline_errors: List[str]             # any non-fatal errors per agent
    pipeline_version: str                  # "phase5-v1"


# ── Per-agent output schemas ─────────────────────────────────────────────────

class RCAOutput(TypedDict, total=False):
    primary_cause: str                     # one-line root cause
    causal_chain: List[str]               # ordered list: sensor → symptom → fault
    confidence: str                        # LOW | MEDIUM | HIGH
    affected_components: List[str]         # e.g. ["bearing", "gearbox"]
    raw_llm_response: str                  # full LLM text for debugging


class RAGOutput(TypedDict, total=False):
    retrieved_docs: List[Dict[str, str]]   # [{title, content, source}]
    maintenance_procedure: str             # top recommended procedure (text)
    safety_warnings: List[str]             # safety notes from knowledge base
    raw_llm_response: str


class ScheduleOutput(TypedDict, total=False):
    priority_rank: int                     # 1 = most urgent in fleet
    recommended_action: str                # e.g. "Inspect bearing assembly"
    deadline_days: int                     # days from now to complete
    maintenance_type: str                  # preventive | corrective | emergency
    assigned_team: str                     # e.g. "Mechanical Team A"
    estimated_downtime_hours: float
    raw_llm_response: str


class PartsOutput(TypedDict, total=False):
    required_parts: List[Dict[str, Any]]   # [{part_name, part_number, qty, lead_days}]
    order_urgency: str                     # immediate | standard | monitor
    estimated_cost_usd: Optional[float]
    supplier_notes: str
    raw_llm_response: str


class ReportOutput(TypedDict, total=False):
    markdown: str                          # full Markdown report
    json_summary: Dict[str, Any]           # structured summary for API response
    report_path: str                       # saved file path
    generated_at: str                      # ISO timestamp