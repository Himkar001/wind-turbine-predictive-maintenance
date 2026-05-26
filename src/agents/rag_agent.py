"""
Phase 5 — RAG Agent
Retrieves relevant maintenance procedures from ChromaDB,
then uses llama3.1:8b to synthesise a grounded recommendation.
"""

from __future__ import annotations
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import TurbineAlertState, RAGOutput
from src.rag.ingest import query_knowledge_base

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

_SYSTEM = """You are a wind turbine maintenance specialist. You have been given:
1. A turbine fault alert with its root cause analysis
2. Relevant maintenance procedure documents retrieved from the knowledge base

Your job is to synthesise a clear, actionable maintenance recommendation.

Respond in this EXACT format — no extra text:
RECOMMENDED_PROCEDURE: <2-4 sentence summary of what the technician should do>
SAFETY_WARNINGS:
- <warning 1>
- <warning 2>
ESTIMATED_DOWNTIME: <X hours>
PRIORITY_ACTION: <single most important first step>"""


def _build_prompt(state: TurbineAlertState, retrieved_docs: list) -> str:
    rca = state.get("rca_output", {})

    docs_text = ""
    for doc in retrieved_docs:
        docs_text += f"\n--- {doc['title']} (relevance={doc['relevance_score']}) ---\n"
        # Truncate doc content to keep prompt manageable for 8b model
        docs_text += doc["content"][:1200] + "...\n"

    return f"""TURBINE ALERT:
  Turbine: {state['turbine_id']}
  Fault Type: {state.get('fault_type', 'unknown')}
  Risk Tier: {state.get('risk_tier', 'UNKNOWN')}
  RUL: {state.get('rul_days', 0):.1f} days ({state.get('rul_urgency', 'unknown')} urgency)
  RCA Primary Cause: {rca.get('primary_cause', 'Not available')}
  Causal Chain: {' → '.join(rca.get('causal_chain', []))}
  Affected Components: {', '.join(rca.get('affected_components', []))}

RETRIEVED MAINTENANCE DOCUMENTS:
{docs_text}

Based on this fault and these documents, provide your maintenance recommendation."""


def _parse_rag(text: str, docs: list) -> RAGOutput:
    lines = text.strip().splitlines()
    result: RAGOutput = {
        "retrieved_docs": [
            {"title": d["title"], "content": d["content"][:300], "source": d["source"]}
            for d in docs
        ],
        "maintenance_procedure": "",
        "safety_warnings": [],
        "raw_llm_response": text,
    }

    mode = None
    for line in lines:
        line = line.strip()
        if line.startswith("RECOMMENDED_PROCEDURE:"):
            result["maintenance_procedure"] = line.replace("RECOMMENDED_PROCEDURE:", "").strip()
            mode = "procedure"
        elif line.startswith("SAFETY_WARNINGS:"):
            mode = "warnings"
        elif line.startswith("ESTIMATED_DOWNTIME:"):
            mode = None
            result["estimated_downtime"] = line.replace("ESTIMATED_DOWNTIME:", "").strip()
        elif line.startswith("PRIORITY_ACTION:"):
            mode = None
            result["priority_action"] = line.replace("PRIORITY_ACTION:", "").strip()
        elif mode == "procedure" and line and not line.startswith("-"):
            result["maintenance_procedure"] += " " + line
        elif mode == "warnings" and line.startswith("-"):
            result["safety_warnings"].append(line.lstrip("- ").strip())

    if not result["maintenance_procedure"]:
        result["maintenance_procedure"] = text[:500]

    return result


def rag_agent(state: TurbineAlertState) -> TurbineAlertState:
    """LangGraph node — retrieves docs and synthesises maintenance recommendation."""
    turbine_id = state.get("turbine_id", "UNKNOWN")
    fault_type = state.get("fault_type", "unknown")
    logger.info(f"[RAG] Retrieving docs for {turbine_id} | fault={fault_type}")

    errors = list(state.get("pipeline_errors", []))

    try:
        # Build a rich query from fault + RCA context
        rca = state.get("rca_output", {})
        query = (
            f"{fault_type} {rca.get('primary_cause', '')} "
            f"{' '.join(rca.get('affected_components', []))}"
        ).strip()

        # Retrieve from ChromaDB — first try fault-specific, fallback to semantic
        docs = query_knowledge_base(query=query, fault_type=fault_type, n_results=2)
        if not docs:
            docs = query_knowledge_base(query=query, n_results=2)

        logger.info(f"[RAG] Retrieved {len(docs)} documents (top: '{docs[0]['title'] if docs else 'none'}')")

        if not docs:
            raise ValueError("No documents retrieved from knowledge base")

        prompt = _build_prompt(state, docs)
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        response = _llm.invoke(messages)
        rag = _parse_rag(response.content, docs)
        logger.info(f"[RAG] {turbine_id} → procedure synthesised ({len(rag['safety_warnings'])} warnings)")

    except Exception as exc:
        logger.error(f"[RAG] {turbine_id} failed: {exc}")
        errors.append(f"rag_agent: {exc}")
        rag = RAGOutput(
            retrieved_docs=[],
            maintenance_procedure=f"RAG retrieval failed: {exc}",
            safety_warnings=["Unable to retrieve safety warnings — consult manual"],
            raw_llm_response="",
        )

    return {**state, "rag_output": rag, "pipeline_errors": errors}