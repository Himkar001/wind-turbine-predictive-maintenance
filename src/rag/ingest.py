"""
Phase 5 — ChromaDB Ingestion Script
Run once to embed the maintenance knowledge base into ChromaDB.

Usage:
    python -m src.rag.ingest
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.rag.knowledge_base import MAINTENANCE_DOCS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "turbine_maintenance_kb"

# Using sentence-transformers for local embedding (no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def get_collection(client: chromadb.PersistentClient, reset: bool = False):
    """Get or create the ChromaDB collection."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest_knowledge_base(reset: bool = False) -> None:
    """Embed all maintenance docs into ChromaDB."""
    logger.info("=" * 60)
    logger.info("ChromaDB Knowledge Base Ingestion — Phase 5")
    logger.info("=" * 60)

    client = get_chroma_client()
    collection = get_collection(client, reset=reset)

    # Check if already ingested
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        logger.info(f"Collection already has {existing_count} documents. Use reset=True to re-ingest.")
        return

    logger.info(f"Ingesting {len(MAINTENANCE_DOCS)} documents into ChromaDB...")

    ids = []
    documents = []
    metadatas = []

    for doc in MAINTENANCE_DOCS:
        ids.append(doc["id"])
        # Combine title + content for richer embedding
        documents.append(f"{doc['title']}\n\n{doc['content']}")
        metadatas.append({
            "title": doc["title"],
            "fault_type": doc["fault_type"],
            "source": doc["source"],
            "doc_id": doc["id"],
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Successfully ingested {len(ids)} documents")
    logger.info(f"Collection size: {collection.count()} documents")
    logger.info(f"ChromaDB path: {CHROMA_DB_PATH.resolve()}")
    logger.info("=" * 60)
    logger.info("Knowledge base ready for RAG queries")
    logger.info("=" * 60)


def query_knowledge_base(query: str, fault_type: str = None, n_results: int = 3) -> list:
    """
    Query ChromaDB for relevant maintenance docs.
    Optionally filter by fault_type for more precise retrieval.
    """
    client = get_chroma_client()
    collection = get_collection(client)

    if collection.count() == 0:
        logger.warning("Knowledge base is empty — run ingest first.")
        return []

    where_filter = {"fault_type": fault_type} if fault_type else None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Fallback without filter if fault_type not found
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    docs = []
    if results and results["documents"]:
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            docs.append({
                "rank": i + 1,
                "title": meta.get("title", ""),
                "fault_type": meta.get("fault_type", ""),
                "source": meta.get("source", ""),
                "content": doc,
                "relevance_score": round(1 - dist, 4),  # cosine similarity
            })

    return docs


if __name__ == "__main__":
    ingest_knowledge_base(reset=True)