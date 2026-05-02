"""
retrieval.py — Vector storage and similarity search (RAG Stages 3 & 4).

Provides two implementations:
  - InMemoryVectorStore  : for development / testing (no database needed)
  - PgVectorStore        : for production (requires PostgreSQL + pgvector)

STUDENT TODO:
  - Swap InMemoryVectorStore for PgVectorStore once you have a database running.
  - Add hybrid search: combine vector similarity with keyword (BM25) matching.
  - Experiment with different similarity thresholds and top_k values.
"""

import math
import logging
from dataclasses import dataclass

from app.config import settings
from app.rag.ingestion import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float   # 0.0 (no match) → 1.0 (identical)


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── In-Memory Vector Store ────────────────────────────────────────────────────
class InMemoryVectorStore:
    """
    Simple in-memory vector store for development and testing.
    Fast to set up — no database required.
    Loses data when the server restarts.
    """

    def __init__(self):
        self._store: list[tuple[Chunk, list[float]]] = []

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        self._store.append((chunk, embedding))

    def add_batch(self, items: list[tuple[Chunk, list[float]]]) -> None:
        self._store.extend(items)
        logger.info("InMemoryVectorStore now holds %d chunks", len(self._store))

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most similar chunks to query_embedding.

        Args:
            query_embedding: Vector from embedding the user's question.
            top_k:           Maximum results to return.
            threshold:       Minimum similarity score (0-1). Lower scores are discarded.

        Returns:
            Sorted list of SearchResult (highest score first).
        """
        top_k     = top_k     if top_k     is not None else settings.retrieval_top_k
        threshold = threshold if threshold is not None else settings.similarity_threshold

        results: list[SearchResult] = []
        for chunk, embedding in self._store:
            score = cosine_similarity(query_embedding, embedding)
            if score >= threshold:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda r: r.score, reverse=True)

        if not results:
            logger.warning("No results above similarity threshold %.2f", threshold)

        return results[:top_k]

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ── pgvector Store ────────────────────────────────────────────────────────────
class PgVectorStore:
    """
    Production vector store backed by PostgreSQL + pgvector.

    Setup:
        1. Install pgvector: https://github.com/pgvector/pgvector
        2. Set DATABASE_URL in your .env
        3. Run the migration below to create the table.

    Migration SQL (run once):
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS embeddings (
            id        SERIAL PRIMARY KEY,
            source    TEXT,
            chunk_idx INTEGER,
            content   TEXT,
            metadata  JSONB,
            embedding vector(1536)   -- match your embedding dimensions
        );
        CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);

    STUDENT TODO: implement the methods below using asyncpg or psycopg2.
    """

    def __init__(self):
        # STUDENT TODO: initialise your database connection here
        # Example: self._conn = psycopg2.connect(settings.database_url)
        logger.warning(
            "PgVectorStore is a stub. Implement the methods using asyncpg or psycopg2. "
            "See the docstring for setup instructions."
        )

    def add_batch(self, items: list[tuple[Chunk, list[float]]]) -> None:
        # STUDENT TODO: INSERT rows into the embeddings table
        raise NotImplementedError("Implement PgVectorStore.add_batch")

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        # STUDENT TODO:
        # SELECT content, source, 1 - (embedding <=> %s) AS score
        # FROM embeddings
        # ORDER BY embedding <=> %s
        # LIMIT %s
        raise NotImplementedError("Implement PgVectorStore.search")


# ── Factory ───────────────────────────────────────────────────────────────────
def get_vector_store():
    """
    Return the vector store configured in .env.

    VECTOR_STORE=memory    → InMemoryVectorStore (default for development)
    VECTOR_STORE=pgvector  → PgVectorStore (production)
    """
    store = settings.vector_store.lower()
    if store == "memory":
        return InMemoryVectorStore()
    elif store == "pgvector":
        return PgVectorStore()
    else:
        raise ValueError(f"Unknown vector store: {store!r}. "
                         "Set VECTOR_STORE=memory or VECTOR_STORE=pgvector in .env")
