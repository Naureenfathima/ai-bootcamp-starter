from __future__ import annotations
from typing import Optional
"""
pipeline.py — End-to-end RAG pipeline (Stages 1-8).

Ties together ingestion → embedding → retrieval → LLM response.
This is the file you will extend most during Module 3.

STUDENT TODO:
  - Add streaming support so responses appear token-by-token.
  - Add source citations to the response (which chunks were used?).
  - Implement query expansion: rewrite the user's question before retrieval.
  - Add evaluation logging: faithfulness, relevance, completeness scores.
"""

import logging
import time

from app.config import settings
from app.llm import get_llm_client
from app.rag.ingestion import Chunk, chunk_text, ingest_directory
from app.rag.embeddings import embed_chunks, get_embedding_provider
from app.rag.retrieval import InMemoryVectorStore, SearchResult, get_vector_store

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    A complete Retrieval-Augmented Generation pipeline.

    The LLM provider is controlled by LLM_PROVIDER in .env — swap between
    Anthropic, Ollama, Groq, Together AI, or OpenAI without changing any code.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_text("My document content here", source="doc1.txt")
        answer = pipeline.query("What does the document say about X?")
        print(answer.response)
    """

    def __init__(self):
        self._llm          = get_llm_client()
        self._vector_store = get_vector_store()
        self._embed        = get_embedding_provider()
        logger.info("RAGPipeline initialised (vector_store=%s, llm_provider=%s)",
                    settings.vector_store, settings.llm_provider)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    def ingest_text(self, text: str, source: str = "unknown") -> int:
        """
        Chunk and embed a piece of text, adding it to the vector store.

        Returns:
            Number of chunks added.
        """
        chunks = chunk_text(text, source=source)
        embedded = embed_chunks(chunks)
        self._vector_store.add_batch(embedded)
        logger.info("Ingested %d chunks from '%s'", len(chunks), source)
        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        """Ingest all .txt files from a directory."""
        chunks = ingest_directory(directory)
        embedded = embed_chunks(chunks)
        self._vector_store.add_batch(embedded)
        logger.info("Ingested directory '%s' → %d total chunks", directory, len(chunks))
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k:Optional[ int] = None) -> list[SearchResult]:
        """
        Embed a query and retrieve the most relevant chunks.

        Returns:
            List of SearchResult sorted by score descending.
        """
        t0 = time.perf_counter()
        query_vector = self._embed.embed(query)
        results      = self._vector_store.search(query_vector, top_k=top_k)
        latency_ms   = round((time.perf_counter() - t0) * 1000, 1)

        logger.info("Retrieval: query='%s...' → %d results (%s ms)",
                    query[:50], len(results), latency_ms)
        return results

    # ── Context Assembly ──────────────────────────────────────────────────────
    def _assemble_context(self, results: list[SearchResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.

        STUDENT TODO: experiment with different formatting styles.
        """
        if not results:
            return "No relevant context found."

        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"[Source {i}: {result.chunk.source} | Score: {result.score:.2f}]")
            lines.append(result.chunk.text)
            lines.append("")  # blank line between chunks

        return "\n".join(lines)

    # ── Generation ────────────────────────────────────────────────────────────
    def query(self, question: str, top_k:Optional[ int] = None) -> "RAGResponse":
        """
        Run the full RAG pipeline for a user question.

        Args:
            question: The user's question in plain text.
            top_k:    Override the default number of chunks to retrieve.

        Returns:
            RAGResponse with the answer, sources, and timing metadata.
        """
        t_total = time.perf_counter()

        # Stage 4: Retrieve
        results = self.retrieve(question, top_k=top_k)

        # Stage 5: Assemble context
        context = self._assemble_context(results)

        # Stage 6+7: LLM generation
        t_llm = time.perf_counter()
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain enough information "
            "to answer, say so clearly. Do not make up information.\n\n"
            f"CONTEXT:\n{context}"
        )

        llm_response   = self._llm.chat(question, system=system_prompt, max_tokens=1024)
        llm_latency_ms = round((time.perf_counter() - t_llm) * 1000, 1)
        total_ms       = round((time.perf_counter() - t_total) * 1000, 1)

        logger.info("RAG query complete | total=%sms | llm=%sms | tokens=%d",
                    total_ms, llm_latency_ms,
                    llm_response.input_tokens + llm_response.output_tokens)

        return RAGResponse(
            response=llm_response.text,
            sources=[r.chunk.source for r in results],
            scores=[r.score for r in results],
            context_chunks=[r.chunk.text for r in results],
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_ms,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
        )


# ── Response model ────────────────────────────────────────────────────────────
from dataclasses import dataclass, field


@dataclass
class RAGResponse:
    response: str
    sources: list[str]
    scores: list[float]
    context_chunks: list[str]
    llm_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
