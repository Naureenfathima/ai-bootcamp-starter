"""
anti_rag/cag.py — Cache-Augmented Generation (CAG).

CAG is the simplest "anti-RAG" approach: instead of retrieving a few relevant
chunks (RAG), you load ALL your documents into the context window at once and
let the LLM find the answer directly.

Why CAG instead of RAG?
  - RAG can miss relevant context (retrieval threshold, chunking boundary).
  - CAG has zero retrieval step — the LLM sees every document, so it never misses.
  - CAG is ideal for small-to-medium knowledge bases (< ~100 documents).
  - RAG is necessary for large corpora that exceed Claude's context window.

The "cache" in CAG refers to the KV cache in transformer models:
  - You mark the document block with cache_control so Anthropic caches its KV state.
  - The first query pays full token cost; subsequent queries over the same documents
    reuse the cached state at ~10% of the original cost.
  - Minimum 1024 tokens must be in the cached block for caching to activate.

Head-to-head comparison:

  Approach │ Retrieval step │ Can miss context │ Cost per query │ Max corpus size
  ─────────┼────────────────┼──────────────────┼────────────────┼─────────────────
  RAG      │ Vector search  │ Yes (threshold)  │ Low            │ Unlimited
  KAG      │ Graph traversal│ Yes (missing edge)│ Medium         │ Unlimited
  CAG      │ None           │ Never            │ High (cached:↓)│ Context window

STUDENT TODO:
  - Observe the cache_creation_input_tokens and cache_read_input_tokens in responses.
    First query: cache_creation > 0. Second query on same documents: cache_read > 0.
  - Add document deduplication: don't ingest the same source twice.
  - Add a token budget guard: refuse to ingest if corpus would exceed a safe threshold.
  - Compare CAG cost vs. RAG cost for the same query on the same data.
"""

import logging
from dataclasses import dataclass, field
import anthropic
from app.config import settings

logger = logging.getLogger(__name__)

# Rough estimate: average English prose is ~4 characters per token
_CHARS_PER_TOKEN = 4
# Claude's context window
_CONTEXT_WINDOW_TOKENS = 200_000
# Warn at 80% utilisation
_WARN_THRESHOLD = 0.8


@dataclass
class CAGDocument:
    """A single document stored in the CAG knowledge base."""
    text: str
    source: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class CAGResult:
    """Result returned by the CAG pipeline."""
    answer: str
    documents_used: int
    estimated_context_tokens: int
    context_window_pct: float
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0  # non-zero on first call (cache being built)
    cache_read_input_tokens: int = 0      # non-zero on subsequent calls (cache hit)


class CAGPipeline:
    """
    Cache-Augmented Generation pipeline.

    Unlike RAG (which retrieves 3-5 similar chunks), CAG loads ALL documents
    into the context window for each query. Anthropic's prompt caching then
    stores the KV state of those documents so repeated queries are cheap.

    This is an excellent classroom demo because:
      1. It is conceptually simple — no vector math, no thresholds.
      2. The cache token counters make the caching benefit visible.
      3. Comparing it side-by-side with KAG/RAG shows clear tradeoffs.

    Usage:
        pipeline = CAGPipeline()
        pipeline.ingest("Alice manages the Platform team.", source="org_chart.txt")
        pipeline.ingest("Bob leads the Platform team.", source="org_chart.txt")

        result = pipeline.query("Who manages the team that Bob leads?")
        print(result.answer)
        # "Alice manages the Platform team, which is the team that Bob leads."

        # Second query — documents are cached, cheaper:
        result2 = pipeline.query("What team does Bob lead?")
        print(result2.cache_read_input_tokens)  # > 0 if corpus >= 1024 tokens
    """

    SYSTEM_PROMPT = (
        "You are a precise assistant. Answer questions using ONLY the provided documents. "
        "If the answer is not found in the documents, say so clearly. "
        "Cite the document number (e.g. [Doc 3]) when referencing specific information."
    )

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._documents: list[CAGDocument] = []
        logger.info("CAGPipeline initialised")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    def ingest(self, text: str, source: str = "unknown") -> dict:
        """
        Add a document to the knowledge base.
        No chunking, no embedding — just store the full text.

        The document will be included in the context of every future query.

        Args:
            text:   Full text of the document.
            source: A label for the document (filename, URL, etc.).

        Returns:
            Current knowledge base statistics, including a context window estimate.
        """
        text = text.strip()
        if not text:
            return {"documents": len(self._documents), "warning": "Empty text ignored"}

        doc = CAGDocument(text=text, source=source)
        self._documents.append(doc)

        total_chars = sum(d.char_count for d in self._documents)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN
        pct = estimated_tokens / _CONTEXT_WINDOW_TOKENS

        logger.info(
            "CAG: ingested %d chars from '%s' (total ~%d tokens, %.1f%% of window)",
            doc.char_count, source, estimated_tokens, pct * 100,
        )

        if pct > _WARN_THRESHOLD:
            logger.warning(
                "CAG: corpus is %.1f%% of the context window — switch to RAG for large corpora",
                pct * 100,
            )

        return {
            "documents": len(self._documents),
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "context_window_pct": round(pct * 100, 1),
            "warning": (
                f"Approaching context limit ({pct:.0%} used) — consider RAG"
                if pct > _WARN_THRESHOLD else None
            ),
        }

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(self, question: str) -> CAGResult:
        """
        Answer a question by loading ALL documents into one context window.

        The document block is marked with cache_control so Anthropic caches
        its KV state. After the first call, cache_read_input_tokens will be
        populated instead of cache_creation_input_tokens — this means the
        model processes the documents at a fraction of the normal cost.

        Args:
            question: The user's natural language question.

        Returns:
            CAGResult including answer, token counts, and cache statistics.
        """
        if not self._documents:
            return CAGResult(
                answer="No documents ingested yet. Call /anti-rag/cag/ingest first.",
                documents_used=0,
                estimated_context_tokens=0,
                context_window_pct=0.0,
                input_tokens=0,
                output_tokens=0,
            )

        # Build one big context block from all documents
        context_parts = []
        for i, doc in enumerate(self._documents, 1):
            context_parts.append(f"[Doc {i} — {doc.source}]\n{doc.text}")
        full_context = "\n\n---\n\n".join(context_parts)

        total_chars = sum(d.char_count for d in self._documents)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN
        context_window_pct = round((estimated_tokens / _CONTEXT_WINDOW_TOKENS) * 100, 1)

        logger.info(
            "CAG: querying with %d docs (~%d tokens, %.1f%% of window)",
            len(self._documents), estimated_tokens, context_window_pct,
        )

        # The document block is marked ephemeral so Anthropic caches its KV state.
        # Key teaching point: cache_control goes on the DOCUMENTS, not the question.
        # Caching only activates when the cached block is >= 1024 tokens.
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Documents:\n\n{full_context}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"Question: {question}",
                    },
                ],
            }],
        )

        answer = response.content[0].text
        usage = response.usage

        # cache_* fields are present when prompt caching is active
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        logger.info(
            "CAG: input=%d cache_created=%d cache_read=%d output=%d",
            usage.input_tokens, cache_creation, cache_read, usage.output_tokens,
        )

        return CAGResult(
            answer=answer,
            documents_used=len(self._documents),
            estimated_context_tokens=estimated_tokens,
            context_window_pct=context_window_pct,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Return current knowledge base statistics."""
        total_chars = sum(d.char_count for d in self._documents)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN
        return {
            "documents": len(self._documents),
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "context_window_pct": round((estimated_tokens / _CONTEXT_WINDOW_TOKENS) * 100, 1),
            "sources": sorted({d.source for d in self._documents}),
        }

    def seed_demo_documents(self) -> dict:
        """
        Pre-populate with demo documents for classroom demos.
        No API call — adds documents directly.

        Uses the same org chart domain as KAGPipeline.seed_demo_graph() so
        students can run identical questions through both pipelines and compare
        the approaches side by side.
        """
        demo_docs = [
            ("Alice is the CTO of the company. She directly manages Bob and Carol.",
             "org_chart.txt"),
            ("Bob is the Backend Lead. He leads the Platform Team.",
             "org_chart.txt"),
            ("Carol is the ML Lead. She leads the AI Team.",
             "org_chart.txt"),
            ("The Platform Team owns two services: the RAG Service and the API Gateway.",
             "services.txt"),
            ("The AI Team owns two services: the Agent Service and the Evaluation Harness.",
             "services.txt"),
            ("The RAG Service depends on VectorDB, "
             "which is a PostgreSQL instance with the pgvector extension.",
             "infra.txt"),
            ("The Agent Service uses the RAG Service to retrieve context "
             "before generating responses. It also calls External APIs for real-time data.",
             "infra.txt"),
            ("The API Gateway is the single entry point for all external client requests. "
             "It handles authentication, rate limiting, and request routing.",
             "infra.txt"),
        ]

        for text, source in demo_docs:
            self.ingest(text, source=source)

        logger.info("CAGPipeline: seeded %d demo documents", len(demo_docs))
        return self.stats()

    def clear(self) -> None:
        """Clear all ingested documents."""
        self._documents.clear()
        logger.info("CAGPipeline: cleared")
