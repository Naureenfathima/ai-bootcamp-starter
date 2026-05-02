"""
production/cache.py — Semantic caching for LLM responses.

Problem: LLM calls are expensive and slow (~1-3 seconds, $0.001-0.01 each).
If 30% of your users ask similar questions, you are paying for the same
computation over and over.

Solution: Semantic cache — instead of exact string matching, embed the
incoming query and compare it to cached query embeddings. If a cached
query is "close enough" (cosine similarity > threshold), return the
cached answer immediately.

Impact:
  - Typical LLM call: 1500ms, $0.005
  - Cache hit:          5ms, $0.000
  - 30% cache hit rate → ~30% cost reduction and ~30% latency reduction

Comparison with regular caching:
  Regular cache:  "What is RAG?" != "Explain RAG to me" (different string → miss)
  Semantic cache: cosine([embed("What is RAG?")], [embed("Explain RAG to me")]) = 0.94 → HIT

STUDENT TODO:
  - Replace the in-memory store with Redis for persistence across restarts.
  - Add a TTL (time-to-live) to auto-expire stale cached answers.
  - Add cache analytics: track hit rate, most-cached queries, cost savings.
  - Consider a two-tier cache: exact match first (free), semantic match second.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One cached item: the original query, its embedding, and the cached answer."""
    query: str
    embedding: list[float]
    answer: str
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: Optional[float] = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """
    A semantic cache for LLM responses.

    Unlike exact-match caches, this uses embedding similarity to detect
    semantically equivalent queries even when the wording differs.

    Usage:
        cache = SemanticCache(similarity_threshold=0.92)

        # Before calling the LLM:
        hit = cache.get(query, query_embedding)
        if hit:
            return hit   # free and instant!

        # After calling the LLM:
        response = llm.generate(query)
        cache.put(query, query_embedding, response)

    Integration with RAGPipeline:
        from app.production.cache import SemanticCache
        from app.rag.embeddings import get_embedding_provider

        embed = get_embedding_provider()
        cache = SemanticCache()

        def cached_rag_query(question):
            vec = embed.embed(question)
            cached = cache.get(question, vec)
            if cached:
                return cached
            result = rag_pipeline.query(question)
            cache.put(question, vec, result.response)
            return result.response
    """

    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 500):
        self._entries: list[CacheEntry] = []
        self._threshold = similarity_threshold
        self._max_size  = max_size
        self._total_queries = 0
        self._cache_hits    = 0
        logger.info("SemanticCache: threshold=%.2f max_size=%d", similarity_threshold, max_size)

    def get(self, query: str, embedding: list[float]) -> Optional[str]:
        """
        Look up a query in the semantic cache.

        Returns:
            The cached answer if a semantically similar query was found,
            otherwise None.
        """
        self._total_queries += 1

        best_score = 0.0
        best_entry = None

        for entry in self._entries:
            score = cosine_similarity(embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._threshold:
            best_entry.hits += 1
            best_entry.last_hit_at = time.time()
            self._cache_hits += 1
            logger.info("Cache HIT: similarity=%.3f query='%s...'", best_score, query[:50])
            return best_entry.answer

        logger.debug("Cache MISS: best_similarity=%.3f query='%s...'", best_score, query[:50])
        return None

    def put(self, query: str, embedding: list[float], answer: str) -> None:
        """Store a new query-answer pair in the cache."""
        # Evict oldest entries if at capacity
        if len(self._entries) >= self._max_size:
            # LRU-like: remove entry with oldest last_hit_at (or created_at)
            self._entries.sort(key=lambda e: e.last_hit_at or e.created_at)
            self._entries.pop(0)

        self._entries.append(CacheEntry(query=query, embedding=embedding, answer=answer))
        logger.debug("Cache PUT: '%s...' (cache size=%d)", query[:50], len(self._entries))

    def invalidate(self, query: str, embedding: list[float]) -> int:
        """Remove all entries semantically similar to the given query."""
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if cosine_similarity(embedding, e.embedding) < self._threshold
        ]
        removed = before - len(self._entries)
        logger.info("Cache INVALIDATE: removed %d entries", removed)
        return removed

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        logger.info("Cache cleared")

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0-1.0)."""
        if self._total_queries == 0:
            return 0.0
        return self._cache_hits / self._total_queries

    def stats(self) -> dict:
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "hit_rate": round(self.hit_rate, 3),
            "threshold": self._threshold,
            "top_queries": sorted(
                [{"query": e.query[:60], "hits": e.hits} for e in self._entries],
                key=lambda x: x["hits"], reverse=True
            )[:5],
        }
