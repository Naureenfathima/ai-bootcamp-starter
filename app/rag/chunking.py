from __future__ import annotations
"""
chunking.py — Pluggable chunking strategies (RAG Stage 1).

Chunking is one of the highest-leverage decisions in RAG pipeline design.
Bad chunking leads to:
  - Sentences cut in half → broken context at retrieval time
  - Chunks mixing unrelated topics → low retrieval precision
  - Too-small chunks → lost context; too-large → noisy retrieval

This module implements four strategies at increasing sophistication.
All strategies return the same Chunk type, so they are drop-in replacements
for each other in any RAGPipeline or PageIndex.

Strategy comparison:

  Strategy         │ Speed    │ API cost │ Respects structure │ Best for
  ─────────────────┼──────────┼──────────┼────────────────────┼──────────────────
  FixedSize        │ Fastest  │ None     │ No                 │ Quick experiments
  Recursive        │ Fast     │ None     │ Paragraphs/lines   │ General purpose
  Sentence         │ Fast     │ None     │ Sentence boundaries│ Q&A over prose
  Semantic         │ Slow     │ API call │ Meaning boundaries │ Precision retrieval

Systems lesson:
  There is no universally "best" chunking strategy. The right choice depends on
  your document type, query distribution, and retrieval quality targets.
  Always benchmark strategies on your actual data before choosing one for production.
  Design the abstraction (ChunkingStrategy Protocol) so you can swap strategies
  without touching the rest of the pipeline.

STUDENT TODO:
  - Benchmark all four strategies on the same query set.
  - Plot retrieval quality (faithfulness + relevance) vs chunk size.
  - Try a hybrid: sentence-chunk then merge short sentences up to a token budget.
  - For PDFs: try page-level chunking and compare with word-level.
"""

import re
import math
import logging
from typing import Protocol, runtime_checkable, Optional

from app.config import settings
from app.rag.ingestion import Chunk

logger = logging.getLogger(__name__)


# ── Protocol: the interface every strategy must satisfy ───────────────────────
@runtime_checkable
class ChunkingStrategy(Protocol):
    """
    Any class implementing chunk() is a valid ChunkingStrategy.
    No inheritance required — this is structural subtyping (duck typing + Protocol).

    This design lets you add a new strategy in a new file without
    modifying this module at all.
    """

    def chunk(self, text: str, source: str) -> list[Chunk]:
        ...


# ── Strategy 1: Fixed-Size ────────────────────────────────────────────────────
class FixedSizeChunker:
    """
    Slide a fixed-width window over words with configurable overlap.

    This is the simplest strategy and what the base RAGPipeline uses.
    It is fast and deterministic but completely ignores sentence and
    paragraph structure — a sentence can be cut in half between chunks.

    When to use:
      - Rapid prototyping when you need any chunking working fast
      - Structured data (tables, logs, CSV) where sentences don't matter
      - Baseline: always compare other strategies against this one

    When NOT to use:
      - Prose documents where coherent sentences carry meaning
      - Documents with strong paragraph structure you want to preserve
    """

    def __init__(
        self,
        chunk_size:Optional[ int] = None,
        chunk_overlap:Optional[ int] = None,
    ):
        self.chunk_size    = chunk_size    or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        chunks: list[Chunk] = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(Chunk(
                text=" ".join(chunk_words),
                source=source,
                chunk_index=len(chunks),
                metadata={"strategy": "fixed_size", "word_count": len(chunk_words)},
            ))
            start += self.chunk_size - self.chunk_overlap
            if start >= len(words):
                break

        logger.debug("FixedSizeChunker: %d chunks from '%s'", len(chunks), source)
        return chunks


# ── Strategy 2: Recursive Character ──────────────────────────────────────────
class RecursiveChunker:
    """
    Split by trying progressively finer separators until chunks fit.

    Separator hierarchy (tried in order — use the coarsest that works):
      1. \\n\\n  — paragraph break  (most structure-preserving)
      2. \\n    — line break
      3. . / ! / ?  — sentence boundary
      4. ,      — clause boundary
      5. (space) — word boundary   (last resort)

    This mirrors LangChain's RecursiveCharacterTextSplitter.

    Why "recursive"? The algorithm recurses through the separator list:
      try paragraph breaks → if a piece is still too long, try line breaks
      → if still too long, try sentence breaks → ... → word breaks.

    When to use:
      - Documents with natural paragraph structure (articles, reports, docs)
      - Mixed-format content (prose + code blocks + lists)
      - General-purpose default when you don't know your document type in advance

    Trade-off: chunks have variable size. Some paragraphs are long, some short.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def __init__(
        self,
        chunk_size:Optional[ int] = None,
        chunk_overlap:Optional[ int] = None,
    ):
        self.chunk_size    = chunk_size    or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        pieces = self._split(text.strip())
        chunks: list[Chunk] = []
        current = ""
        current_words = 0

        for piece in pieces:
            piece_words = len(piece.split())
            if current_words + piece_words > self.chunk_size and current:
                chunks.append(Chunk(
                    text=current.strip(),
                    source=source,
                    chunk_index=len(chunks),
                    metadata={"strategy": "recursive", "word_count": current_words},
                ))
                # Carry overlap from the tail of the completed chunk
                tail = current.split()[-self.chunk_overlap:]
                current = " ".join(tail) + " " + piece if tail else piece
                current_words = len(current.split())
            else:
                current = (current + " " + piece).strip() if current else piece
                current_words = len(current.split())

        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                source=source,
                chunk_index=len(chunks),
                metadata={"strategy": "recursive", "word_count": len(current.split())},
            ))

        logger.debug("RecursiveChunker: %d chunks from '%s'", len(chunks), source)
        return chunks

    def _split(self, text: str) -> list[str]:
        """Use the highest-priority separator that actually splits the text."""
        for sep in self.SEPARATORS:
            parts = text.split(sep)
            if len(parts) > 1:
                return [p.strip() for p in parts if p.strip()]
        return [text]


# ── Strategy 3: Sentence-Aware ────────────────────────────────────────────────
class SentenceChunker:
    """
    Split into sentences first, then group sentences until the chunk is full.

    Every chunk starts and ends on a sentence boundary — no sentence is ever
    cut in half. This produces the most coherent chunks for Q&A tasks where
    the full meaning of a sentence must be preserved.

    When to use:
      - Customer support Q&A
      - Article or blog post retrieval
      - Any domain where complete sentences carry the unit of meaning

    Trade-off: chunks may exceed the target size if a single sentence is very
    long (e.g. academic prose with 150-word sentences). Works best on
    well-punctuated prose.
    """

    SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")

    def __init__(
        self,
        chunk_size:Optional[ int] = None,
        chunk_overlap:Optional[ int] = None,
    ):
        self.chunk_size    = chunk_size    or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = self.SENTENCE_END.split(text)
        if len(sentences) == 1:
            sentences = [text]

        chunks: list[Chunk] = []
        bucket: list[str] = []
        bucket_words = 0

        for sentence in sentences:
            s_words = len(sentence.split())
            if bucket_words + s_words > self.chunk_size and bucket:
                chunks.append(Chunk(
                    text=" ".join(bucket),
                    source=source,
                    chunk_index=len(chunks),
                    metadata={
                        "strategy": "sentence",
                        "sentence_count": len(bucket),
                        "word_count": bucket_words,
                    },
                ))
                # Overlap: keep last few sentences
                overlap_n = max(1, self.chunk_overlap // max(s_words, 1))
                bucket = bucket[-overlap_n:] + [sentence]
                bucket_words = sum(len(s.split()) for s in bucket)
            else:
                bucket.append(sentence)
                bucket_words += s_words

        if bucket:
            chunks.append(Chunk(
                text=" ".join(bucket),
                source=source,
                chunk_index=len(chunks),
                metadata={
                    "strategy": "sentence",
                    "sentence_count": len(bucket),
                    "word_count": bucket_words,
                },
            ))

        logger.debug("SentenceChunker: %d chunks (%d sentences) from '%s'",
                     len(chunks), len(sentences), source)
        return chunks


# ── Strategy 4: Semantic ──────────────────────────────────────────────────────
class SemanticChunker:
    """
    Split at points where MEANING changes, detected via embedding similarity.

    Algorithm:
      1. Split text into sentences (using SentenceChunker's regex)
      2. Embed each sentence (one batch API call)
      3. Compute cosine similarity between adjacent sentence embeddings
      4. Mark breakpoints where similarity drops below a percentile threshold
         (= topic change between adjacent sentences)
      5. Group sentences between breakpoints into chunks

    This produces the semantically purest chunks: each chunk covers one topic.
    The trade-off is API cost (one embedding call per sentence at build time).

    Systems note:
      This is an example of using AI to improve AI — using embeddings to decide
      where to chunk a document that will later be retrieved by embeddings.
      It is worth it when precision matters more than index build time.

    STUDENT TODO:
      - Tune breakpoint_percentile (75 = top 25% of similarity drops are breakpoints)
      - Try different embedding models for chunking detection vs. retrieval
      - Add caching: skip re-embedding if the document hasn't changed
    """

    SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")

    def __init__(
        self,
        chunk_size:Optional[ int] = None,
        breakpoint_percentile: int = 75,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.breakpoint_percentile = breakpoint_percentile

    def chunk(self, text: str, source: str) -> list[Chunk]:
        from app.rag.embeddings import get_embedding_provider

        text = re.sub(r"\s+", " ", text).strip()
        sentences = self.SENTENCE_END.split(text)
        if len(sentences) <= 1:
            logger.debug("SemanticChunker: single sentence, returning 1 chunk")
            return [Chunk(
                text=text, source=source, chunk_index=0,
                metadata={"strategy": "semantic", "word_count": len(text.split())},
            )]

        provider = get_embedding_provider()
        embeddings = provider.embed_batch(sentences)
        logger.debug("SemanticChunker: embedded %d sentences", len(sentences))

        similarities = [
            self._cosine(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Identify breakpoints at the lowest similarity percentile
        sorted_sims = sorted(similarities)
        threshold_idx = int(len(sorted_sims) * (self.breakpoint_percentile / 100))
        threshold = sorted_sims[min(threshold_idx, len(sorted_sims) - 1)]
        breakpoints = {
            i + 1 for i, sim in enumerate(similarities) if sim < threshold
        }

        chunks: list[Chunk] = []
        current: list[str] = []

        for i, sentence in enumerate(sentences):
            if i in breakpoints and current:
                chunk_text = " ".join(current)
                chunks.append(Chunk(
                    text=chunk_text, source=source, chunk_index=len(chunks),
                    metadata={"strategy": "semantic", "word_count": len(chunk_text.split())},
                ))
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                text=chunk_text, source=source, chunk_index=len(chunks),
                metadata={"strategy": "semantic", "word_count": len(chunk_text.split())},
            ))

        logger.info("SemanticChunker: %d chunks from %d sentences (threshold=%.3f)",
                    len(chunks), len(sentences), threshold)
        return chunks

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return dot / mag if mag else 0.0


# ── Factory ───────────────────────────────────────────────────────────────────
def get_chunker(strategy: str = "fixed", **kwargs) -> ChunkingStrategy:
    """
    Return a chunking strategy by name.

    Args:
        strategy: "fixed" | "recursive" | "sentence" | "semantic"
        **kwargs: Forwarded to the strategy constructor (chunk_size, chunk_overlap, …)

    Raises:
        ValueError: if strategy name is not recognised.

    Example:
        chunker = get_chunker("sentence", chunk_size=300, chunk_overlap=30)
        chunks  = chunker.chunk(document_text, source="whitepaper.pdf")
    """
    registry: dict[str, type] = {
        "fixed":     FixedSizeChunker,
        "recursive": RecursiveChunker,
        "sentence":  SentenceChunker,
        "semantic":  SemanticChunker,
    }
    cls = registry.get(strategy.lower())
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Available: {sorted(registry)}"
        )
    return cls(**kwargs)
