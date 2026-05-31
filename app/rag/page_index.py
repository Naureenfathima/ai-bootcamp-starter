"""
page_index.py — Hierarchical page-level document index.

PageIndex implements two-tier retrieval:
  Tier 1 (coarse) — Find relevant PAGES by embedding similarity on page summaries.
  Tier 2 (fine)   — Within those pages, retrieve specific CHUNKS.

This mirrors how humans read a long report: scan section headings to find the
right section, then read only that section in detail.

Why it matters:
  Flat chunk lists lose document structure. A 50-page whitepaper chunked into
  300 equal word-windows will mix content from completely unrelated sections.
  PageIndex preserves the hierarchical structure so retrieval stays coherent.

Architecture:

  Document
  ├── Page 1 ──→ LLM summary ──→ summary embedding [Tier 1 retrieval]
  │   ├── Chunk 1.1
  │   ├── Chunk 1.2  ◄──────────────────────────── [Tier 2 retrieval]
  │   └── Chunk 1.3
  ├── Page 2 ──→ LLM summary
  │   └── ...

Build once (offline), query many times (online). The LLM summarisation
happens only at build time — each query is cheap.

Systems lesson:
  Two-tier indexes are a general pattern in data engineering:
    - B-tree index pages → leaf pages (databases)
    - Index segments → posting lists (search engines)
    - CDN edge → origin (content delivery)
  The trade-off is always: pay at write time to save at read time.

STUDENT TODO:
  - Add GET /rag/page-index/toc — return table of contents as JSON
  - Generate 3 "example questions" per page — retrieve by question similarity instead of
    by summary similarity for even better precision
  - Cache LLM summaries to disk so rebuilds skip the API call if content is unchanged
  - Experiment with different page sizes (500 vs 2000 words)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from app.config import settings
from app.rag.ingestion import Chunk, chunk_text
from app.rag.embeddings import get_embedding_provider
from app.rag.retrieval import InMemoryVectorStore, SearchResult, cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_PAGE_WORDS = 800    # words per logical page


@dataclass
class Page:
    """A logical page within a document."""
    page_num: int           # 1-indexed
    source: str
    text: str               # raw text for this page
    summary: str            # LLM-generated summary
    word_count: int
    summary_embedding: list[float] = field(default_factory=list)
    chunks: list[Chunk]     = field(default_factory=list)


@dataclass
class PageIndexResult:
    """Result returned by PageIndex.query()."""
    answer: str
    pages_searched: list[int]       # page numbers that contributed
    chunks_used: list[str]          # actual chunk texts fed to the LLM
    input_tokens: int
    output_tokens: int
    retrieval_latency_ms: float
    total_latency_ms: float


class PageIndex:
    """
    Two-tier hierarchical document index.

    Build phase (offline — runs once per document):
        index = PageIndex()
        stats = index.build_from_text(long_text, source="whitepaper.txt")

    Query phase (online — fast):
        result = index.query("What are the key findings on scalability?")

    Inspect:
        toc = index.table_of_contents()   # list of {page, summary}
    """

    def __init__(
        self,
        page_size_words: int = DEFAULT_PAGE_WORDS,
        top_pages: int = 2,
        top_chunks_per_page: int = 2,
    ):
        self._client      = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._embed       = get_embedding_provider()
        self._pages: list[Page] = []
        self._page_size   = page_size_words
        self._top_pages   = top_pages
        self._top_chunks  = top_chunks_per_page
        logger.info("PageIndex initialised (page_size=%d words)", page_size_words)

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_from_text(self, text: str, source: str) -> dict:
        """
        Full build pipeline:
          1. Split document into word-count pages
          2. LLM-summarise each page  (Tier 1 index — one API call per page)
          3. Chunk each page and embed chunks  (Tier 2 index)

        Returns build statistics (pages, chunks, latency).
        """
        t0 = time.perf_counter()
        words = text.split()
        page_texts = [
            " ".join(words[i: i + self._page_size])
            for i in range(0, len(words), self._page_size)
        ]

        logger.info("PageIndex.build: splitting into %d pages (%d total words) from '%s'",
                    len(page_texts), len(words), source)

        total_chunks = 0
        for page_num, page_text in enumerate(page_texts, start=1):
            # Phase 2: LLM summary for Tier 1
            summary = self._summarise(page_text, page_num, source)
            summary_vec = self._embed.embed(summary)

            # Phase 3: fine-grained chunks for Tier 2
            page_chunks = chunk_text(page_text, source=f"{source}:p{page_num}")
            chunk_vecs  = self._embed.embed_batch([c.text for c in page_chunks])
            total_chunks += len(page_chunks)

            self._pages.append(Page(
                page_num=page_num,
                source=source,
                text=page_text,
                summary=summary,
                word_count=len(page_text.split()),
                summary_embedding=summary_vec,
                chunks=list(page_chunks),
            ))
            logger.debug("PageIndex: page %d built (%d chunks)", page_num, len(page_chunks))

        build_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("PageIndex.build done: %d pages, %d chunks, %sms",
                    len(self._pages), total_chunks, build_ms)
        return {
            "pages": len(self._pages),
            "chunks": total_chunks,
            "source": source,
            "build_latency_ms": build_ms,
        }

    def _summarise(self, page_text: str, page_num: int, source: str) -> str:
        """One LLM call to summarise a single page."""
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise this excerpt from '{source}' (page {page_num}) "
                    f"in 2-3 sentences. Preserve key facts, entities, and concepts.\n\n"
                    f"{page_text[:2000]}"
                ),
            }],
        )
        return response.content[0].text.strip()

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, question: str) -> PageIndexResult:
        """
        Two-tier retrieval + generation.

        Tier 1: cosine similarity between question embedding and page summary embeddings
                → select top_pages most relevant pages

        Tier 2: cosine similarity within each selected page's chunks
                → select top_chunks_per_page most relevant chunks per page

        Generate: Claude answers using the selected chunks as context.
        """
        if not self._pages:
            return PageIndexResult(
                answer="No documents indexed. Call build_from_text() first.",
                pages_searched=[], chunks_used=[],
                input_tokens=0, output_tokens=0,
                retrieval_latency_ms=0, total_latency_ms=0,
            )

        t_total = time.perf_counter()
        t_ret = time.perf_counter()

        # Tier 1: rank pages by summary similarity
        query_vec = self._embed.embed(question)
        ranked_pages = sorted(
            self._pages,
            key=lambda p: cosine_similarity(query_vec, p.summary_embedding),
            reverse=True,
        )
        top_pages = ranked_pages[: self._top_pages]

        logger.info("PageIndex.query: top pages = [%s]",
                    ", ".join(f"p{p.page_num}" for p in top_pages))

        # Tier 2: rank chunks within each selected page
        selected_chunks: list[str] = []
        for page in top_pages:
            if not page.chunks:
                continue
            chunk_vecs = self._embed.embed_batch([c.text for c in page.chunks])
            scored = [
                (chunk, cosine_similarity(query_vec, vec))
                for chunk, vec in zip(page.chunks, chunk_vecs)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_chunks.extend(c.text for c, _ in scored[: self._top_chunks])

        retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        # Generation
        context_parts = []
        for i, page in enumerate(top_pages):
            context_parts.append(f"[Page {page.page_num} summary: {page.summary}]")
        context_parts.extend(selected_chunks)
        context = "\n\n".join(context_parts)

        message = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=(
                "You are a precise assistant. Answer using ONLY the provided page excerpts. "
                "Reference page numbers when citing facts."
            ),
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }],
        )

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        return PageIndexResult(
            answer=message.content[0].text,
            pages_searched=[p.page_num for p in top_pages],
            chunks_used=selected_chunks,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            retrieval_latency_ms=retrieval_ms,
            total_latency_ms=total_ms,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    def table_of_contents(self) -> list[dict]:
        """Return page summaries as a structured TOC."""
        return [
            {
                "page": p.page_num,
                "source": p.source,
                "word_count": p.word_count,
                "summary": p.summary,
            }
            for p in self._pages
        ]

    def get_page(self, page_num: int) -> Optional[Page]:
        """Retrieve a specific page by 1-indexed number."""
        for p in self._pages:
            if p.page_num == page_num:
                return p
        return None

    def stats(self) -> dict:
        return {
            "pages": len(self._pages),
            "total_words": sum(p.word_count for p in self._pages),
            "sources": sorted({p.source for p in self._pages}),
        }
