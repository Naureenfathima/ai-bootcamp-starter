"""
ingestion.py — Document loading and chunking (RAG Stage 1).

This module reads documents from disk and splits them into chunks that
can be embedded and stored in the vector database.

STUDENT TODO:
  - Try different chunk_size values (256, 512, 1024) and compare retrieval quality.
  - Add support for PDF ingestion using PyPDF2 or pdfplumber.
  - Add a recursive character splitter for HTML / Markdown content.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text from a document."""
    text: str
    source: str          # filename or URL
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def load_text_file(path: str | Path) -> str:
    """Read a plain text file and return its contents."""
    return Path(path).read_text(encoding="utf-8")


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text:          The full document text.
        source:        A label for where the text came from (filename, URL, etc.).
        chunk_size:    Target words per chunk (defaults to settings.chunk_size).
        chunk_overlap: Words to repeat at chunk boundaries (defaults to settings.chunk_overlap).

    Returns:
        List of Chunk objects.
    """
    chunk_size    = chunk_size    if chunk_size    is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                text=" ".join(chunk_words),
                source=source,
                chunk_index=index,
                metadata={"word_count": len(chunk_words)},
            )
        )
        index += 1
        # Move forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap
        if start >= len(words):
            break

    logger.info("Chunked '%s' into %d chunks (size=%d, overlap=%d)",
                source, len(chunks), chunk_size, chunk_overlap)
    return chunks


def ingest_directory(directory: str | Path) -> list[Chunk]:
    """
    Load and chunk all .txt files in a directory.

    STUDENT TODO: extend this to handle .pdf and .md files.
    """
    directory = Path(directory)
    all_chunks: list[Chunk] = []

    for filepath in sorted(directory.glob("*.txt")):
        text = load_text_file(filepath)
        chunks = chunk_text(text, source=filepath.name)
        all_chunks.extend(chunks)
        logger.info("Ingested: %s (%d chunks)", filepath.name, len(chunks))

    logger.info("Total chunks ingested: %d", len(all_chunks))
    return all_chunks
