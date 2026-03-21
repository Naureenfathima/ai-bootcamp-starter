"""
embeddings.py — Text embedding generation (RAG Stage 2).

Converts text chunks into numerical vectors using an embedding model.
The same model MUST be used at ingestion time and at query time.

STUDENT TODO:
  - Try different embedding models and compare retrieval quality on your dataset.
  - Add batch processing to avoid hitting API rate limits on large document sets.
  - Add a local embedding option using sentence-transformers (no API needed).
"""

import logging
from typing import Protocol

from app.config import settings
from app.rag.ingestion import Chunk

logger = logging.getLogger(__name__)


# ── Protocol (interface) ──────────────────────────────────────────────────────
class EmbeddingProvider(Protocol):
    """Any class that can turn text into a list of floats is a valid provider."""

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


# ── OpenAI Provider ───────────────────────────────────────────────────────────
class OpenAIEmbeddings:
    """
    Embeddings via the OpenAI API.

    Usage:
        provider = OpenAIEmbeddings()
        vector = provider.embed("What is RAG?")
    """

    def __init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model  = settings.embedding_model
        except ImportError:
            raise ImportError("Run: pip install openai")

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in resp.data]


# ── Simple Local Provider (no API needed) ─────────────────────────────────────
class LocalEmbeddings:
    """
    Embeddings using sentence-transformers — runs entirely on your machine.
    Great for development without an API key.

    Usage:
        provider = LocalEmbeddings()
        vector = provider.embed("What is RAG?")

    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()


# ── Factory ───────────────────────────────────────────────────────────────────
def get_embedding_provider() -> EmbeddingProvider:
    """
    Return the embedding provider configured in .env.

    EMBEDDING_PROVIDER=openai   → OpenAIEmbeddings
    EMBEDDING_PROVIDER=local    → LocalEmbeddings (sentence-transformers)
    """
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbeddings()
    elif provider == "local":
        return LocalEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}. "
                         "Set EMBEDDING_PROVIDER=openai or EMBEDDING_PROVIDER=local in .env")


# ── Embed chunks ──────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """
    Generate embeddings for a list of Chunk objects.

    Returns:
        List of (chunk, embedding_vector) tuples.
    """
    provider = get_embedding_provider()
    texts    = [chunk.text for chunk in chunks]

    logger.info("Embedding %d chunks with provider='%s' ...", len(chunks), settings.embedding_provider)
    vectors = provider.embed_batch(texts)
    logger.info("Embedding complete.")

    return list(zip(chunks, vectors))
