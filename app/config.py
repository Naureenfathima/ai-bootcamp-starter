"""
config.py — Central configuration using environment variables.

All secrets and settings live here. Never hardcode values in other files.
Load this with:  from app.config import settings
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "AI Bootcamp Starter"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Anthropic / Claude ────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20241022"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Options: "openai" | "cohere" | "local"
    embedding_provider: str = "openai"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Vector Database ───────────────────────────────────────────────────────
    # Options: "memory" (dev) | "pgvector" (prod)
    vector_store: str = "memory"
    database_url: str = "postgresql://user:password@localhost:5432/aibootcamp"

    # ── Voice AI ──────────────────────────────────────────────────────────────
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 512           # tokens per chunk
    chunk_overlap: int = 64         # overlap between chunks
    retrieval_top_k: int = 3        # how many chunks to retrieve
    similarity_threshold: float = 0.7

    # ── Agent ─────────────────────────────────────────────────────────────────
    max_agent_steps: int = 10       # safety limit on tool-call loops
    agent_temperature: float = 0.2  # low = more deterministic

    # ── Agentic RAG ───────────────────────────────────────────────────────────
    agentic_max_iterations: int = 3         # max retrieve-reflect cycles per sub-question
    agentic_reflection_threshold: float = 0.7  # score above which evidence is "sufficient"

    # ── Page Index ────────────────────────────────────────────────────────────
    page_index_page_size: int = 800         # words per logical page

    # ── Voice AI ─────────────────────────────────────────────────────────────
    voice_stt_confidence_threshold: float = 0.5  # below this, ask user to repeat

    # ── LLM Provider (swap without changing code) ────────────────────────────
    # Options: "anthropic" | "openai" | "ollama" | "groq" | "together" | "openrouter"
    llm_provider: str = "anthropic"
    llm_model: str = ""            # overrides default model for the chosen provider

    # Provider-specific keys (only the one matching llm_provider is needed)
    groq_api_key: str = ""
    together_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once)."""
    return Settings()


# Convenience alias — import this in other modules
settings = get_settings()
