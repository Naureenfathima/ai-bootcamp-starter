"""
memory.py — Agent memory management.

Agents need memory to maintain context across turns. This module provides
four memory types mirroring how human memory works:

  WorkingMemory   — current task context (short-term, per-request scratchpad)
  EpisodicMemory  — conversation history (session-scoped, with summarisation)
  SemanticMemory  — long-term facts, backed by the RAG vector store
  PersistentStore — database-backed persistence across sessions (stub)

Memory architecture:
  ┌──────────────────────────────────────────────────────────┐
  │                     Agent Memory                         │
  │                                                          │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
  │  │  Working    │  │  Episodic   │  │    Semantic     │  │
  │  │  Memory     │  │  Memory     │  │    Memory       │  │
  │  │  (dict)     │  │  (messages) │  │  (RAG / vector) │  │
  │  │  per-task   │  │  per-session│  │  cross-session  │  │
  │  └─────────────┘  └──────┬──────┘  └────────┬────────┘  │
  │                          │                  │            │
  │                   when full:          facts persist      │
  │                   summarise           across restarts    │
  └──────────────────────────────────────────────────────────┘

STUDENT TODO:
  - Implement SemanticMemory using the RAG pipeline from Module 3.
  - Implement PersistentStore using Redis or SQLite.
  - Add memory importance scoring: not every message deserves to be remembered.
  - Add memory decay: old memories become less relevant over time.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Max messages to keep in history before summarisation kicks in
MAX_HISTORY_MESSAGES = 20
# How many messages to keep after summarisation (always keep the most recent)
KEEP_AFTER_SUMMARY = 6


@dataclass
class Message:
    role: str    # "user" | "assistant"
    content: str


# ── Working Memory ─────────────────────────────────────────────────────────────
class WorkingMemory:
    """
    Holds the current task's context: variables, intermediate results, notes.

    Cleared at the start of each new task.
    Think of this as a scratchpad for the agent's current job.

    Example:
        wm = WorkingMemory()
        wm.set("user_intent", "book a flight")
        wm.set("destination", "London")
        print(wm.as_dict())   # {"user_intent": "book a flight", "destination": "London"}
    """

    def __init__(self):
        self._store: dict = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        logger.debug("WorkingMemory.set: %s = %r", key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def clear(self) -> None:
        self._store.clear()

    def as_dict(self) -> dict:
        return dict(self._store)

    def as_context_string(self) -> str:
        """Format working memory as a context string for the LLM system prompt."""
        if not self._store:
            return ""
        lines = ["Current task context:"]
        for k, v in self._store.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ── Episodic Memory ───────────────────────────────────────────────────────────
class EpisodicMemory:
    """
    Stores the conversation history for a session.

    Two strategies for handling long conversations:

    Strategy A — Sliding window (default, simple):
        Drop the oldest messages when the window is full.
        PRO: simple, fast, zero LLM cost.
        CON: context is permanently lost.

    Strategy B — Summarisation (advanced):
        When the window fills, call Claude to summarise the oldest messages
        and replace them with a compact summary injected into the context.
        PRO: no context loss, manageable token usage.
        CON: extra LLM call (small cost).

    STUDENT TODO: implement the summarisation strategy by completing
    the _summarise_oldest() method below and passing use_summarisation=True.
    """

    def __init__(self, use_summarisation: bool = False):
        self._messages: list[Message] = []
        self._use_summarisation = use_summarisation
        self._summary: str = ""   # accumulated summary of compressed messages

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        logger.debug("EpisodicMemory: added %s message (%d chars)", role, len(content))

        if len(self._messages) > MAX_HISTORY_MESSAGES:
            if self._use_summarisation:
                self._summarise_oldest()
            else:
                # Sliding window — drop oldest
                dropped = len(self._messages) - MAX_HISTORY_MESSAGES
                self._messages = self._messages[dropped:]
                logger.debug("EpisodicMemory: slid window, dropped %d messages", dropped)

    def _summarise_oldest(self) -> None:
        """
        Summarise the oldest messages using Claude and replace them with
        a compact summary injected at the start of the context.

        STUDENT TODO: implement this using the Anthropic client.

        Full implementation:
            import anthropic
            from app.config import settings

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            n_to_summarise = len(self._messages) - KEEP_AFTER_SUMMARY
            oldest = self._messages[:n_to_summarise]
            oldest_text = "\\n".join(f"{m.role}: {m.content}" for m in oldest)

            response = client.messages.create(
                model=settings.claude_model,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarise this conversation excerpt in 3-4 sentences, "
                        "preserving key facts, decisions, and user preferences:\\n\\n"
                        + oldest_text
                    )
                }]
            )
            new_summary = response.content[0].text.strip()
            self._summary = (self._summary + " " + new_summary).strip()
            self._messages = self._messages[n_to_summarise:]
            logger.info("EpisodicMemory: summarised %d messages into %d chars",
                        n_to_summarise, len(new_summary))
        """
        # Stub: falls back to sliding window until you implement the above
        n_to_drop = len(self._messages) - KEEP_AFTER_SUMMARY
        self._messages = self._messages[n_to_drop:]
        logger.debug("EpisodicMemory: stub summarisation, dropped %d old messages", n_to_drop)

    def as_claude_messages(self) -> list[dict]:
        """
        Return history in Claude API format, prepending any accumulated summary.
        When summarisation is active, the LLM always has full context.
        """
        messages = []

        # Prepend the running summary so the LLM always has full context
        if self._summary:
            messages.append({
                "role": "user",
                "content": f"[Earlier conversation summary]: {self._summary}"
            })
            messages.append({
                "role": "assistant",
                "content": "Understood, I have the context from our earlier conversation."
            })

        messages += [{"role": m.role, "content": m.content} for m in self._messages]
        return messages

    def last_n(self, n: int) -> list[Message]:
        return self._messages[-n:]

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    def __len__(self) -> int:
        return len(self._messages)


# ── Semantic Memory ───────────────────────────────────────────────────────────
class SemanticMemory:
    """
    Long-term memory backed by the RAG vector store.

    Unlike EpisodicMemory (conversation history), SemanticMemory stores
    facts that should be recalled across sessions. Examples:
      - "The user prefers Python over JavaScript"
      - "The user's company uses PostgreSQL for production"
      - "Project X requires rate limiting on the /query endpoint"

    Architecture:
      When the agent learns a new fact → embed it → store in vector DB
      Before each agent response → retrieve relevant facts → inject into prompt

    STUDENT TODO: connect this to the RAGPipeline from app/rag/pipeline.py.

    Example — remember():
        from app.rag.pipeline import RAGPipeline
        self._rag.ingest_text(fact, source="semantic_memory")

    Example — recall():
        results = self._rag.retrieve(query, top_k=5)
        return [r.chunk.text for r in results if r.score > 0.75]
    """

    def __init__(self):
        # STUDENT TODO: replace stub with:
        # from app.rag.pipeline import RAGPipeline
        # self._rag = RAGPipeline()
        self._facts: list[dict] = []
        logger.info("SemanticMemory initialised (stub — connect to RAGPipeline for persistence)")

    def remember(self, fact: str, source: str = "agent", importance: float = 0.5) -> None:
        """
        Store a fact in long-term memory.

        Args:
            fact:       The fact to remember, e.g. "User prefers async Python."
            source:     Where this fact came from ("agent", "user", "tool").
            importance: 0.0-1.0 score. Higher = surfaces more often in recall.
        """
        # STUDENT TODO: self._rag.ingest_text(fact, source="semantic_memory")
        self._facts.append({"fact": fact, "source": source, "importance": importance})
        logger.info("SemanticMemory: stored — '%s...'", fact[:60])

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve facts relevant to the current context.

        Args:
            query: The current user query or agent state description.
            top_k: Maximum number of facts to return.

        Returns:
            List of relevant fact strings, sorted by importance.
        """
        # STUDENT TODO: replace with vector retrieval:
        # results = self._rag.retrieve(query, top_k=top_k)
        # return [r.chunk.text for r in results if r.score > 0.75]
        if not self._facts:
            return []
        sorted_facts = sorted(self._facts, key=lambda f: f["importance"], reverse=True)
        return [f["fact"] for f in sorted_facts[:top_k]]

    def as_context_string(self, query: str) -> str:
        """Format recalled facts as a context block for the system prompt."""
        facts = self.recall(query)
        if not facts:
            return ""
        lines = ["Long-term memories (relevant facts):"]
        for fact in facts:
            lines.append(f"  - {fact}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._facts.clear()

    def __len__(self) -> int:
        return len(self._facts)


# ── Persistent Store (cross-session) ─────────────────────────────────────────
class PersistentStore:
    """
    Persist agent memory across server restarts.

    STUDENT TODO: implement using Redis (fast, scalable) or SQLite (zero setup).

    Redis sketch:
        import redis, json
        self._redis = redis.from_url(settings.redis_url)

        def save_session(self, session_id, memory):
            data = json.dumps([{"role": m.role, "content": m.content}
                               for m in memory._messages])
            self._redis.setex(f"session:{session_id}", 86400, data)  # 24h TTL

        def load_session(self, session_id):
            raw = self._redis.get(f"session:{session_id}")
            if raw:
                return [Message(**m) for m in json.loads(raw)]
            return []

    SQLite sketch:
        import sqlite3, json
        conn = sqlite3.connect("data/memory.db")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(id TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)"
        )
    """

    def save_session(self, session_id: str, memory: EpisodicMemory) -> None:
        """Persist a session's episodic memory. STUDENT TODO: implement."""
        logger.warning("PersistentStore.save_session is a stub — implement with Redis or SQLite")

    def load_session(self, session_id: str) -> Optional[list[Message]]:
        """Load a previously saved session. STUDENT TODO: implement."""
        logger.warning("PersistentStore.load_session is a stub — implement with Redis or SQLite")
        return None

    def delete_session(self, session_id: str) -> None:
        """Delete a session from the store. STUDENT TODO: implement."""
        pass
