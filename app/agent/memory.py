"""
memory.py — Agent memory management.

Agents need memory to maintain context across turns. This module provides
three memory types mirroring how human memory works:

  WorkingMemory  — current task context (short-term, per-request)
  EpisodicMemory — conversation history (session-scoped)
  SemanticMemory — long-term facts (persistent across sessions, not implemented here)

STUDENT TODO:
  - Add summarisation: when history exceeds the context window, summarise
    older messages using Claude and replace them with a compact summary.
  - Implement SemanticMemory using the RAG pipeline from Module 3.
  - Add memory persistence: save EpisodicMemory to a database between sessions.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Max messages to keep in history before sliding window kicks in
MAX_HISTORY_MESSAGES = 20


@dataclass
class Message:
    role: str    # "user" | "assistant"
    content: str


class WorkingMemory:
    """
    Holds the current task's context: variables, intermediate results, notes.

    Cleared at the start of each new task.
    Think of this as a scratchpad for the agent's current job.
    """

    def __init__(self):
        self._store: dict = {}

    def set(self, key: str, value) -> None:
        self._store[key] = value
        logger.debug("WorkingMemory.set: %s = %r", key, value)

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def clear(self) -> None:
        self._store.clear()

    def as_dict(self) -> dict:
        return dict(self._store)


class EpisodicMemory:
    """
    Stores the conversation history for a session.

    Implements a sliding window: when history exceeds MAX_HISTORY_MESSAGES,
    the oldest messages are dropped to keep the context window manageable.

    STUDENT TODO: replace the sliding window with a summarisation strategy
    so that no context is truly lost — it is just compressed.
    """

    def __init__(self):
        self._messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))

        # Sliding window: drop oldest messages if over limit
        if len(self._messages) > MAX_HISTORY_MESSAGES:
            dropped = len(self._messages) - MAX_HISTORY_MESSAGES
            self._messages = self._messages[dropped:]
            logger.debug("EpisodicMemory: dropped %d old messages (window=%d)",
                         dropped, MAX_HISTORY_MESSAGES)

    def as_claude_messages(self) -> list[dict]:
        """Return history in the format Claude's API expects."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def last_n(self, n: int) -> list[Message]:
        return self._messages[-n:]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
