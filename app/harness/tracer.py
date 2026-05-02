"""
harness/tracer.py — Request tracing and LLM observability.

Observability for LLM apps means being able to answer:
  - How long did each stage take? (STT, retrieval, LLM, TTS)
  - How many tokens did we use? (cost tracking)
  - Which prompt version produced this response? (prompt versioning)
  - What context was retrieved? (debugging hallucinations)
  - Did the eval score change after my last deploy? (regression detection)

This module provides lightweight, zero-dependency tracing you can run
locally. For production, export traces to Langfuse, LangSmith, or Datadog.

STUDENT TODO:
  - Export spans to Langfuse: https://langfuse.com/docs/sdk/python
  - Add prompt versioning: hash prompt templates and store version IDs with spans.
  - Add cost alerts: warn when a session exceeds a token budget.
  - Integrate with the CI pipeline to track metrics over time.
"""

import time
import uuid
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """
    A single timed operation within a request trace.

    Analogous to a span in OpenTelemetry or Zipkin.
    """
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"   # "running" | "ok" | "error"
    error: Optional[str] = None

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        self.end_time = time.perf_counter()
        self.status = status
        self.error = error

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return round((time.perf_counter() - self.start_time) * 1000, 1)
        return round((self.end_time - self.start_time) * 1000, 1)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """
    A complete trace for a single request (e.g. one RAG query or agent run).
    Contains multiple spans representing each stage.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = "request"
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def finish(self) -> None:
        self.end_time = time.perf_counter()

    @property
    def total_ms(self) -> float:
        if self.end_time is None:
            return round((time.perf_counter() - self.start_time) * 1000, 1)
        return round((self.end_time - self.start_time) * 1000, 1)

    @property
    def total_tokens(self) -> int:
        total = 0
        for span in self.spans:
            total += span.metadata.get("input_tokens", 0)
            total += span.metadata.get("output_tokens", 0)
        return total

    @property
    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate based on Claude claude-3-5-sonnet pricing.
        Update COST_PER_1K_TOKENS as pricing changes.
        """
        COST_PER_1K_INPUT  = 0.003   # $3 / 1M input tokens
        COST_PER_1K_OUTPUT = 0.015   # $15 / 1M output tokens
        input_tokens  = sum(s.metadata.get("input_tokens", 0) for s in self.spans)
        output_tokens = sum(s.metadata.get("output_tokens", 0) for s in self.spans)
        return round(
            (input_tokens / 1000) * COST_PER_1K_INPUT
            + (output_tokens / 1000) * COST_PER_1K_OUTPUT,
            6
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "total_ms": self.total_ms,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }

    def log_summary(self) -> None:
        """Print a human-readable trace summary to the logger."""
        logger.info(
            "TRACE [%s] %s | total=%sms | tokens=%d | cost=$%.5f",
            self.trace_id, self.name, self.total_ms,
            self.total_tokens, self.estimated_cost_usd,
        )
        for span in self.spans:
            status_icon = "✓" if span.status == "ok" else "✗"
            logger.info(
                "  %s span[%s] %s | %sms | tokens=%d",
                status_icon, span.span_id, span.name, span.duration_ms,
                span.metadata.get("input_tokens", 0) + span.metadata.get("output_tokens", 0),
            )


class RequestTracer:
    """
    Context-manager-based request tracer.

    Wraps any operation in a span, times it, and attaches metadata.
    Traces are stored in memory and can be exported to external systems.

    Usage:
        tracer = RequestTracer()

        with tracer.new_trace("rag_query") as trace:
            with tracer.span(trace, "retrieval") as span:
                results = vector_store.search(query)
                span.metadata["chunks_retrieved"] = len(results)

            with tracer.span(trace, "llm_generation") as span:
                response = claude.generate(...)
                span.metadata["input_tokens"] = response.usage.input_tokens
                span.metadata["output_tokens"] = response.usage.output_tokens

        trace.log_summary()
    """

    def __init__(self, max_traces: int = 1000):
        self._traces: list[Trace] = []
        self._max_traces = max_traces

    @contextmanager
    def new_trace(self, name: str, metadata: dict | None = None):
        """Create and manage a new trace for a request."""
        trace = Trace(name=name, metadata=metadata or {})
        try:
            yield trace
        except Exception as e:
            trace.metadata["error"] = str(e)
            raise
        finally:
            trace.finish()
            self._store_trace(trace)
            trace.log_summary()

    @contextmanager
    def span(self, trace: Trace, name: str, metadata: dict | None = None):
        """Create a child span within a trace."""
        s = Span(name=name, trace_id=trace.trace_id, metadata=metadata or {})
        trace.add_span(s)
        try:
            yield s
            s.finish(status="ok")
        except Exception as e:
            s.finish(status="error", error=str(e))
            raise

    def _store_trace(self, trace: Trace) -> None:
        """Store a trace, evicting the oldest if at capacity."""
        if len(self._traces) >= self._max_traces:
            self._traces.pop(0)
        self._traces.append(trace)

    def get_recent(self, n: int = 10) -> list[Trace]:
        """Return the n most recent traces."""
        return self._traces[-n:]

    def get_stats(self) -> dict:
        """Aggregate stats across all stored traces."""
        if not self._traces:
            return {"traces": 0}

        total_ms      = [t.total_ms for t in self._traces]
        total_tokens  = [t.total_tokens for t in self._traces]
        total_cost    = sum(t.estimated_cost_usd for t in self._traces)

        return {
            "traces": len(self._traces),
            "avg_latency_ms": round(sum(total_ms) / len(total_ms), 1),
            "p95_latency_ms": round(sorted(total_ms)[int(len(total_ms) * 0.95)], 1),
            "avg_tokens": round(sum(total_tokens) / len(total_tokens), 1),
            "total_cost_usd": round(total_cost, 4),
        }

    def export_jsonl(self) -> str:
        """Export all traces as JSONL for external analysis or Langfuse import."""
        return "\n".join(json.dumps(t.to_dict()) for t in self._traces)


# ── Prompt Version Registry ───────────────────────────────────────────────────
class PromptVersionRegistry:
    """
    Track which prompt version produced each response.

    Prompt versioning is critical for debugging regressions:
    "Did quality drop because of the prompt change, or the model change?"

    Usage:
        registry = PromptVersionRegistry()
        registry.register("rag_system_v1", "You are a helpful assistant...")
        registry.register("rag_system_v2", "You are a precise assistant...")

        active = registry.get_active("rag_system")
        # use active.template in your API calls
        # log active.version_id with each span for traceability

    STUDENT TODO:
        - Persist versions to a database.
        - Add A/B testing: randomly route traffic to v1 vs v2 and compare eval scores.
        - Add a changelog: record why each version was changed.
    """

    def __init__(self):
        self._versions: dict[str, list[dict]] = {}

    def register(self, name: str, template: str, notes: str = "") -> str:
        """
        Register a new version of a named prompt.
        Returns the version ID (name_v{n}).
        """
        if name not in self._versions:
            self._versions[name] = []
        version_num = len(self._versions[name]) + 1
        version_id  = f"{name}_v{version_num}"
        self._versions[name].append({
            "version_id": version_id,
            "template": template,
            "notes": notes,
            "registered_at": time.time(),
        })
        logger.info("PromptVersionRegistry: registered %s", version_id)
        return version_id

    def get_active(self, name: str) -> Optional[dict]:
        """Return the latest (active) version of a named prompt."""
        versions = self._versions.get(name, [])
        return versions[-1] if versions else None

    def get_version(self, version_id: str) -> Optional[dict]:
        """Retrieve a specific version by ID."""
        for versions in self._versions.values():
            for v in versions:
                if v["version_id"] == version_id:
                    return v
        return None

    def list_versions(self, name: str) -> list[dict]:
        """List all versions of a named prompt."""
        return self._versions.get(name, [])


# ── Global singletons ─────────────────────────────────────────────────────────
global_tracer   = RequestTracer()
prompt_registry = PromptVersionRegistry()
