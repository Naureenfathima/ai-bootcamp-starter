"""
production/resilience.py — Retry, fallback, and circuit breaker patterns.

LLM APIs fail. Networks are unreliable. Models get overloaded.
Production systems must handle these failures gracefully:

  Retry:           Automatically retry transient failures with exponential backoff
  Fallback:        If the primary system fails, use a backup (cheaper model, cache)
  Circuit Breaker: Stop hammering a failing service — "open the circuit" after
                   N failures and let it recover before retrying

These three patterns together give you:
  - Resilience against transient errors (retry)
  - Graceful degradation (fallback)
  - Protection against cascading failures (circuit breaker)

STUDENT TODO:
  - Add jitter to exponential backoff to avoid thundering herd.
  - Log every retry and circuit break event to your observability system.
  - Set different thresholds per service (Anthropic vs Deepgram vs ElevenLabs).
  - Add a half-open state: probe the service before fully re-closing the circuit.
"""

import logging
import time
import functools
from typing import Callable, TypeVar, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ── Retry Decorator ───────────────────────────────────────────────────────────
def with_retry(
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_attempts:          Total attempts (1 = no retry).
        base_delay_s:          Wait time before first retry (seconds).
        backoff_factor:        Multiply delay by this after each failure.
        retryable_exceptions:  Only retry on these exception types.

    Usage:
        @with_retry(max_attempts=3, base_delay_s=1.0)
        def call_llm(prompt):
            return client.messages.create(...)

    Backoff timeline (base=1s, factor=2):
        Attempt 1 fails → wait 1s
        Attempt 2 fails → wait 2s
        Attempt 3 fails → raise exception
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay_s
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "Retry exhausted: %s failed after %d attempts: %s",
                            func.__name__, max_attempts, exc
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                        attempt, max_attempts, func.__name__, exc, delay
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise last_exception  # should never reach here
        return wrapper
    return decorator


# ── Circuit Breaker ───────────────────────────────────────────────────────────
class CircuitState(Enum):
    CLOSED   = "closed"    # normal operation
    OPEN     = "open"      # failures exceeded threshold — blocking calls
    HALF_OPEN = "half_open" # testing if service has recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for protecting external service calls.

    States:
      CLOSED    → calls pass through normally
      OPEN      → calls immediately fail (no network traffic) after N failures
      HALF_OPEN → one probe call allowed; success closes, failure re-opens

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30)

        @breaker.call
        def call_llm(prompt):
            return client.messages.create(...)

    Timeline:
        Call 1 OK → CLOSED
        Call 2 OK → CLOSED
        Call 3 FAIL → CLOSED (failure_count=1)
        ...
        Call 7 FAIL → OPEN (failure_count=5, blocked for 30s)
        [30 seconds pass]
        HALF_OPEN → probe call allowed
        Probe succeeds → CLOSED, reset failure count
        Probe fails → OPEN again
    """

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _total_calls: int = field(default=0, init=False)
    _total_rejections: int = field(default=0, init=False)

    def _should_attempt(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN (probe allowed)", self.name)
                return True
            return False
        if self._state == CircuitState.HALF_OPEN:
            return True
        return False

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info("CircuitBreaker[%s]: HALF_OPEN → CLOSED (service recovered)", self.name)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count > 0:
                self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — reopen
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning("CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed)", self.name)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "CircuitBreaker[%s]: CLOSED → OPEN (failures=%d, threshold=%d)",
                self.name, self._failure_count, self.failure_threshold
            )

    def call(self, func: Callable) -> Callable:
        """Decorator: wrap a function with circuit breaker protection."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._total_calls += 1
            if not self._should_attempt():
                self._total_rejections += 1
                raise RuntimeError(
                    f"Circuit breaker [{self.name}] is OPEN. "
                    f"Service unavailable. Retry after {self.recovery_timeout_s}s."
                )
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as exc:
                self._on_failure()
                raise
        return wrapper

    @property
    def state(self) -> str:
        return self._state.value

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "total_calls": self._total_calls,
            "total_rejections": self._total_rejections,
            "rejection_rate": round(
                self._total_rejections / max(self._total_calls, 1), 3
            ),
        }


# ── Fallback Chain ────────────────────────────────────────────────────────────
class FallbackChain:
    """
    Execute a list of functions in order, returning the first successful result.

    Use this to implement graceful degradation:
      1. Try the primary (best, most expensive) option
      2. If it fails, try the secondary (cheaper, faster) option
      3. If that fails too, return a hardcoded safe default

    Usage:
        chain = FallbackChain([
            lambda q: claude_opus_call(q),        # primary: best quality
            lambda q: claude_haiku_call(q),        # fallback: cheaper
            lambda q: "I'm currently unavailable. Please try again shortly.",  # safe default
        ])

        result = chain.execute(query)
    """

    def __init__(self, handlers: list[Callable]):
        self._handlers = handlers
        self._stats = {"attempts": 0, "primary_success": 0, "fallback_used": 0}

    def execute(self, *args, **kwargs) -> Any:
        """Try each handler in order; return the first successful result."""
        self._stats["attempts"] += 1
        last_exc = None

        for i, handler in enumerate(self._handlers):
            try:
                result = handler(*args, **kwargs)
                if i == 0:
                    self._stats["primary_success"] += 1
                else:
                    self._stats["fallback_used"] += 1
                    logger.info("FallbackChain: used handler %d/%d", i + 1, len(self._handlers))
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "FallbackChain: handler %d/%d failed: %s — trying next",
                    i + 1, len(self._handlers), exc
                )

        raise RuntimeError(
            f"All {len(self._handlers)} handlers failed. Last error: {last_exc}"
        )

    def stats(self) -> dict:
        return {
            **self._stats,
            "primary_success_rate": round(
                self._stats["primary_success"] / max(self._stats["attempts"], 1), 3
            ),
        }
