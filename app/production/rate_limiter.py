"""
production/rate_limiter.py — Token bucket rate limiting per user/plan.

Rate limiting prevents:
  - Runaway agent loops from bankrupting you
  - Abuse by a single user degrading service for everyone
  - Exceeding provider API limits (Anthropic, Deepgram, etc.)

Token Bucket Algorithm:
  - Each user has a "bucket" of tokens
  - Tokens are added at a fixed rate (refill_rate tokens/second)
  - Each request consumes tokens proportional to its cost
  - If the bucket is empty → request is rejected (429 Too Many Requests)

Plan-based limits example:
  Free:       10 requests/min, 10k tokens/day
  Pro:        60 requests/min, 100k tokens/day
  Enterprise: 300 requests/min, unlimited tokens/day

STUDENT TODO:
  - Replace the in-memory store with Redis for distributed rate limiting.
  - Add token-based limits: count actual LLM tokens consumed, not just requests.
  - Add per-endpoint limits: /rag/query might have a tighter limit than /health.
  - Expose rate limit headers in API responses: X-RateLimit-Remaining, X-RateLimit-Reset.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a user has exceeded their rate limit."""
    def __init__(self, user_id: str, retry_after_s: float):
        self.user_id = user_id
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Rate limit exceeded for user '{user_id}'. "
            f"Retry after {retry_after_s:.1f}s."
        )


@dataclass
class Plan:
    """Rate limit configuration for a subscription plan."""
    name: str
    requests_per_minute: int
    tokens_per_day: int        # -1 = unlimited
    burst_multiplier: float = 1.5   # allow short bursts above the rate


# Built-in plans
FREE_PLAN       = Plan("free",       requests_per_minute=10,  tokens_per_day=10_000)
PRO_PLAN        = Plan("pro",        requests_per_minute=60,  tokens_per_day=100_000)
ENTERPRISE_PLAN = Plan("enterprise", requests_per_minute=300, tokens_per_day=-1)


@dataclass
class UserBucket:
    """Token bucket state for a single user."""
    user_id: str
    plan: Plan
    tokens: float = field(init=False)
    last_refill: float = field(default_factory=time.monotonic)
    daily_tokens_used: int = 0
    daily_reset_at: float = field(default_factory=lambda: time.time() + 86400)

    def __post_init__(self):
        self.tokens = float(self.plan.requests_per_minute)

    @property
    def refill_rate(self) -> float:
        """Tokens per second = requests_per_minute / 60."""
        return self.plan.requests_per_minute / 60.0

    @property
    def max_tokens(self) -> float:
        return self.plan.requests_per_minute * self.plan.burst_multiplier

    def refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.tokens + new_tokens, self.max_tokens)
        self.last_refill = now

        # Reset daily counter if the day has rolled over
        if time.time() > self.daily_reset_at:
            self.daily_tokens_used = 0
            self.daily_reset_at = time.time() + 86400

    def can_consume(self, cost: float = 1.0) -> bool:
        self.refill()
        return self.tokens >= cost

    def consume(self, cost: float = 1.0, llm_tokens: int = 0) -> None:
        self.tokens -= cost
        self.daily_tokens_used += llm_tokens

    def retry_after(self, cost: float = 1.0) -> float:
        """Return seconds until enough tokens are available."""
        deficit = cost - self.tokens
        return max(0.0, deficit / self.refill_rate)


class TokenBucketRateLimiter:
    """
    Per-user token bucket rate limiter.

    Usage in a FastAPI route:
        limiter = TokenBucketRateLimiter()

        @router.post("/rag/query")
        async def query(request: QueryRequest, user_id: str = Header(...)):
            limiter.check(user_id)         # raises RateLimitExceeded if over limit
            result = pipeline.query(...)
            limiter.record_usage(user_id, llm_tokens=result.input_tokens + result.output_tokens)
            return result

    To assign plans:
        limiter.set_plan("user_abc", PRO_PLAN)
        limiter.set_plan("enterprise_user", ENTERPRISE_PLAN)
    """

    def __init__(self, default_plan: Plan = FREE_PLAN):
        self._buckets: dict[str, UserBucket] = {}
        self._default_plan = default_plan
        self._total_checks = 0
        self._total_rejections = 0

    def _get_bucket(self, user_id: str) -> UserBucket:
        if user_id not in self._buckets:
            self._buckets[user_id] = UserBucket(user_id=user_id, plan=self._default_plan)
        return self._buckets[user_id]

    def set_plan(self, user_id: str, plan: Plan) -> None:
        """Assign a subscription plan to a user."""
        self._buckets[user_id] = UserBucket(user_id=user_id, plan=plan)
        logger.info("RateLimiter: %s assigned plan '%s'", user_id, plan.name)

    def check(self, user_id: str, cost: float = 1.0) -> None:
        """
        Check if a user is within their rate limit.

        Raises RateLimitExceeded if over the limit.
        Call this at the start of every API handler.
        """
        self._total_checks += 1
        bucket = self._get_bucket(user_id)

        # Check daily token limit
        if (bucket.plan.tokens_per_day > 0
                and bucket.daily_tokens_used >= bucket.plan.tokens_per_day):
            self._total_rejections += 1
            raise RateLimitExceeded(user_id, retry_after_s=bucket.retry_after(cost))

        if not bucket.can_consume(cost):
            self._total_rejections += 1
            retry_after = bucket.retry_after(cost)
            logger.warning("RateLimit: %s exceeded (retry in %.1fs)", user_id, retry_after)
            raise RateLimitExceeded(user_id, retry_after_s=retry_after)

        bucket.consume(cost)

    def record_usage(self, user_id: str, llm_tokens: int) -> None:
        """Record actual LLM token usage for daily limit tracking."""
        bucket = self._get_bucket(user_id)
        bucket.daily_tokens_used += llm_tokens

    def user_stats(self, user_id: str) -> dict:
        """Return the current rate limit state for a user."""
        bucket = self._get_bucket(user_id)
        bucket.refill()
        return {
            "user_id": user_id,
            "plan": bucket.plan.name,
            "tokens_available": round(bucket.tokens, 2),
            "max_tokens": bucket.max_tokens,
            "requests_per_minute": bucket.plan.requests_per_minute,
            "daily_tokens_used": bucket.daily_tokens_used,
            "daily_tokens_limit": bucket.plan.tokens_per_day,
        }

    def global_stats(self) -> dict:
        return {
            "total_users": len(self._buckets),
            "total_checks": self._total_checks,
            "total_rejections": self._total_rejections,
            "rejection_rate": round(
                self._total_rejections / max(self._total_checks, 1), 3
            ),
        }
