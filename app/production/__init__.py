"""
production/ — Production-grade patterns for AI systems.

Raw LLM APIs are not production-ready on their own. This module covers
the engineering patterns that separate a demo from a real product:

  cache.py         — Semantic caching: serve repeated queries from cache
  rate_limiter.py  — Token bucket rate limiting per user/plan
  resilience.py    — Retry, fallback, and circuit breaker patterns
  cost_tracker.py  — Real-time cost tracking and budget alerts
  multi_tenant.py  — User-level isolation and data segregation

These patterns directly address common student questions:
  Q: "Are production patterns like async processing, caching, and rate limiting taught?"
  A: Yes — all covered in this module.

  Q: "Do you cover cost optimization for LLM apps?"
  A: Yes — cost_tracker.py with strategies for optimisation.

  Q: "Are reliability patterns like retries, fallbacks, circuit breakers covered?"
  A: Yes — resilience.py covers all three with runnable implementations.

  Q: "Do you cover multi-tenant AI architecture and user-level isolation?"
  A: Yes — multi_tenant.py covers the core patterns.

STUDENT TODO:
  - Replace the in-memory stores with Redis for production persistence.
  - Add Prometheus metrics: expose cache hit rate, rate limit rejections, costs.
  - Wire the circuit breaker around every external API call.
"""

from app.production.cache import SemanticCache
from app.production.rate_limiter import TokenBucketRateLimiter, RateLimitExceeded
from app.production.resilience import with_retry, CircuitBreaker, FallbackChain
from app.production.cost_tracker import CostTracker, CostBudget
from app.production.multi_tenant import TenantContext, TenantIsolationMiddleware

__all__ = [
    "SemanticCache",
    "TokenBucketRateLimiter", "RateLimitExceeded",
    "with_retry", "CircuitBreaker", "FallbackChain",
    "CostTracker", "CostBudget",
    "TenantContext", "TenantIsolationMiddleware",
]
