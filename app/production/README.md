# Production Module — `app/production/`

Five patterns that separate a working prototype from a production AI system. Each file is a self-contained, reusable primitive — wrap any external API call with these.

**Module:** 5

---

## Why These Patterns Matter

LLM APIs are expensive, slow, and unreliable. Production systems must handle:
- **Cost** — LLM calls at scale are budget-breaking without caching and tracking
- **Latency** — 1-3 second LLM calls multiply badly under concurrent traffic
- **Failures** — APIs go down, rate limits hit, networks blip
- **Multi-tenancy** — different users must never see each other's data

---

## File Map

| File | Pattern | Impact |
|------|---------|--------|
| `cache.py` | Semantic caching | 30-50% cost and latency reduction |
| `rate_limiter.py` | Token bucket rate limiting | Prevent abuse, enforce per-plan quotas |
| `resilience.py` | Retry + circuit breaker + fallback | Survive transient API failures gracefully |
| `cost_tracker.py` | Real-time cost tracking | Budget enforcement and cost visibility |
| `multi_tenant.py` | User-level data isolation | Required for any multi-user deployment |

---

## Semantic Cache (`cache.py`)

Regular caches miss on "Explain RAG" vs "What is RAG?" — different strings. A semantic cache embeds the query and uses cosine similarity to detect *semantically equivalent* questions.

```
Incoming query → embed → compare to cached embeddings
  similarity ≥ threshold → return cached answer  (~5ms, $0.000)
  similarity < threshold → call LLM              (~1500ms, $0.005)
```

**How to use:**
```python
from app.production.cache import SemanticCache
from app.rag.embeddings import get_embedding_provider

cache = SemanticCache(similarity_threshold=0.92)
embed = get_embedding_provider()

vec = embed.embed(question)
hit = cache.get(question, vec)
if hit:
    return hit                         # instant, free

result = rag_pipeline.query(question)
cache.put(question, vec, result.response)
```

**Key setting:** `similarity_threshold` — higher means stricter matching. Start at 0.92, tune down if you see incorrect cache hits.

### Student TODO
- Back the cache with Redis for persistence across restarts
- Add a TTL so stale answers expire automatically
- Track hit rate per endpoint and report in `/harness/tracer/stats`

---

## Rate Limiter (`rate_limiter.py`)

Token bucket algorithm: each user gets a bucket that refills at a fixed rate. Bursts are allowed up to the bucket capacity; sustained traffic is throttled to the refill rate.

Per-plan quotas:

| Plan | Requests/min | Burst capacity |
|------|-------------|----------------|
| free | 10 | 20 |
| pro | 60 | 120 |
| enterprise | 300 | 600 |

**How to use:**
```python
from app.production.rate_limiter import RateLimiter

limiter = RateLimiter()
if not limiter.allow(user_id=request.user_id, plan="pro"):
    raise HTTPException(429, "Rate limit exceeded")
```

### Student TODO
- Persist rate limiter state to Redis so limits survive server restarts
- Add per-endpoint limits (e.g. `/rag/query` vs `/voice/process` have different costs)
- Return `Retry-After` headers so clients know when to try again

---

## Resilience (`resilience.py`)

Three primitives. Combine them for maximum resilience.

### `@with_retry` — Exponential backoff
```python
@with_retry(max_attempts=3, base_delay_s=1.0, backoff_factor=2.0)
def call_llm(prompt):
    return client.messages.create(...)
# Attempt 1 fails → wait 1s → Attempt 2 fails → wait 2s → Attempt 3 or raise
```

### `CircuitBreaker` — Stop hammering failing services
Three states: `CLOSED` (normal) → `OPEN` (blocking, after N failures) → `HALF_OPEN` (probing recovery)

```python
breaker = CircuitBreaker(name="anthropic", failure_threshold=5, recovery_timeout_s=30)

@breaker.call
def call_llm(prompt):
    return client.messages.create(...)
```

When the circuit opens, calls fail immediately without touching the network. After `recovery_timeout_s`, one probe is allowed through. If it succeeds, the circuit closes.

### `FallbackChain` — Graceful degradation
```python
chain = FallbackChain([
    lambda q: claude_opus_call(q),      # primary: best quality
    lambda q: claude_haiku_call(q),     # fallback: cheaper, faster
    lambda q: "Service temporarily unavailable.",  # safe default
])
result = chain.execute(query)
```

### Student TODO
- Add jitter to retry backoff to avoid thundering herd (many clients retrying simultaneously)
- Set different thresholds per service (Anthropic vs Deepgram vs ElevenLabs)
- Log every circuit break event to your observability system

---

## Cost Tracker (`cost_tracker.py`)

Tracks token usage and estimates dollar cost in real time. Fires budget alerts when spend exceeds configurable thresholds.

```python
from app.production.cost_tracker import CostTracker

tracker = CostTracker(budget_usd=10.0)
tracker.record(
    model="claude-3-5-sonnet-20241022",
    input_tokens=1200,
    output_tokens=450,
    endpoint="/rag/query",
)

print(tracker.stats())
# {"total_cost_usd": 0.042, "budget_remaining_usd": 9.958, "calls_today": 12, ...}
```

Includes optimisation tips: which queries are most expensive, where caching would have the biggest impact, and model downgrades that would preserve quality.

---

## Multi-Tenancy (`multi_tenant.py`)

User-level data isolation so tenant A never sees tenant B's data. Uses Python context variables to propagate `tenant_id` through the request stack without passing it explicitly to every function.

```
Incoming request
    │ middleware extracts tenant_id from Authorization header
    ▼
FastAPI context variable: tenant_id = "tenant_abc"
    │
    ▼
VectorStore.search() → automatically scopes to tenant_abc's namespace
RAGPipeline.ingest() → stores with tenant_abc prefix
```

**Isolation layers:**
1. **In-memory store:** separate dict namespace per tenant
2. **pgvector store:** `WHERE tenant_id = $1` on every query
3. **Request tracing:** every trace tagged with `tenant_id`

### Student TODO
- Add tenant-scoped rate limiting: each tenant has its own quota
- Add tenant-scoped cost tracking: generate per-tenant bills
- Implement tenant provisioning: create/delete tenant namespaces via API
