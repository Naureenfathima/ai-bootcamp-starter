"""
production/multi_tenant.py — Multi-tenant AI architecture and user-level isolation.

Multi-tenancy means multiple customers share the same infrastructure while
remaining completely isolated from each other's data and activity.

Why this matters for AI:
  - A tenant's documents must never appear in another tenant's RAG results
  - A tenant's conversation history must be invisible to other tenants
  - A tenant's rate limits and costs must be tracked independently
  - Compliance (GDPR, SOC 2, HIPAA) often requires strict data segregation

Isolation strategies:
  ┌─────────────────┬────────────────────────────────────────────────────┐
  │ Strategy        │ How it works                                       │
  ├─────────────────┼────────────────────────────────────────────────────┤
  │ Row-level       │ Filter every DB query by tenant_id (cheapest)      │
  │ Schema-level    │ Each tenant gets their own DB schema (medium cost)  │
  │ Database-level  │ Separate DB per tenant (most isolated, expensive)   │
  │ Namespace-level │ Separate vector store namespace per tenant          │
  └─────────────────┴────────────────────────────────────────────────────┘

This module implements row-level isolation using a tenant context that
is automatically injected into every RAG and agent operation.

STUDENT TODO:
  - Apply TenantContext.require() at the start of every API handler.
  - Extend InMemoryVectorStore to filter by tenant_id.
  - Add an audit log: who queried what and when (SOC 2 compliance).
  - Add tenant provisioning: create a new tenant with a single API call.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable — carries the current tenant ID through the call stack
# This is safe in async FastAPI applications (each request gets its own context)
_current_tenant: ContextVar[Optional[str]] = ContextVar("current_tenant", default=None)


@dataclass
class Tenant:
    """Represents a tenant (customer/organisation) in the system."""
    id: str
    name: str
    plan: str = "free"               # "free" | "pro" | "enterprise"
    data_region: str = "us-east-1"  # for data residency compliance
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class TenantRegistry:
    """In-memory tenant registry. Replace with a database in production."""

    def __init__(self):
        self._tenants: dict[str, Tenant] = {}

    def create(self, name: str, plan: str = "free") -> Tenant:
        tenant = Tenant(id=str(uuid.uuid4())[:8], name=name, plan=plan)
        self._tenants[tenant.id] = tenant
        logger.info("Created tenant: %s (%s, plan=%s)", tenant.name, tenant.id, plan)
        return tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    def exists(self, tenant_id: str) -> bool:
        return tenant_id in self._tenants

    def list_all(self) -> list[Tenant]:
        return list(self._tenants.values())


# Global registry
tenant_registry = TenantRegistry()


class TenantContext:
    """
    Manages the current tenant context using Python's contextvars.

    contextvars are safe in async code — each FastAPI request gets its own
    copy, so there is no cross-request contamination.

    Usage in a FastAPI route:
        @router.post("/rag/query")
        async def query(
            request: QueryRequest,
            x_tenant_id: str = Header(...),   # tenant sends their ID in headers
        ):
            with TenantContext.set(x_tenant_id):
                result = rag_pipeline.query(request.question)
            return result

    Usage inside a service (reads from context):
        tenant_id = TenantContext.get()
        results = vector_store.search(query, tenant_id=tenant_id)
    """

    @staticmethod
    def set(tenant_id: str):
        """Context manager: set the tenant for the duration of a block."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            token = _current_tenant.set(tenant_id)
            try:
                yield tenant_id
            finally:
                _current_tenant.reset(token)

        return _ctx()

    @staticmethod
    def get() -> Optional[str]:
        """Return the current tenant ID, or None if not set."""
        return _current_tenant.get()

    @staticmethod
    def require() -> str:
        """
        Return the current tenant ID, raising if not set.
        Use this in services that must always run in tenant context.
        """
        tenant_id = _current_tenant.get()
        if tenant_id is None:
            raise RuntimeError(
                "No tenant context set. All API calls must include X-Tenant-ID header. "
                "Use TenantContext.set(tenant_id) before calling this service."
            )
        return tenant_id


class TenantIsolationMiddleware:
    """
    FastAPI middleware that extracts the tenant ID from request headers
    and sets it in the context for the duration of the request.

    Add to your FastAPI app:
        from app.production.multi_tenant import TenantIsolationMiddleware
        app.add_middleware(TenantIsolationMiddleware)

    Request header required:
        X-Tenant-ID: <tenant_id>

    Tenants that don't send this header receive a 401 (or a default tenant
    if you configure ALLOW_ANONYMOUS=True for single-tenant mode).
    """

    HEADER_NAME = "X-Tenant-ID"
    ALLOW_ANONYMOUS = True    # set False in strict multi-tenant mode
    DEFAULT_TENANT  = "default"

    def __init__(self, app, allow_anonymous: bool = True):
        self._app = app
        self._allow_anonymous = allow_anonymous

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            tenant_id = headers.get(
                self.HEADER_NAME.lower().encode(),
                self.DEFAULT_TENANT.encode() if self._allow_anonymous else None,
            )

            if tenant_id is None:
                # Reject unauthenticated requests in strict mode
                from starlette.responses import JSONResponse
                response = JSONResponse(
                    {"error": f"Missing {self.HEADER_NAME} header"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return

            tenant_id = tenant_id.decode()
            token = _current_tenant.set(tenant_id)
            try:
                await self._app(scope, receive, send)
            finally:
                _current_tenant.reset(token)
        else:
            await self._app(scope, receive, send)


class TenantScopedVectorStore:
    """
    A wrapper around InMemoryVectorStore that enforces tenant isolation.

    Every add() and search() call is automatically scoped to the current tenant.

    STUDENT TODO: apply the same pattern to your production PgVectorStore
    by adding a tenant_id column and filtering all queries by it.

    Example PostgreSQL query with tenant isolation:
        SELECT content, 1 - (embedding <=> %s) AS score
        FROM embeddings
        WHERE tenant_id = %s      ← tenant isolation here
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    def __init__(self):
        from app.rag.retrieval import InMemoryVectorStore
        from app.rag.ingestion import Chunk
        # Store per tenant: {tenant_id: [(chunk, embedding), ...]}
        self._stores: dict[str, list] = {}

    def _get_store(self, tenant_id: str) -> list:
        if tenant_id not in self._stores:
            self._stores[tenant_id] = []
        return self._stores[tenant_id]

    def add(self, chunk, embedding: list[float]) -> None:
        tenant_id = TenantContext.require()
        self._get_store(tenant_id).append((chunk, embedding))
        logger.debug("TenantVectorStore: added chunk for tenant %s", tenant_id)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list:
        from app.rag.retrieval import cosine_similarity, SearchResult
        tenant_id = TenantContext.require()
        store = self._get_store(tenant_id)

        results = []
        for chunk, emb in store:
            score = cosine_similarity(query_embedding, emb)
            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def count(self, tenant_id: str | None = None) -> int:
        if tenant_id:
            return len(self._get_store(tenant_id))
        return sum(len(s) for s in self._stores.values())

    def stats(self) -> dict:
        return {
            "tenants": len(self._stores),
            "chunks_by_tenant": {tid: len(store) for tid, store in self._stores.items()},
        }
