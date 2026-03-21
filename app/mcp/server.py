"""
mcp/server.py — Model Context Protocol (MCP) server stub.

MCP defines a standard protocol for communication between AI components.
Instead of ad-hoc integrations, each service exposes a well-defined
interface that other components can call predictably.

This file provides a minimal MCP-inspired request/response schema.
It is intentionally simple so you can extend it in Module 5.

STUDENT TODO:
  - Implement the full MCP specification: https://modelcontextprotocol.io
  - Add authentication: each MCP call should include an auth token.
  - Add request validation: reject malformed requests early.
  - Add rate limiting: prevent runaway loops from hammering external APIs.
  - Connect this to your RAG and Agent modules as MCP "resources" and "tools".
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── MCP Request / Response ─────────────────────────────────────────────────────
@dataclass
class MCPRequest:
    """
    A standardised request envelope for MCP calls.

    Fields:
        method:  The operation to perform, e.g. "rag/query", "agent/run".
        params:  A dict of parameters for the method.
        request_id: Auto-generated UUID for tracing.
    """
    method: str
    params: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MCPResponse:
    """
    A standardised response envelope for MCP calls.

    Fields:
        request_id: Echoes back the request's ID for correlation.
        result:     The method's return value (None on error).
        error:      Human-readable error message (None on success).
        latency_ms: How long the call took.
    """
    request_id: str
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


# ── MCP Handler Registry ──────────────────────────────────────────────────────
class MCPServer:
    """
    A simple MCP server that dispatches requests to registered handlers.

    Register handlers with the @server.handler("method/name") decorator.

    Usage:
        server = MCPServer()

        @server.handler("math/add")
        def handle_add(params):
            return params["a"] + params["b"]

        response = server.dispatch(MCPRequest(method="math/add", params={"a": 2, "b": 3}))
        print(response.result)  # 5
    """

    def __init__(self):
        self._handlers: dict[str, callable] = {}

    def handler(self, method: str):
        """Decorator to register a handler for an MCP method."""
        def decorator(func):
            self._handlers[method] = func
            logger.info("MCP: registered handler for '%s'", method)
            return func
        return decorator

    def dispatch(self, request: MCPRequest) -> MCPResponse:
        """
        Route an MCPRequest to the appropriate handler.

        Returns MCPResponse with result on success, or error on failure.
        """
        t0 = time.perf_counter()
        logger.info("MCP dispatch: method='%s' id=%s", request.method, request.request_id)

        handler = self._handlers.get(request.method)
        if handler is None:
            logger.warning("MCP: no handler for method '%s'", request.method)
            return MCPResponse(
                request_id=request.request_id,
                error=f"Unknown method: '{request.method}'. "
                      f"Available: {sorted(self._handlers.keys())}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )

        try:
            result = handler(request.params)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.info("MCP: '%s' succeeded in %sms", request.method, latency_ms)
            return MCPResponse(
                request_id=request.request_id,
                result=result,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.exception("MCP: '%s' failed: %s", request.method, exc)
            return MCPResponse(
                request_id=request.request_id,
                error=str(exc),
                latency_ms=latency_ms,
            )

    def list_methods(self) -> list[str]:
        return sorted(self._handlers.keys())


# ── Global MCP server instance ────────────────────────────────────────────────
mcp_server = MCPServer()


# ── Example handlers (remove / replace in production) ─────────────────────────
@mcp_server.handler("system/ping")
def handle_ping(params: dict) -> dict:
    """Simple liveness check."""
    return {"pong": True, "echo": params.get("message", "")}


@mcp_server.handler("system/list_methods")
def handle_list_methods(params: dict) -> list[str]:
    """Return all registered MCP methods."""
    return mcp_server.list_methods()
