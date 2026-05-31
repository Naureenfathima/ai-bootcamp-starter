# MCP Module — `app/mcp/`

Model Context Protocol (MCP) defines a standard request/response interface for AI components. Instead of every integration being custom, each service exposes the same envelope — method name, params, result — so components can call each other predictably.

**Module:** 5

---

## The Problem MCP Solves

Without a standard protocol, wiring your RAG pipeline to an agent, a voice assistant, and an external tool requires custom glue code for every pair of components. With MCP:

- Every component exposes a `dispatch(MCPRequest) → MCPResponse` interface
- Callers only need to know the method name and params
- Adding a new component means registering handlers — not changing callers

---

## How It Works

```python
from app.mcp.server import mcp_server, MCPRequest

# Register a handler (once, at startup)
@mcp_server.handler("rag/query")
def handle_rag_query(params: dict) -> dict:
    result = rag_pipeline.query(params["question"])
    return {"answer": result.response, "sources": result.sources}

# Call from anywhere
response = mcp_server.dispatch(MCPRequest(
    method="rag/query",
    params={"question": "What is our refund policy?"}
))

if response.ok:
    print(response.result["answer"])
else:
    print(response.error)
```

Every response includes `request_id` (for tracing), `result`, `error`, and `latency_ms`.

---

## Built-in Handlers

| Method | Description |
|--------|-------------|
| `system/ping` | Liveness check |
| `system/list_methods` | Returns all registered method names |

---

## Registering Your Modules as MCP Methods

The exercise for this module is wiring your existing RAG and Agent modules as MCP handlers:

```python
# In app/mcp/server.py or a new app/mcp/handlers.py:

@mcp_server.handler("rag/ingest")
def handle_ingest(params):
    n = rag_pipeline.ingest_text(params["text"], source=params.get("source", "unknown"))
    return {"chunks_added": n}

@mcp_server.handler("agent/run")
def handle_agent(params):
    result = agent.run(params["message"])
    return {"response": result.response, "steps": result.steps_taken}
```

Once registered, any component (voice pipeline, another agent, an external client) can call these methods through the same `dispatch()` interface without importing the underlying classes.

---

## Student TODOs

1. **Register RAG and Agent as MCP methods** — `rag/ingest`, `rag/query`, `agent/run`, `agent/reset`
2. **Add authentication** — each `MCPRequest` should carry an auth token; validate it in `dispatch()`
3. **Add request validation** — reject requests with missing required params before calling the handler
4. **Rate limit MCP calls** — connect to `app/production/rate_limiter.py`
5. **Read the MCP spec** — https://modelcontextprotocol.io — this stub implements the envelope pattern; the full spec covers resources, tools, and sampling

---

## Connection to the Bigger Picture

MCP is why production AI systems are built as composable services rather than monoliths. When your RAG, Agent, Voice, and Evaluation modules each speak the same protocol, you can:

- Swap out one component without touching the others
- Run components on separate servers and call them over HTTP
- Expose your system as a set of capabilities that other AI agents can discover and call
