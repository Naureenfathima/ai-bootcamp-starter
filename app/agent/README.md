# Agent Module — `app/agent/`

An autonomous agent that uses tools to complete multi-step tasks. Rather than answering in one shot, the agent calls tools, observes results, and keeps reasoning until it has a complete answer.

**Module:** 4  
**API prefix:** `/agent`

---

## How the Agentic Loop Works

```
User message
    │
    ▼
[Claude + tools] ──→ stop_reason == "end_turn" ──→ Final answer
                └──→ stop_reason == "tool_use"  ──→ Execute tool ──→ [back to Claude]
```

Each iteration is one "step". The loop continues until Claude produces a text response or `MAX_AGENT_STEPS` is reached. The step limit is a hard safety ceiling — never build an unbounded LLM loop in production.

---

## File Map

| File | What it does |
|------|-------------|
| `agent.py` | The `Agent` class and agentic loop |
| `tools.py` | Tool definitions: calculator, current time, web search |
| `memory.py` | Four memory types for managing context across turns |

---

## Tools (`tools.py`)

Each tool has a name, description, and JSON schema that Claude uses to decide when and how to call it. The agent ships with three tools:

| Tool | What it does |
|------|-------------|
| `calculator` | Evaluates math expressions — never hallucinate arithmetic |
| `get_current_time` | Returns current time and timezone |
| `web_search` | Stub — implement with Tavily or DuckDuckGo |

**Test it:**
```
POST /agent/run
{"message": "What is sqrt(144) + the current time?"}
```
The agent will call `calculator`, then `get_current_time`, then synthesise both results.

**Adding a new tool** means adding one `Tool` object with a `name`, `description`, `parameters` schema, and an `execute` function. The agent picks it up automatically.

---

## Memory (`memory.py`)

The most important architectural concept in this module: agents need different *kinds* of memory for different purposes.

```
┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Working      │  │  Episodic     │  │  Semantic       │  │  Persistent      │
│  Memory       │  │  Memory       │  │  Memory         │  │  Store           │
│               │  │               │  │                 │  │                  │
│  dict scratchpad  conversation    │  long-term facts   │  cross-session     │
│  per-task     │  │  per-session  │  │  vector-backed   │  │  Redis / SQLite  │
│  (implemented)│  │  (implemented)│  │  (TODO)         │  │  (TODO)          │
└───────────────┘  └───────────────┘  └─────────────────┘  └──────────────────┘
```

### WorkingMemory
Key-value scratchpad for the current task. Cleared on each new request. Use it to store intermediate results the agent needs mid-task.

### EpisodicMemory
Conversation history for the session. Two strategies:
- **Sliding window** (default): drop oldest messages when full — simple, zero cost
- **Summarisation** (advanced): compress oldest messages with Claude — no context loss, small API cost

**Your task:** implement `_summarise_oldest()` in `memory.py` using the Anthropic client. The full code is in the docstring — it's a guided implementation.

### SemanticMemory
Long-term facts that persist across requests. Intended to be backed by `RAGPipeline` from `app/rag/pipeline.py`. Currently a sorted in-memory list.

**Your task:** connect `remember()` and `recall()` to `RAGPipeline` so the agent can actually store and retrieve facts using vector search.

### PersistentStore
Cross-session storage so conversation history survives server restarts. The file contains Redis and SQLite sketches with all the code — implementing them is a guided exercise.

---

## Key Settings (`.env`)

```
MAX_AGENT_STEPS=10      # safety ceiling on tool-call iterations
AGENT_TEMPERATURE=0.2   # low = more deterministic tool decisions
```

---

## Student TODOs

1. Implement `EpisodicMemory._summarise_oldest()` — code template is in the docstring
2. Connect `SemanticMemory` to `RAGPipeline` — two methods to implement
3. Implement `PersistentStore` with Redis or SQLite — sketches in the docstring
4. Add a new tool (e.g. a database lookup, a file reader, a calculator upgrade)
5. Add chain-of-thought prompting: ask the agent to plan before acting
