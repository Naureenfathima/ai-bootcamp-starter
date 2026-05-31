# Repository Handbook
### AI Bootcamp Starter — BlockseBlock AI Systems Engineering

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [How to Read This Handbook](#2-how-to-read-this-handbook)
3. [Repository Map](#3-repository-map)
4. [System Architecture](#4-system-architecture)
5. [Getting Started](#5-getting-started)
6. [Configuration Reference](#6-configuration-reference)
7. [Module Guide](#7-module-guide)
   - [LLM Provider Abstraction](#71-llm-provider-abstraction)
   - [RAG — Retrieval-Augmented Generation](#72-rag--retrieval-augmented-generation)
   - [Anti-RAG — When Not to Use RAG](#73-anti-rag--when-not-to-use-rag)
   - [Voice AI](#74-voice-ai)
   - [Autonomous Agents](#75-autonomous-agents)
   - [Agent Harness](#76-agent-harness)
   - [Evaluation](#77-evaluation)
   - [Production Patterns](#78-production-patterns)
   - [Model Context Protocol (MCP)](#79-model-context-protocol-mcp)
8. [API Reference](#8-api-reference)
9. [LLM Provider Switching](#9-llm-provider-switching)
10. [Testing](#10-testing)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Docker and Deployment](#12-docker-and-deployment)
13. [Student Task Index](#13-student-task-index)
14. [Glossary](#14-glossary)

---

## 1. What This Project Is

This is the **hands-on project backbone** for the BlockseBlock AI Systems Engineering Bootcamp. Every module of the course adds a working layer to this codebase until you have a deployed, production-grade AI system accessible at a public URL.

**What you build by the end of the course:**

- A knowledge base that answers questions over your own documents (RAG)
- A voice assistant that listens, thinks, and speaks (Voice AI)
- An autonomous agent that uses tools and remembers context across conversations (Agents)
- Evaluation infrastructure that tells you whether your system is actually working (Harness + Eval)
- Production safeguards: rate limiting, caching, resilience, multi-tenancy

**Who this is for:**

| Role | How you'll use this |
|------|-------------------|
| **Bootcamp student** | Work through modules, complete TODOs, build your capstone |
| **Community member (technical)** | Explore patterns, extend modules, understand production AI architecture |
| **Community member (non-technical)** | Run the demo, understand what each system does, follow the API examples |
| **Instructor / TA** | Add domain tasks, update seeds, extend the evaluation suite |

---

## 2. How to Read This Handbook

**If you are non-technical:** Read sections 1, 3, 4, 8, and 14. You will understand what every part of the system does and be able to run the live demos without writing any code.

**If you are a first-time contributor:** Work through sections 5, 6, and 7 in order. Each module section explains the concept before the code.

**If you are looking for something specific:** Use the Table of Contents. Every section is self-contained.

**Conventions used in this document:**

```
Code blocks contain commands to run or code to read.
```

> **Note** blocks highlight important information.

> **Concept** blocks explain an idea before the implementation.

---

## 3. Repository Map

```
ai-bootcamp-starter/
│
├── app/                        Core application code
│   ├── main.py                 FastAPI entry point — wires all routers together
│   ├── config.py               All settings loaded from .env (single source of truth)
│   ├── llm.py                  Unified LLM client — swap providers without changing code
│   │
│   ├── rag/                    Module 3 — Retrieval-Augmented Generation
│   │   ├── ingestion.py        Text → chunks
│   │   ├── embeddings.py       Chunks → vectors (OpenAI or local sentence-transformers)
│   │   ├── retrieval.py        Vector store: in-memory or pgvector
│   │   ├── pipeline.py         End-to-end RAGPipeline class
│   │   ├── chunking.py         Four chunking strategies
│   │   ├── metadata.py         Metadata-aware retrieval
│   │   ├── page_index.py       Two-tier hierarchical index
│   │   ├── agentic_rag.py      Iterative retrieve-reflect-rewrite loop
│   │   ├── corrective_rag.py   Grade docs, fall back to web if poor quality
│   │   └── graph_rag.py        Entity graph + community summaries
│   │
│   ├── anti_rag/               Module 3 — Alternatives to RAG
│   │   ├── kag.py              Knowledge graph traversal
│   │   ├── cag.py              Full context window + prompt caching
│   │   ├── structured_knowledge.py  Text-to-SQL
│   │   ├── fine_tuning.py      Fine-tuning concepts and decision guide
│   │   └── router.py           Hybrid router: picks the right strategy per query
│   │
│   ├── voice/                  Module 4 — Voice AI pipeline
│   │   ├── stt.py              Speech-to-text via Deepgram
│   │   ├── tts.py              Text-to-speech via ElevenLabs
│   │   └── pipeline.py         Orchestrates STT → LLM → TTS
│   │
│   ├── agent/                  Module 4 — Autonomous agents
│   │   ├── agent.py            Agentic loop: LLM + tools + memory
│   │   ├── tools.py            Tool definitions (calculator, time, web search stub)
│   │   └── memory.py           Four memory types: Working, Episodic, Semantic, Persistent
│   │
│   ├── harness/                Module 5 — Agent test harness
│   │   ├── tasks.py            Task and TaskSuite definitions
│   │   ├── executor.py         Runs agent, records full trajectory
│   │   ├── scorer.py           Three-dimension scoring: outcome, tool use, efficiency
│   │   └── runner.py           Runs full suites, returns GREEN/YELLOW/RED report
│   │
│   ├── eval/                   Module 5 — RAG evaluation
│   │   ├── evaluator.py        LLM-as-judge for RAG responses
│   │   ├── metrics.py          Faithfulness, relevance, recall, completeness scores
│   │   └── tracer.py           Request tracing and prompt version registry
│   │
│   ├── production/             Module 5 — Production patterns
│   │   ├── cache.py            Semantic cache (skip LLM for near-duplicate queries)
│   │   ├── rate_limiter.py     Token-bucket rate limiter per tenant/plan
│   │   ├── resilience.py       Retry, circuit breaker, fallback chain
│   │   ├── cost_tracker.py     Per-request token cost tracking
│   │   └── multi_tenant.py     Tenant isolation and quota enforcement
│   │
│   ├── mcp/                    Module 5 — Model Context Protocol
│   │   └── server.py           MCP handler registration
│   │
│   └── api/                    FastAPI route handlers (thin layer over modules)
│       ├── health.py           GET /health
│       ├── rag.py              /rag/* routes
│       ├── agent.py            /agent/* routes
│       ├── voice.py            /voice/* routes
│       ├── anti_rag.py         /anti-rag/* routes
│       ├── harness.py          /harness/* routes
│       └── eval.py             /eval/* routes
│
├── tests/                      Test suite (runs without API keys)
│   ├── test_rag.py             RAG pipeline unit tests
│   └── test_agent.py           Agent tools and memory unit tests
│
├── .github/workflows/ci.yml    GitHub Actions: test → docker build → (deploy)
├── Dockerfile                  Production container image
├── docker-compose.yml          Local stack with PostgreSQL + pgvector
├── requirements.txt            Python dependencies
├── .env.example                Template for environment variables
└── CLAUDE.md                   Instructions for AI coding assistants
```

---

## 4. System Architecture

### How a request flows through the system

```
User Request
     │
     ▼
FastAPI (app/main.py)
     │  middleware: logging, CORS, error handling
     │
     ├──► /rag/*        RAGPipeline
     │         ingest: text → chunk → embed → store
     │         query:  embed query → retrieve → generate
     │
     ├──► /agent/*      Agent
     │         loop:    LLM + tools + memory → final response
     │
     ├──► /voice/*      VoicePipeline
     │         STT → LLM → TTS → audio response
     │
     ├──► /anti-rag/*   KAG / CAG / SQL / router
     │
     ├──► /harness/*    HarnessRunner
     │         task → executor → scorer → report
     │
     └──► /eval/*       LLMEvaluator + Tracer
```

### How modules share the LLM

All modules call `get_llm_client()` from `app/llm.py`. This returns a client for whichever provider is set in `.env`. No module imports `anthropic` or `openai` directly — they all go through the abstraction layer.

```
LLM_PROVIDER=ollama  →  OpenAICompatibleClient (localhost:11434)
LLM_PROVIDER=anthropic → AnthropicClient
LLM_PROVIDER=groq    →  OpenAICompatibleClient (api.groq.com)
LLM_PROVIDER=openai  →  OpenAICompatibleClient (api.openai.com)
```

The agent uses the same abstraction with an extended `chat_with_tools()` method that normalises tool calling across all providers.

### Data persistence layers

```
In-memory (default, zero setup)
  └── InMemoryVectorStore    — embeddings live in Python list, reset on restart
  └── EpisodicMemory         — conversation history in memory per session

PostgreSQL + pgvector (production)
  └── PgVectorStore          — persistent embeddings, survives restart
  └── VECTOR_STORE=pgvector in .env

Redis (optional, for caching + sessions)
  └── SemanticCache          — skip LLM for identical/near-identical queries
  └── PersistentStore        — agent memory survives restart
```

---

## 5. Getting Started

### Prerequisites

| Tool | Required | Notes |
|------|----------|-------|
| Python 3.9+ | Yes | 3.11 recommended |
| pip | Yes | Comes with Python |
| Ollama | Recommended | Free local LLM, no API costs |
| Anthropic API key | Optional | Required only for agent harness scoring |
| Docker | Optional | Needed for pgvector or containerised deployment |

### Local setup (recommended path)

```bash
# 1. Clone the repo
git clone https://github.com/naureenfathima/ai-bootcamp-starter.git
cd ai-bootcamp-starter

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Open .env and fill in ANTHROPIC_API_KEY at minimum
# (or skip it and use Ollama — see Section 9)

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** — you will see the full interactive API.

### Zero-cost dev mode

You can run the entire system without any paid API keys:

```bash
# In .env:
EMBEDDING_PROVIDER=local    # uses sentence-transformers (local model)
VECTOR_STORE=memory         # no database needed
LLM_PROVIDER=ollama         # local Ollama instead of Claude/GPT
LLM_MODEL=qwen2.5:7b
```

Then install and start Ollama:

```bash
brew install ollama           # macOS
ollama pull qwen2.5:7b        # download the model (~4.7 GB, one time)
ollama serve                  # starts on http://localhost:11434
```

### With Docker (includes pgvector)

```bash
cp .env.example .env          # fill in your keys
docker compose up             # starts app + PostgreSQL + pgvector
```

After the stack is up, run the pgvector migration once:

```bash
docker compose exec db psql -U bootcamp -d aibootcamp -c "
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY, source TEXT, chunk_idx INTEGER,
    content TEXT, metadata JSONB, embedding vector(1536)
  );
  CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
"
```

---

## 6. Configuration Reference

All settings live in `.env`. Copy `.env.example` to `.env` and edit it. **Never commit `.env` — it contains secrets.**

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `AI Bootcamp Starter` | Application name shown in API docs |
| `APP_VERSION` | `1.0.0` | Shown in health endpoint and docs |
| `DEBUG` | `false` | Enables DEBUG log level when `true` |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | Active provider: `anthropic` / `openai` / `ollama` / `groq` / `together` / `openrouter` |
| `LLM_MODEL` | `qwen2.5:7b` | Model name for the active provider. Leave blank to use provider's default |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `CLAUDE_MODEL` | `claude-3-5-sonnet-20241022` | Claude model version |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `GROQ_API_KEY` | — | Required when `LLM_PROVIDER=groq` |
| `TOGETHER_API_KEY` | — | Required when `LLM_PROVIDER=together` |
| `OPENROUTER_API_KEY` | — | Required when `LLM_PROVIDER=openrouter` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server URL (use after SSH tunnel for remote) |

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | `openai` or `local` (sentence-transformers) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector size — must match the model |

### Vector Database

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE` | `memory` | `memory` (dev) or `pgvector` (production) |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string for pgvector |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Words per chunk during ingestion |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks (preserves context at boundaries) |
| `RETRIEVAL_TOP_K` | `3` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum cosine similarity to include a chunk in results |
| `AGENTIC_MAX_ITERATIONS` | `3` | Max retrieve-reflect cycles in Agentic RAG |
| `AGENTIC_REFLECTION_THRESHOLD` | `0.7` | Evidence quality score above which iteration stops |
| `PAGE_INDEX_PAGE_SIZE` | `800` | Words per logical page in the hierarchical page index |

### Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_AGENT_STEPS` | `10` | Safety cap on tool-call iterations per request |
| `AGENT_TEMPERATURE` | `0.2` | Lower = more deterministic responses |

### Voice AI

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | — | Speech-to-text |
| `ELEVENLABS_API_KEY` | — | Text-to-speech |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice (default: Rachel) |
| `VOICE_STT_CONFIDENCE_THRESHOLD` | `0.5` | Below this score, ask the user to repeat |

---

## 7. Module Guide

---

### 7.1 LLM Provider Abstraction

**File:** `app/llm.py`

> **Concept:** Every module in this project needs to call an LLM. Without an abstraction, each module would import `anthropic` or `openai` directly and be locked to that provider. The Adapter pattern solves this: one interface, many implementations.

The `LLMClient` base class defines two methods:

```python
client.chat(user_message, system, max_tokens)         # simple completion
client.chat_with_tools(messages, tools, system, ...)  # agentic loop with tool calling
```

Both return normalised dataclasses (`LLMResponse`, `AgentTurnResponse`) regardless of which provider is active. Downstream code never sees the provider-specific format.

**Switching providers:**

```bash
# In .env
LLM_PROVIDER=ollama      # local, free
LLM_PROVIDER=anthropic   # Claude
LLM_PROVIDER=groq        # fast cloud, free tier
LLM_PROVIDER=together    # $25 free credit
LLM_PROVIDER=openrouter  # 200+ models
```

**Tool calling normalisation:** The agent maintains messages in OpenAI format internally. When `AnthropicClient.chat_with_tools()` is called, it converts the message list to Anthropic format before the API call and converts the response back. `OpenAICompatibleClient` uses the same format natively.

---

### 7.2 RAG — Retrieval-Augmented Generation

**Files:** `app/rag/`  |  **Endpoints:** `/rag/*`

> **Concept:** Large language models have a knowledge cutoff and no access to your private data. RAG solves this by retrieving relevant information at query time and injecting it into the prompt. The LLM answers using your documents, not its training data alone.

#### The standard pipeline

```
Ingest path:
  Your text
    │
    ├── chunk_text()       Split into overlapping chunks (ingestion.py)
    ├── embed()            Each chunk → numerical vector (embeddings.py)
    └── store.add()        Vectors stored for search (retrieval.py)

Query path:
  User question
    │
    ├── embed(question)    Question → vector
    ├── store.search()     Find most similar chunks (cosine similarity)
    └── LLM.chat()         "Given these chunks, answer: {question}"
```

#### Four chunking strategies (`chunking.py`)

| Strategy | Best for | How it works |
|----------|----------|-------------|
| `fixed_size` | General text | Splits every N words with overlap |
| `sentence` | Articles, prose | Splits at sentence boundaries |
| `paragraph` | Documentation | Splits at double newlines |
| `semantic` | Dense technical content | Groups sentences by embedding similarity |

#### Advanced RAG variants

**Agentic RAG** (`agentic_rag.py`) — Iterative loop. After retrieving, the LLM reflects on whether the evidence is sufficient. If not, it rewrites the query and retrieves again. Stops when evidence quality exceeds `AGENTIC_REFLECTION_THRESHOLD` or max iterations is hit.

**Corrective RAG** (`corrective_rag.py`) — Grades each retrieved chunk. If all chunks score below threshold, falls back to a web search rather than hallucinating. Uses the `LLMEvaluator` from `app/eval/` as the grader.

**GraphRAG** (`graph_rag.py`) — Builds an entity graph from ingested text (nodes = entities, edges = relationships). Supports two query modes: `local` (entity-focused, fast) and `global` (community summaries, broader context).

**Page Index** (`page_index.py`) — Two-tier hierarchy. First tier: logical page summaries. Second tier: individual chunks within pages. Query hits page tier first, then retrieves detailed chunks only from relevant pages. Reduces noise for large document collections.

**Metadata-aware retrieval** (`metadata.py`) — Attaches structured metadata (author, date, document type) to chunks at ingest time. Enables filtered search: "only from documents authored after 2024".

#### Demo flow

```bash
# 1. Add some text
POST /rag/ingest
{"text": "The transformer architecture was introduced in 2017 by Vaswani et al."}

# 2. Ask a question
POST /rag/query
{"question": "Who introduced the transformer architecture?"}
# → Response cites retrieved chunk, score ~0.85

# 3. Check what's indexed
GET /rag/status
# → {"count": 1}
```

---

### 7.3 Anti-RAG — When Not to Use RAG

**Files:** `app/anti_rag/`  |  **Endpoints:** `/anti-rag/*`

> **Concept:** RAG is not always the right tool. When your knowledge base is small and stable, a graph-based approach may reason more precisely. When you have structured data, SQL is more reliable than embedding search. This module teaches you to recognise which tool to use.

#### KAG vs CAG comparison

| | KAG (Knowledge-Augmented Generation) | CAG (Context-Augmented Generation) |
|-|--------------------------------------|-------------------------------------|
| **How it works** | Graph traversal over entities | Entire document corpus in the context window |
| **Best for** | Multi-hop entity questions | Synthesis across many documents |
| **Misses context when** | A graph edge is missing | Never — it sees everything |
| **Token cost** | Lower per query | High first call, low on repeat (cached) |
| **Shows reasoning** | Yes — `reasoning_path` in response | No — implicit in context |

#### Demo script

```bash
# Seed the org-chart knowledge base (no API key needed)
POST /anti-rag/demo/seed

# Run the same question through both
POST /anti-rag/kag/query
{"question": "What does Bob's team own?", "max_hops": 2}
# → reasoning_path shows graph traversal

POST /anti-rag/cag/query
{"question": "What does Bob's team own?"}
# First call: cache_creation_input_tokens > 0
# Second call: cache_read_input_tokens > 0  ← this is the caching lesson

# Side-by-side comparison
POST /anti-rag/compare
{"question": "Who manages the team that owns the RAG Service?"}
```

#### Hybrid router

`POST /anti-rag/route` classifies query intent and selects the strategy automatically:
- Entity/relationship questions → KAG
- Synthesis/summary questions → CAG
- Tabular/numerical questions → SQL

---

### 7.4 Voice AI

**Files:** `app/voice/`  |  **Endpoints:** `/voice/*`

> **Concept:** A voice AI system converts speech to text, processes the text with an LLM, then converts the response back to speech. The hard part is latency — users expect near-real-time responses. Each stage has a latency budget.

#### Pipeline

```
Audio input
    │
    ▼
Deepgram STT          ~150–300 ms
    │  transcript + confidence score
    │  if confidence < VOICE_STT_CONFIDENCE_THRESHOLD → ask to repeat
    ▼
LLM (get_llm_client)  ~400–800 ms
    │  text response
    ▼
ElevenLabs TTS        ~200–400 ms
    │
    ▼
Audio output
```

#### Endpoints

```bash
# Speech-to-text only (returns transcript)
POST /voice/transcribe
Content-Type: multipart/form-data
file: <audio file>

# Full pipeline: audio in → audio out
POST /voice/process
Content-Type: multipart/form-data
file: <audio file>

# Clear conversation history
POST /voice/reset
```

---

### 7.5 Autonomous Agents

**Files:** `app/agent/`  |  **Endpoints:** `/agent/*`

> **Concept:** A language model by itself only generates text. An agent adds the ability to take actions — calling tools, storing information, and looping until a task is complete. The key insight: the LLM decides when to call a tool and what to pass to it, not the programmer.

#### The agentic loop

```
User message
     │
     ▼
LLM with tools ──── wants to call a tool?
     │                       │ Yes
     │ No (final answer)     ▼
     │              Execute tool(name, args)
     │                       │
     │              Feed result back to LLM
     │                       │
     └──────────────────◄────┘
     │
     ▼
Response to user
```

The loop runs until either the LLM produces a final text answer or `MAX_AGENT_STEPS` is reached.

#### Available tools

| Tool | What it does |
|------|-------------|
| `calculator` | Evaluates mathematical expressions safely (`sqrt`, `log`, `sin`, etc.) |
| `get_current_time` | Returns current UTC date and time |
| `search_web` | Stub — implement with Tavily or Serper API |

#### Adding a new tool

```python
# In app/agent/tools.py
class MyTool(Tool):
    name = "my_tool"
    description = "What this tool does and when to use it."
    input_schema = {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."}
        },
        "required": ["param"],
    }

    def execute(self, param: str) -> str:
        # Your implementation
        return result
```

Register it by adding `MyTool()` to `DEFAULT_TOOLS` at the bottom of `tools.py`.

#### Memory types

```
WorkingMemory    Per-task scratchpad. Cleared every request.
                 Holds intermediate results, current intent.

EpisodicMemory   Conversation history for the current session.
                 Sliding window by default; summarisation mode available.

SemanticMemory   Long-term facts across sessions.
                 Backed by RAG vector store (student TODO: wire up).

PersistentStore  Saves sessions to Redis or SQLite (student TODO: implement).
```

#### Demo

```bash
# Single tool
POST /agent/run
{"message": "What is the square root of 1764?"}
# → calls calculator, returns 42

# Multi-tool
POST /agent/run
{"message": "What is 25% of 840, and what time is it?"}
# → calls calculator AND get_current_time

# Multi-turn conversation
POST /agent/run  {"message": "My name is Alice."}
POST /agent/run  {"message": "What is my name?"}
# → "Your name is Alice." (episodic memory)

# Reset memory
POST /agent/reset
```

---

### 7.6 Agent Harness

**Files:** `app/harness/`  |  **Endpoints:** `/harness/*`

> **Concept:** A test harness is different from a unit test. Unit tests check code logic. An agent harness runs the actual agent against defined goals and measures whether it succeeded. It captures every tool call (the "trajectory") so you can see exactly what the agent did at each step.

#### What the harness measures

```
Task (goal + success criteria)
    │
    ▼
AgentExecutor  →  runs agent, records full trajectory
    │               [{step, tool_name, tool_input, tool_output}]
    ▼
TaskScorer     →  three dimensions:
    │    outcome    (60%)  did the response meet success criteria?  [LLM judge]
    │    tool_use   (25%)  did it call the expected tools?          [deterministic]
    │    efficiency (15%)  was the step count reasonable?           [deterministic]
    ▼
HarnessReport  →  pass_rate, avg_score, GREEN / YELLOW / RED
```

#### Running the harness

```bash
# List available tasks
GET /harness/tasks

# Run one task
POST /harness/run/task
{"task_id": "<id from above>"}
# Returns: score breakdown, full trajectory, latency, token count

# Run full suite
POST /harness/run/suite
# Returns: HarnessReport with GREEN / YELLOW / RED status

# Define and run a custom task inline
POST /harness/run/custom
{
  "description": "Compound interest calculation",
  "input": "If I invest $5,000 at 7% for 3 years compounded annually, what is the result?",
  "expected_tools": ["calculator"],
  "success_criteria": "Answer is approximately $6,125.22",
  "tags": ["math", "finance"]
}
```

#### Report status thresholds

| Status | Pass rate | Meaning |
|--------|-----------|---------|
| `GREEN` | 100% | All tasks passed — safe to deploy |
| `YELLOW` | 70–99% | Some regressions — investigate before deploying |
| `RED` | < 70% | Significant regression — do not deploy |

---

### 7.7 Evaluation

**Files:** `app/eval/`  |  **Endpoints:** `/eval/*`

> **Concept:** How do you know your RAG system is actually giving good answers? You measure it. The evaluation module scores static (question, context, answer) triples across four dimensions using an LLM as the judge.

#### Scoring dimensions

| Dimension | Question asked | Score |
|-----------|---------------|-------|
| **Faithfulness** | Does the answer contain only information from the context? | 0–1 |
| **Relevance** | Does the answer actually address the question? | 0–1 |
| **Context recall** | Does the answer use all the important information in the context? | 0–1 |
| **Completeness** | Is the answer thorough, or does it miss important aspects? | 0–1 |

#### Request tracing

Every request can be traced with spans for each stage (retrieval, LLM call, etc.):

```bash
# View latency and token stats across all traced requests
GET /eval/tracer/stats

# View the most recent traced requests
GET /eval/tracer/recent
```

#### Prompt version registry

Register and retrieve prompt versions to compare how prompt changes affect scores:

```bash
POST /eval/prompts/register
{"name": "rag_system_v2", "template": "Answer the question using only..."}

GET /eval/prompts/rag_system_v2
```

---

### 7.8 Production Patterns

**Files:** `app/production/`

> **Concept:** A system that works in a demo may not survive real traffic. This module implements five patterns that every production AI system needs.

#### Semantic Cache (`cache.py`)

Stores previous (query → response) pairs. Before calling the LLM, the cache checks whether a semantically similar query was already answered. If yes, returns the cached answer immediately.

```
Query: "What is the capital of France?"
Cache hit: "What is France's capital?" → "Paris"
→ Returns "Paris" without an LLM call
```

Cost: Zero tokens. Latency: ~1 ms instead of ~1000 ms.

#### Rate Limiter (`rate_limiter.py`)

Token-bucket algorithm. Each tenant has a bucket with a maximum token count. Every request consumes tokens. Tokens replenish over time. Requests that would empty the bucket are rejected with HTTP 429.

#### Resilience (`resilience.py`)

Three layered patterns:

| Pattern | What it does |
|---------|-------------|
| **Retry with backoff** | Retries failed API calls with exponential delay |
| **Circuit breaker** | After N failures, stops calling the failing service for a cooldown period, preventing cascade failures |
| **Fallback chain** | If primary LLM fails, tries secondary, then tertiary |

#### Cost Tracker (`cost_tracker.py`)

Tracks token usage per request, per tenant, and per time window. Alerts when a tenant approaches their budget limit.

#### Multi-Tenancy (`multi_tenant.py`)

Isolates data and enforces quotas per tenant. Each tenant's documents, conversation history, and usage metrics are completely separated.

---

### 7.9 Model Context Protocol (MCP)

**Files:** `app/mcp/`

> **Concept:** MCP is an open standard that lets external tools (like Claude Desktop or other AI clients) call your application's capabilities as if they were native tools. Your RAG pipeline and agent become composable building blocks in any MCP-compatible system.

Register handlers in `app/mcp/server.py` and any MCP client can discover and call them. This is the architecture that allows AI systems to extend each other's capabilities without custom integrations.

---

## 8. API Reference

The full interactive API is at **http://localhost:8000/docs** when the server is running.

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Returns `{"status": "ok", "version": "...", "app_name": "..."}` |

### RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/rag/ingest` | Add text to the knowledge base. Body: `{"text": "..."}` |
| POST | `/rag/ingest/file` | Upload a `.txt` file (multipart form) |
| GET | `/rag/status` | Returns count of indexed chunks |
| POST | `/rag/query` | Ask a question. Body: `{"question": "..."}` |
| POST | `/rag/agentic/query` | Iterative retrieve-reflect loop |
| POST | `/rag/corrective/query` | Grade docs, fall back to web if needed |
| POST | `/rag/graph/build` | Build entity graph from ingested text |
| POST | `/rag/graph/query` | Query via GraphRAG. Body: `{"query": "...", "mode": "local\|global"}` |
| POST | `/rag/page-index/build` | Build two-tier hierarchical index |
| POST | `/rag/page-index/query` | Two-tier retrieval |

### Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/run` | Run the agent. Body: `{"message": "..."}` |
| POST | `/agent/reset` | Clear all memory |
| GET | `/agent/tools` | List registered tools and their schemas |

### Voice

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/voice/transcribe` | Speech-to-text only (returns transcript) |
| POST | `/voice/process` | Full pipeline: audio in → audio out |
| POST | `/voice/reset` | Clear conversation history |

### Anti-RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/anti-rag/demo/seed` | Load org-chart demo data into KAG + CAG |
| POST | `/anti-rag/kag/query` | Knowledge graph traversal query |
| POST | `/anti-rag/cag/query` | Full context window query |
| POST | `/anti-rag/compare` | Side-by-side KAG vs CAG for the same question |
| POST | `/anti-rag/sql/query` | Text-to-SQL query |
| POST | `/anti-rag/route` | Show which strategy the router selects |
| GET | `/anti-rag/fine-tuning/guide` | Fine-tuning vs RAG decision framework |

### Agent Harness

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/harness/tasks` | List all tasks in the default test suite |
| POST | `/harness/run/task` | Run one task by ID. Body: `{"task_id": "..."}` |
| POST | `/harness/run/suite` | Run the full suite. Returns `HarnessReport` |
| POST | `/harness/run/custom` | Define and run a one-off task inline |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/evaluate` | Score a single (question, context, answer) triple |
| POST | `/eval/evaluate/batch` | Score a list of triples, returns regression report |
| GET | `/eval/tracer/stats` | Latency, token, and cost statistics |
| GET | `/eval/tracer/recent` | Most recent traced requests |
| POST | `/eval/prompts/register` | Register a prompt version |
| GET | `/eval/prompts/{name}` | Retrieve a registered prompt |

---

## 9. LLM Provider Switching

The `LLM_PROVIDER` setting in `.env` controls which LLM all modules use. No code changes required.

### Provider comparison

| Provider | Cost | Speed | Best for |
|----------|------|-------|----------|
| **Ollama** | Free | Medium | Development, privacy, offline use |
| **Anthropic (Claude)** | Pay-per-token | Medium | Best quality, long context |
| **Groq** | Free tier | Very fast | Rapid prototyping, latency-sensitive |
| **Together AI** | $25 free | Fast | Open-source models, fine-tuning |
| **OpenRouter** | Pay-per-token | Varies | Access to 200+ models via one key |
| **OpenAI** | Pay-per-token | Fast | GPT-4o, well-documented |

### Ollama (local, recommended for development)

```bash
brew install ollama
ollama pull qwen2.5:7b    # 4.7 GB, good quality
ollama serve

# .env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

For 8 GB RAM, `llama3.2:3b` (2 GB) is faster and still capable for most tasks.

### Remote Ollama via SSH tunnel

If Ollama runs on a server you access over SSH:

```bash
# Forward the remote port to your local machine
ssh -L 11434:localhost:11434 user@your-server-ip

# Add your SSH public key for passwordless access
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@your-server-ip

# .env stays the same — tunnel makes remote look local
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Groq (cloud, free tier)

```bash
# Get a free API key at console.groq.com
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.1-8b-instant   # fastest, free
```

---

## 10. Testing

### Run the full test suite

```bash
pytest tests/ -v
```

All tests run without real API keys — external services are stubbed or mocked.

### Run a specific test file or test

```bash
pytest tests/test_rag.py -v
pytest tests/test_agent.py::TestCalculatorTool::test_addition -v
```

### What is tested

| File | What it covers |
|------|---------------|
| `tests/test_rag.py` | Chunking correctness, cosine similarity math, vector store search and filtering |
| `tests/test_agent.py` | Calculator safety, time tool format, episodic memory window, working memory |

### Writing new tests

Tests go in `tests/`. Follow the existing pattern: group related tests in a class, use `pytest.approx` for floats. Mark tests that require real API keys with `@pytest.mark.integration` and exclude them in CI.

---

## 11. CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

GitHub Actions runs automatically on every push to `main` and every pull request.

### Pipeline stages

```
Push to main / Pull Request
         │
         ▼
    ┌─────────┐
    │  test   │  Python 3.11, pytest tests/, no real API keys
    └────┬────┘
         │ passes
         ▼
    ┌─────────┐
    │ docker  │  docker build — validates the image builds cleanly
    └────┬────┘
         │ passes
         ▼
    ┌──────────┐
    │  deploy  │  (student TODO: uncomment and configure)
    └──────────┘
```

### Activating the deploy step

In `.github/workflows/ci.yml`, uncomment the `deploy` block and add your `RAILWAY_TOKEN` to GitHub Secrets (Settings → Secrets → Actions):

```yaml
deploy:
  name: Deploy to Production
  runs-on: ubuntu-latest
  needs: [test, docker]
  if: github.ref == 'refs/heads/main'
  steps:
    - name: Deploy to Railway
      env:
        RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
      run: |
        npm install -g @railway/cli
        railway up --service ai-bootcamp-starter
```

---

## 12. Docker and Deployment

### Image design

The `Dockerfile` uses Python 3.11-slim to keep the image small. Key decisions:

- Dependencies are installed before copying application code — this leverages Docker layer caching so re-builds only reinstall packages when `requirements.txt` changes.
- The app runs as a non-root user (`appuser`) — this is a security requirement for production containers.
- A `HEALTHCHECK` is configured — Docker will restart the container if `/health` stops responding.

### Local development with docker compose

```bash
docker compose up          # start app + PostgreSQL + pgvector
docker compose down        # stop and remove containers
docker compose logs -f     # tail all logs
```

### Production deployment options

**Railway (easiest):**
1. Push to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Set environment variables in the Railway dashboard
4. Auto-deploys on every push to `main`

**Render:**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Docker on any VPS:**
```bash
docker build -t ai-bootcamp .
docker run -d -p 8000:8000 --env-file .env --restart=unless-stopped ai-bootcamp
```

---

## 13. Student Task Index

The codebase contains intentional gaps marked `# STUDENT TODO`. This section lists them by module and difficulty.

### Module 3 — RAG

| Task | File | Difficulty |
|------|------|------------|
| Replace `InMemoryVectorStore` with `PgVectorStore` | `app/rag/retrieval.py` | Medium |
| Add metadata filtering to search | `app/rag/metadata.py` | Medium |
| Add a `cohere` embedding provider | `app/rag/embeddings.py` | Easy |

### Module 4 — Agents & Voice

| Task | File | Difficulty |
|------|------|------------|
| Implement `EpisodicMemory._summarise_oldest()` | `app/agent/memory.py` | Medium |
| Connect `SemanticMemory` to `RAGPipeline` | `app/agent/memory.py` | Medium |
| Implement `PersistentStore` with Redis or SQLite | `app/agent/memory.py` | Hard |
| Implement `WebSearchTool` with Tavily or Serper API | `app/agent/tools.py` | Easy |

### Module 5 — Harness, Eval, Production

| Task | File | Difficulty |
|------|------|------------|
| Write 5+ domain-specific harness tasks | `app/harness/tasks.py` | Easy |
| Parallelise the harness runner | `app/harness/runner.py` | Medium |
| Persist `HarnessReport` to JSON after each run | `app/harness/runner.py` | Easy |
| Add CI integration for the harness | `.github/workflows/ci.yml` | Easy |
| Add score breakdown by tag | `app/harness/runner.py` | Medium |
| Implement `SemanticCache` with Redis | `app/production/cache.py` | Hard |
| Implement `PersistentStore` with Redis | `app/agent/memory.py` | Hard |
| Add a HuggingFace Inference Endpoint provider | `app/llm.py` | Medium |

### Module 6 — Capstone

Choose one of: **RAG System**, **Voice AI Assistant**, or **Autonomous Agent**.

Your capstone submission must include:
- A working deployed URL
- A system diagram
- A completed agent harness suite (minimum 5 domain tasks, `GREEN` report)
- A RAG evaluation batch report (minimum 10 question-answer pairs)

---

## 14. Glossary

**Agent** — A system where an LLM decides what actions to take (tool calls) and loops until it completes a task, rather than generating a single response.

**Agentic loop** — The while-loop pattern: call LLM → if tool call, execute → feed result back → repeat until final answer.

**Anthropic** — The company that makes Claude. The Anthropic API uses a different message format than OpenAI's.

**CAG (Context-Augmented Generation)** — Putting your entire knowledge base into the LLM's context window instead of retrieving relevant parts. Works well for small corpora with caching.

**Chunking** — Splitting a large document into smaller overlapping segments so each can be embedded independently and retrieved by relevance.

**Circuit breaker** — A pattern that stops calling a failing service after repeated errors, allowing it time to recover before retrying.

**Cosine similarity** — A mathematical measure of how similar two vectors are (0 = unrelated, 1 = identical, -1 = opposite). Used to rank retrieved chunks by relevance to a query.

**Embedding** — A list of numbers (a vector) that represents the semantic meaning of text. Two texts with similar meaning will have similar vectors.

**Episodic memory** — Short-term memory of the current conversation. Analogous to remembering what was said five minutes ago.

**Faithfulness** — An evaluation metric: does the answer contain only information from the provided context, with no invented facts?

**FastAPI** — The Python web framework this project uses. Automatically generates interactive API documentation at `/docs`.

**Groq** — A cloud inference provider known for very fast LLM responses. Has a free tier.

**KAG (Knowledge-Augmented Generation)** — Answering questions by traversing a structured knowledge graph (nodes = entities, edges = relationships) instead of embedding search.

**LLM (Large Language Model)** — A neural network trained on large amounts of text that can generate, summarise, translate, and reason about language. Examples: Claude, GPT-4, Llama.

**MCP (Model Context Protocol)** — An open standard for AI systems to expose and consume capabilities as composable tools, similar to how APIs work for web services.

**Ollama** — A tool that runs LLMs locally on your machine. Free, private, no API costs.

**pgvector** — A PostgreSQL extension that adds vector similarity search to a standard relational database.

**Prompt caching** — Storing the computation for a long prompt prefix so repeated calls reuse the cached result rather than reprocessing. Reduces cost and latency for static context.

**RAG (Retrieval-Augmented Generation)** — Answering questions by first retrieving relevant documents, then having the LLM answer using those documents as context.

**Semantic cache** — A cache that uses embedding similarity to find cached responses for questions that are semantically equivalent, even if worded differently.

**Sentence-transformers** — A Python library for generating text embeddings locally, without an API key. Slower than cloud embeddings but free.

**Trajectory** — In the agent harness, the complete record of every tool call an agent made: step number, tool name, inputs, and outputs.

**Token** — The unit LLMs process. Roughly 0.75 words. All API pricing is per token. A 1,000-word document ≈ 1,300 tokens.

**Tool calling** — The mechanism by which an LLM requests that a specific function be executed. The LLM outputs a structured `{name, arguments}` object; the application executes the function and returns the result.

**Vector** — A list of numbers used to represent text as a point in high-dimensional space. Semantically similar texts map to nearby points.

**Vector store** — A database optimised for storing and searching vectors by similarity. This project supports in-memory (dev) and pgvector (production).

**Working memory** — Per-task scratchpad storage that holds intermediate results for the current request and is cleared when the request is done.

---

*Handbook maintained by Naureen Fathima — BlockseBlock AI Systems Engineering Bootcamp.*
