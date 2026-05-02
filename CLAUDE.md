# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the dev server
uvicorn app.main:app --reload --port 8000

# Run all tests (no API keys required)
pytest tests/ -v

# Run a single test file
pytest tests/test_rag.py -v

# Run a single test
pytest tests/test_agent.py::TestCalculatorTool::test_addition -v

# Docker
docker build -t ai-bootcamp .
docker run -p 8000:8000 --env-file .env ai-bootcamp
docker-compose up   # includes PostgreSQL + pgvector
```

For dev without paid APIs, set in `.env`:
```
EMBEDDING_PROVIDER=local
VECTOR_STORE=memory
```

## Architecture

FastAPI app (`app/main.py`) wires six routers, each backed by a module:

| Router prefix | Module | What it does |
|---|---|---|
| `/rag` | `app/rag/` | Ingest → chunk → embed → store → retrieve → generate |
| `/anti-rag` | `app/anti_rag/` | KAG (entity graph), Text-to-SQL, LoRA concepts, hybrid router |
| `/agent` | `app/agent/` | Tool-calling agentic loop with four memory types |
| `/voice` | `app/voice/` | Deepgram STT → Claude → ElevenLabs TTS pipeline |
| `/harness` | `app/harness/` | LLM-as-judge eval, request tracing, prompt version registry |
| `/production` | consumed internally | Semantic cache, token-bucket rate limiter, circuit breaker, cost tracker, multi-tenancy |

All settings are loaded once via `app/config.py` (`pydantic-settings` reading `.env`). Import as `from app.config import settings`.

### RAG pipeline (`app/rag/`)
Data flows through four files in order: `ingestion.py` (word-level chunking with overlap) → `embeddings.py` (OpenAI or local `sentence-transformers`) → `retrieval.py` (`InMemoryVectorStore` or pgvector) → `pipeline.py` (end-to-end `RAGPipeline` class used by the API and `SemanticMemory`).

### Agent memory (`app/agent/memory.py`)
Four classes with increasing scope:
- `WorkingMemory` — per-task dict scratchpad, cleared each request
- `EpisodicMemory` — conversation history; sliding window by default, summarisation path left as a student TODO
- `SemanticMemory` — long-term facts; currently a sorted in-memory list, intended to be wired to `RAGPipeline`
- `PersistentStore` — cross-session persistence stub (Redis/SQLite implementation is a student TODO)

### Student TODOs
The codebase has intentional stubs marked `# STUDENT TODO`. Key ones:
- `EpisodicMemory._summarise_oldest()` — implement with Anthropic client
- `SemanticMemory.remember()` / `recall()` — connect to `RAGPipeline`
- `PersistentStore` — implement with Redis or SQLite
- `InMemoryVectorStore` → `PgVectorStore` migration
- Deploy step in `.github/workflows/ci.yml`

## Anti-RAG: CAG vs KAG demo script

The `app/anti_rag/` module covers four alternative approaches. CAG and KAG are the two hands-on ones students demo side by side.

### Concept summary

| | CAG | KAG |
|---|---|---|
| Retrieval step | None — all docs in context | Graph traversal |
| Can miss context | Never | Yes (missing edge) |
| Best for | Small corpora, synthesis queries | Multi-hop entity reasoning |
| First-call cost | High (pays full tokens) | Medium |
| Repeat-call cost | Low (prompt cache hit) | Medium |

### Step 1 — Seed demo data (no API key needed for this step)

```
POST /anti-rag/demo/seed
```

Loads an identical org-chart knowledge base into both KAG and CAG without any API calls. The data: Alice (CTO) → Bob (Backend Lead) → Platform Team → RAG Service → VectorDB; Carol (ML Lead) → AI Team → Agent Service.

### Step 2 — Run the same question through each pipeline

**KAG (graph traversal):**
```
POST /anti-rag/kag/query
{"question": "What does Bob's team own?", "max_hops": 2}
```
Look at `reasoning_path` — students can see which graph edges were traversed.

**CAG (full context window):**
```
POST /anti-rag/cag/query
{"question": "What does Bob's team own?"}
```
First call: `cache_creation_input_tokens > 0`. Run it again: `cache_read_input_tokens > 0`. This is the caching lesson.

### Step 3 — Side-by-side comparison

```
POST /anti-rag/compare
{"question": "Who manages the team that owns the RAG Service?"}
```

Returns KAG and CAG answers together with token counts and a `teaching_note`.

### Step 4 — Show where each approach wins

**KAG wins (multi-hop entity reasoning):**
```
POST /anti-rag/compare
{"question": "Who manages the team that owns the RAG Service?"}
```
KAG traverses `rag_service → platform_team → bob → alice` in one structured hop. CAG reads all documents and has to infer the chain.

**CAG wins (cross-document synthesis):**
```
POST /anti-rag/compare
{"question": "List everything owned by each team."}
```
CAG reads all documents and synthesizes a complete picture. KAG only traverses from entities it finds in the query text.

**Router decision:**
```
POST /anti-rag/route
{"query": "Summarize everything about each team's services"}
```
Shows how the hybrid router classifies query intent and picks a strategy.

### Key numbers to call out in class

- `context_window_pct` in CAG responses: how much of Claude's 200K window the corpus uses
- `cache_creation_input_tokens` → `cache_read_input_tokens` across two identical CAG calls: the caching benefit
- `reasoning_path` in KAG: the explicit graph traversal chain (students can see the "thinking")
- Token cost of KAG vs CAG for the same question: KAG is cheaper per call on large corpora

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs tests then a Docker build on every push/PR to `main`. Tests run with `ANTHROPIC_API_KEY=test_key`, `EMBEDDING_PROVIDER=local`, `VECTOR_STORE=memory` so no real credentials are needed.
