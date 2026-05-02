# AI Bootcamp Starter 🚀

> Starter project for the **BlockseBlock AI Systems Engineering Bootcamp**
> Instructor: Naureen Fathima

This is your foundation. Every module of the course adds a layer to this project until you have a **deployed, production-grade AI product** with a public URL.

---

## What's Inside

```
ai-bootcamp-starter/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── config.py        # All settings (read from .env)
│   ├── rag/             # Module 3 — RAG pipeline
│   │   ├── ingestion.py   chunking & document loading
│   │   ├── embeddings.py  embedding generation
│   │   ├── retrieval.py   vector store & similarity search
│   │   └── pipeline.py    end-to-end RAG (ingest + query)
│   ├── anti_rag/        # Module 3 (extended) — Anti-RAG approaches
│   │   ├── kag.py         Knowledge Augmented Generation (entity graphs)
│   │   ├── structured_knowledge.py  Text-to-SQL for structured data
│   │   ├── fine_tuning.py  LoRA/PEFT concepts + training data builder
│   │   └── router.py      Hybrid router: pick the right strategy per query
│   ├── voice/           # Module 4 — Voice AI
│   │   ├── stt.py         speech-to-text (Deepgram)
│   │   ├── tts.py         text-to-speech (ElevenLabs)
│   │   └── pipeline.py    full voice pipeline
│   ├── agent/           # Module 4 — Autonomous Agent
│   │   ├── tools.py       tool definitions (calculator, time, web search)
│   │   ├── memory.py      working + episodic + semantic + persistent memory
│   │   └── agent.py       agentic loop with tool use
│   ├── harness/         # Module 4/5 — Evaluation Harness
│   │   ├── evaluator.py   LLM-as-judge evaluation engine
│   │   ├── metrics.py     faithfulness, relevance, recall, completeness
│   │   └── tracer.py      request tracing + prompt version registry
│   ├── production/      # Module 5 — Production patterns
│   │   ├── cache.py       semantic caching (30-50% cost reduction)
│   │   ├── rate_limiter.py  token bucket rate limiting per user/plan
│   │   ├── resilience.py  retry, fallback, circuit breaker
│   │   ├── cost_tracker.py  real-time cost tracking and budget alerts
│   │   └── multi_tenant.py  user-level isolation and data segregation
│   ├── mcp/             # Module 5 — MCP Protocol
│   │   └── server.py      MCP request/response dispatcher
│   └── api/             # REST API routes
│       ├── health.py      GET /health
│       ├── rag.py         POST /rag/ingest, POST /rag/query
│       ├── agent.py       POST /agent/run
│       └── voice.py       POST /voice/process
├── tests/               # Unit tests (run with pytest)
├── .github/workflows/   # CI pipeline (GitHub Actions)
├── Dockerfile           # Production container
├── docker-compose.yml   # Local dev with PostgreSQL + pgvector
├── requirements.txt
└── .env.example         ← copy this to .env
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/naureenfathima/ai-bootcamp-starter.git
cd ai-bootcamp-starter

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API keys

```bash
cp .env.example .env
# Open .env and fill in your keys
```

| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/ |
| `OPENAI_API_KEY` | https://platform.openai.com/ (optional, for OpenAI embeddings) |
| `DEEPGRAM_API_KEY` | https://console.deepgram.com/ (for voice AI) |
| `ELEVENLABS_API_KEY` | https://elevenlabs.io/ (for voice AI) |

> **For development**, set `EMBEDDING_PROVIDER=local` and `VECTOR_STORE=memory` — no paid APIs needed to run the RAG pipeline.

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs to see the interactive API.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run without any API keys — all external services are either unused or stubbed.

---

## Module Build Guide

Each module adds to this project. Follow this order:

### Module 2 — AI Systems Foundations
- Read `app/config.py` and understand how settings are loaded from `.env`
- Read `app/main.py` and understand the middleware and error handler
- **Your task:** Add a logging statement that records the token count of every Claude API call

### Module 3 — RAG Systems
- Read `app/rag/` — start with `ingestion.py`, then `embeddings.py`, then `retrieval.py`, then `pipeline.py`
- Run the server and test `/rag/ingest` and `/rag/query` via `/docs`
- **Your task:** Replace `InMemoryVectorStore` with `PgVectorStore` using pgvector

### Module 3 (extended) — Anti-RAG Approaches
RAG is not always the right tool. Read `app/anti_rag/` to understand when to use alternatives:
- `kag.py` — **KAG (Knowledge Augmented Generation)**: entities + relations in a graph instead of vector chunks. Best for multi-hop reasoning ("Who manages what?").
- `structured_knowledge.py` — **Text-to-SQL**: if your data is in a database, write SQL instead of doing vector search. Always correct for aggregations.
- `fine_tuning.py` — **Fine-tuning concepts (LoRA/PEFT)**: when to bake knowledge into weights instead of retrieving it at runtime. Includes a training data generator.
- `router.py` — **Hybrid router**: classifies each query and routes it to the right strategy.
- **Your task:** Call `/anti-rag/route` with 5 different questions and explain the routing logic.

### Module 4 — Voice AI & Agents with Memory
- Read `app/voice/pipeline.py` and `app/agent/agent.py`
- Explore `app/agent/memory.py` — understand the four memory types:
  - `WorkingMemory` — per-task scratchpad
  - `EpisodicMemory` — conversation history with summarisation
  - `SemanticMemory` — long-term facts backed by the RAG vector store
  - `PersistentStore` — cross-session persistence stub (implement with Redis)
- Add your Deepgram and ElevenLabs keys to `.env` and test `/voice/process`
- Test the agent at `/agent/run` with: `{"message": "What is sqrt(144) + the current time?"}`
- **Your task:** Implement `EpisodicMemory._summarise_oldest()` using the Anthropic client.

### Module 4/5 — Evaluation Harness
- Read `app/harness/` — this is how you measure quality before deploying changes
- Test the evaluator at `/harness/evaluate` with a (question, context, answer) triple
- Understand the three core metrics: faithfulness, relevance, context_recall
- Use `/harness/tracer/stats` to see per-request latency and cost breakdowns
- **Your task:** Build a test suite of 10 golden Q&A pairs and run `/harness/evaluate/batch`.

### Module 5 — MCP & Production Patterns
- Read `app/mcp/server.py` and register your RAG and Agent as MCP methods
- Explore `app/production/` — every file covers a production concern:
  - `cache.py` — semantic caching: 30-50% cost reduction with ~5ms cache hits
  - `rate_limiter.py` — per-user token bucket rate limiting (free/pro/enterprise plans)
  - `resilience.py` — retry with exponential backoff, circuit breaker, fallback chain
  - `cost_tracker.py` — real-time cost tracking, budget alerts, optimisation tips
  - `multi_tenant.py` — user-level isolation using context variables
- Containerise: `docker build -t ai-bootcamp . && docker run -p 8000:8000 --env-file .env ai-bootcamp`
- Push to Railway or Render and get a public URL
- **Your task:** Uncomment the deploy step in `.github/workflows/ci.yml`

### Module 6 — Capstone
- Choose your path: **RAG System**, **Voice AI Assistant**, or **Autonomous Agent**
- Extend this starter into a complete, deployed product
- Your capstone README must include: system diagram, tech stack, setup guide, public URL

---

## Student FAQ

**Do you cover data pipelines (batch/streaming) and queue systems like Kafka/SQS?**
We don't cover Kafka/SQS specifically, but we do teach event-driven architecture and how to leverage it based on your use case. The production patterns module shows you how to integrate async processing into AI systems.

**Is LLM observability included (evaluation, tracing, prompt/version tracking)?**
Yes — the `app/harness/` module covers all three: LLM-as-judge evaluation, per-request tracing with cost breakdowns, and the `PromptVersionRegistry` for tracking prompt changes over time.

**Are production patterns like async processing, caching, and rate limiting taught?**
Yes — `app/production/` covers semantic caching, token bucket rate limiting per plan, retry/circuit breaker/fallback, and cost tracking.

**Do you cover cost optimisation for LLM apps?**
Yes — `app/production/cost_tracker.py` covers real-time tracking and optimisation strategies (semantic caching, model selection, prompt compression, retrieval tuning). We go through the strategy in depth even if not every technique is implemented end-to-end.

**Is fine-tuning (LoRA/PEFT) and dataset prep included?**
We cover fine-tuning conceptually within the Anti-RAG module. `app/anti_rag/fine_tuning.py` includes the LoRA/PEFT decision framework, a training data builder (generate datasets with Claude), and the `when-to-fine-tune-vs-RAG` decision logic. Hands-on LoRA training is not in scope, but the concepts and dataset preparation are fully covered.

**Do you teach multi-tenant AI architecture and user-level isolation?**
Yes — `app/production/multi_tenant.py` covers row-level isolation with context variables, tenant-scoped vector stores, and a FastAPI middleware for extracting tenant identity from request headers.

**Are reliability patterns like retries, fallbacks, and circuit breakers covered?**
Yes — `app/production/resilience.py` implements all three as reusable Python primitives you can wrap around any external API call.

**Do you go beyond Docker into cloud architecture (AWS/GCP, scaling, queues)?**
We don't cover cloud-specific services. Instead we teach the general architectural patterns (event-driven design, horizontal scaling, queue-based decoupling) and show you how to map these to whatever cloud services your team uses. Students don't need cloud access — the course proposes system architecture best practices and guides you through applying them to your own infrastructure.

---

## API Reference

Once running, visit **http://localhost:8000/docs** for the full interactive API documentation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/rag/ingest` | POST | Add text to the knowledge base |
| `/rag/ingest/file` | POST | Upload a .txt file |
| `/rag/query` | POST | Ask a question against the knowledge base |
| `/rag/status` | GET | How many chunks are indexed |
| `/agent/run` | POST | Run the agent on a message |
| `/agent/reset` | POST | Clear agent memory |
| `/agent/tools` | GET | List available tools |
| `/voice/transcribe` | POST | Speech-to-text only |
| `/voice/process` | POST | Full voice pipeline (audio in, audio out) |
| `/voice/reset` | POST | Clear conversation history |
| `/anti-rag/kag/extract` | POST | Extract entities/relations into knowledge graph |
| `/anti-rag/kag/query` | POST | Query via graph traversal (KAG) |
| `/anti-rag/kag/stats` | GET | Knowledge graph statistics |
| `/anti-rag/sql/query` | POST | Text-to-SQL query |
| `/anti-rag/sql/schema` | PUT | Update the database schema |
| `/anti-rag/route` | POST | Show which strategy the router picks |
| `/anti-rag/fine-tuning/generate-data` | POST | Generate synthetic training data |
| `/anti-rag/fine-tuning/guide` | GET | Fine-tuning vs RAG decision framework |
| `/anti-rag/fine-tuning/lora-config` | POST | Recommended LoRA config for model+task |
| `/harness/evaluate` | POST | Evaluate a response (LLM-as-judge) |
| `/harness/evaluate/batch` | POST | Batch evaluation + regression report |
| `/harness/tracer/stats` | GET | Latency, token, cost statistics |
| `/harness/tracer/recent` | GET | Recent request traces |
| `/harness/prompts/register` | POST | Register a prompt version |
| `/harness/prompts/{name}` | GET | Get active prompt version |

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│  FastAPI    │  ← app/main.py (routing, middleware, error handling)
└─────┬───────┘
      │
   ┌──┴──────────────────────────────────┐
   │                                     │
   ▼                                     ▼
┌──────────┐                     ┌───────────────┐
│  RAG     │                     │    Agent      │
│ Pipeline │                     │  (tool loop)  │
└──┬───────┘                     └───────┬───────┘
   │                                     │
   ▼                                     ▼
┌──────────┐   ┌───────────┐   ┌─────────────────┐
│ Vector   │   │ Embedding │   │  Tools          │
│ Store    │   │ Model     │   │  (calc/search)  │
└──────────┘   └───────────┘   └─────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Claude API │
              │ (Anthropic) │
              └─────────────┘
```

---

## Deployment

### Option 1: Railway (recommended for beginners)
1. Push this repo to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Set your environment variables in the Railway dashboard
4. Railway auto-deploys on every push to main ✅

### Option 2: Render
1. Go to https://render.com → New Web Service → Connect GitHub
2. Set Build Command: `pip install -r requirements.txt`
3. Set Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables and deploy ✅

### Option 3: Docker
```bash
docker build -t ai-bootcamp .
docker run -p 8000:8000 --env-file .env ai-bootcamp
```

---

## Common Issues

**`ANTHROPIC_API_KEY` not found** — Make sure you copied `.env.example` to `.env` and filled in your key.

**`ImportError: No module named 'sentence_transformers'`** — Run `pip install sentence-transformers` or switch to `EMBEDDING_PROVIDER=openai` in `.env`.

**`No results above similarity threshold`** — Try lowering `SIMILARITY_THRESHOLD` in `.env` (e.g. to `0.5`) or add more documents to the knowledge base.

**Voice pipeline returns 500** — Check that `DEEPGRAM_API_KEY` and `ELEVENLABS_API_KEY` are set in `.env`.

---

## Contributing

Found a bug or want to improve the starter code? Open an issue or PR on GitHub.

---

*Built with ❤️ for the BlockseBlock AI Engineering Bootcamp*
