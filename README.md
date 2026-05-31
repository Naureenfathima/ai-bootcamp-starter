# AI Bootcamp Starter

> **BlockseBlock AI Systems Engineering Bootcamp** — Instructor: Naureen Fathima

Every module of the course adds a layer to this project until you have a deployed, production-grade AI system with a public URL.

---

## Project Structure

```
app/
├── main.py              FastAPI entry point, middleware, error handling
├── config.py            All settings loaded from .env (pydantic-settings)
├── llm.py               Unified LLM client — swap providers via LLM_PROVIDER
├── rag/                 Module 3 — RAG pipeline + advanced variants
├── anti_rag/            Module 3 — When not to use RAG (KAG, CAG, SQL, fine-tuning)
├── voice/               Module 4 — STT → LLM → TTS pipeline
├── agent/               Module 4 — Agentic loop with tools and four memory types
├── harness/             Module 5 — Agent harness: tasks, trajectory capture, scoring
├── eval/                Module 5 — RAG evaluation: LLM-as-judge, tracer, prompt registry
├── production/          Module 5 — Cache, rate limiter, resilience, cost, multi-tenancy
├── mcp/                 Module 5 — Model Context Protocol server
└── api/                 FastAPI route handlers for each module
```

Each module folder has its own **README.md** explaining the concepts and what to implement.

---

## Quick Start

```bash
git clone https://github.com/naureenfathima/ai-bootcamp-starter.git
cd ai-bootcamp-starter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in your keys

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive API.

**Zero-cost dev mode** — set these in `.env` and no paid API keys are needed:
```
EMBEDDING_PROVIDER=local
VECTOR_STORE=memory
```

---

## API Keys

| Key | Where to get it | Required for |
|-----|----------------|-------------|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/ | Everything (scoring, eval, agent) |
| `OPENAI_API_KEY` | https://platform.openai.com/ | OpenAI embeddings or chat (optional) |
| `DEEPGRAM_API_KEY` | https://console.deepgram.com/ | Voice: speech-to-text |
| `ELEVENLABS_API_KEY` | https://elevenlabs.io/ | Voice: text-to-speech |

---

## Local LLM with Ollama

Run the agent and RAG pipeline entirely offline — no API costs, no rate limits.

**Install Ollama and pull the default model:**
```bash
# macOS
brew install ollama
ollama serve &          # starts the server at localhost:11434
ollama pull qwen2.5:7b  # ~4.7 GB — good balance of speed and quality on M1/M2
```

**`.env` settings (already configured if you copied `.env.example`):**
```
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

**Remote Ollama via SSH tunnel** — if Ollama runs on a server you access over SSH:
```bash
# Run this once in a terminal — keeps the tunnel open
ssh -L 11434:localhost:11434 user@your-server-ip

# Add your public key to the server for passwordless access:
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@your-server-ip
```
Then use `OLLAMA_BASE_URL=http://localhost:11434/v1` as normal — the tunnel forwards the remote port.

**Lighter model for 8 GB RAM:**
```bash
ollama pull llama3.2:3b  # ~2 GB, faster iteration
# then set LLM_MODEL=llama3.2:3b in .env
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run without real API keys — external services are stubbed.

---

## Module Build Guide

### Module 2 — AI Systems Foundations
- Understand `app/config.py` (settings from `.env`) and `app/main.py` (middleware)
- Read `app/llm.py` — the unified client that lets you swap providers without touching other code
- **Task:** Add a logging statement that records token counts for every LLM call

### Module 3 — RAG Systems
Read `app/rag/` in this order: `ingestion.py` → `embeddings.py` → `retrieval.py` → `pipeline.py` → `chunking.py` → `metadata.py` → `page_index.py` → `agentic_rag.py` → `corrective_rag.py` → `graph_rag.py`

Then `app/anti_rag/`: `kag.py` → `cag.py` → `structured_knowledge.py` → `fine_tuning.py` → `router.py`

- **Task:** Replace `InMemoryVectorStore` with `PgVectorStore` using pgvector

### Module 4 — Voice AI & Agents with Memory
- Read `app/voice/pipeline.py` — the STT → LLM → TTS flow and latency budget
- Read `app/agent/agent.py` and `app/agent/memory.py` — the agentic loop and four memory types
- Test: `POST /agent/run {"message": "What is sqrt(144) + the current time?"}`
- **Task:** Implement `EpisodicMemory._summarise_oldest()` (code template in the docstring)

### Module 5 — Agent Harness + Evaluation
- Read `app/harness/` — task runner, trajectory capture, three-dimension scoring
- Test: `GET /harness/tasks` → `POST /harness/run/task {"task_id": "<id>"}` → `POST /harness/run/suite`
- Read `app/eval/` — LLM-as-judge for RAG, request tracer, prompt registry
- Test: `POST /eval/evaluate` with a (question, context, answer) triple
- **Task:** Write 5 domain-specific tasks in `app/harness/tasks.py` and run the full suite

### Module 5 — Production Patterns + MCP
- Read every file in `app/production/` — each covers one production concern
- Read `app/mcp/server.py` and register your RAG and Agent as MCP methods
- Containerise: `docker build -t ai-bootcamp . && docker run -p 8000:8000 --env-file .env ai-bootcamp`
- **Task:** Uncomment the deploy step in `.github/workflows/ci.yml` and get a public URL

### Module 6 — Capstone
Choose: **RAG System**, **Voice AI Assistant**, or **Autonomous Agent**. Extend this starter into a deployed product. Your capstone README must include: system diagram, tech stack, setup guide, public URL.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/rag/ingest` | POST | Add text to the knowledge base |
| `/rag/ingest/file` | POST | Upload a .txt file |
| `/rag/query` | POST | Ask a question against the knowledge base |
| `/rag/status` | GET | Number of chunks indexed |
| `/rag/agentic/query` | POST | Agentic RAG — iterative retrieve/reflect loop |
| `/rag/corrective/query` | POST | Corrective RAG — grade docs, web fallback |
| `/rag/graph/build` | POST | Build GraphRAG entity graph from text |
| `/rag/graph/query` | POST | Query via GraphRAG (local or global) |
| `/rag/page-index/build` | POST | Build hierarchical page index |
| `/rag/page-index/query` | POST | Two-tier page → chunk retrieval |
| `/agent/run` | POST | Run the agent on a message |
| `/agent/reset` | POST | Clear agent memory |
| `/agent/tools` | GET | List available tools |
| `/voice/transcribe` | POST | Speech-to-text only |
| `/voice/process` | POST | Full voice pipeline (audio in, audio out) |
| `/voice/reset` | POST | Clear conversation history |
| `/anti-rag/demo/seed` | POST | Seed demo org-chart into KAG + CAG |
| `/anti-rag/kag/query` | POST | Graph traversal query (KAG) |
| `/anti-rag/cag/query` | POST | Full context window query (CAG) |
| `/anti-rag/compare` | POST | Side-by-side KAG vs CAG |
| `/anti-rag/sql/query` | POST | Text-to-SQL query |
| `/anti-rag/route` | POST | Show which strategy the router picks |
| `/anti-rag/fine-tuning/guide` | GET | Fine-tuning vs RAG decision framework |
| `/harness/tasks` | GET | List tasks in the default agent test suite |
| `/harness/run/task` | POST | Run one task by ID — full trajectory + score breakdown |
| `/harness/run/suite` | POST | Run all tasks — returns GREEN / YELLOW / RED report |
| `/harness/run/custom` | POST | Define and run a one-off task inline |
| `/eval/evaluate` | POST | Evaluate a RAG response (LLM-as-judge) |
| `/eval/evaluate/batch` | POST | Batch RAG evaluation + regression report |
| `/eval/tracer/stats` | GET | Latency, token, cost statistics |
| `/eval/prompts/register` | POST | Register a prompt version |

---

## Common Issues

| Error | Fix |
|-------|-----|
| `ANTHROPIC_API_KEY not found` | Copy `.env.example` to `.env` and fill in your key |
| `No module named 'sentence_transformers'` | `pip install sentence-transformers` or set `EMBEDDING_PROVIDER=openai` |
| `No results above similarity threshold` | Lower `SIMILARITY_THRESHOLD` in `.env` to `0.5` |
| Voice pipeline 500 error | Check `DEEPGRAM_API_KEY` and `ELEVENLABS_API_KEY` in `.env` |

---

## Deployment

**Railway (recommended):** Push to GitHub → railway.app → New Project → Deploy from GitHub → set env vars → auto-deploys on push.

**Render:** New Web Service → Build: `pip install -r requirements.txt` → Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Docker:** `docker build -t ai-bootcamp . && docker run -p 8000:8000 --env-file .env ai-bootcamp`
