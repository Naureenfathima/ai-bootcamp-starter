# AI Bootcamp Starter 🚀

> Starter project for the **BlockseBlock 30-Day AI Systems Engineering Bootcamp**
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
│   ├── voice/           # Module 4 — Voice AI
│   │   ├── stt.py         speech-to-text (Deepgram)
│   │   ├── tts.py         text-to-speech (ElevenLabs)
│   │   └── pipeline.py    full voice pipeline
│   ├── agent/           # Module 4 — Autonomous Agent
│   │   ├── tools.py       tool definitions (calculator, time, web search)
│   │   ├── memory.py      working + episodic memory
│   │   └── agent.py       agentic loop with tool use
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

### Module 4 — Voice AI & Agents
- Read `app/voice/pipeline.py` and `app/agent/agent.py`
- Add your Deepgram and ElevenLabs keys to `.env` and test `/voice/process`
- Test the agent at `/agent/run` with: `{"message": "What is sqrt(144) + the current time?"}`
- **Your task:** Add a new tool to `app/agent/tools.py` (ideas: web search, weather, RAG lookup)

### Module 5 — MCP & Production
- Read `app/mcp/server.py` and register your RAG and Agent as MCP methods
- Containerise: `docker build -t ai-bootcamp . && docker run -p 8000:8000 --env-file .env ai-bootcamp`
- Push to Railway or Render and get a public URL
- **Your task:** Uncomment the deploy step in `.github/workflows/ci.yml`

### Module 6 — Capstone
- Choose your path: **RAG System**, **Voice AI Assistant**, or **Autonomous Agent**
- Extend this starter into a complete, deployed product
- Your capstone README must include: system diagram, tech stack, setup guide, public URL

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
