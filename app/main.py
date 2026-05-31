"""
main.py — FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000

Production (Docker):
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.api.agent import router as agent_router
from app.api.voice import router as voice_router
from app.api.anti_rag import router as anti_rag_router
from app.api.harness import router as harness_router
from app.api.eval import router as eval_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run setup on startup and cleanup on shutdown."""
    logger.info("🚀  Starting %s v%s", settings.app_name, settings.app_version)
    # TODO: initialise database connections, warm up models, etc.
    yield
    logger.info("👋  Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Starter project for the BlockseBlock AI Systems Engineering Bootcamp. "
        "Covers RAG, Anti-RAG (CAG, KAG, Text-to-SQL, fine-tuning), "
        "Voice AI, Autonomous Agents with memory, Evaluation Harness, "
        "and Production patterns (caching, rate limiting, resilience, multi-tenancy)."
    ),
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "%s %s → %s  (%s ms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ── Global Error Handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(rag_router,       prefix="/rag",       tags=["RAG"])
app.include_router(agent_router,     prefix="/agent",     tags=["Agent"])
app.include_router(voice_router,     prefix="/voice",     tags=["Voice"])
app.include_router(anti_rag_router,  prefix="/anti-rag",  tags=["Anti-RAG"])
app.include_router(harness_router,   prefix="/harness",   tags=["Agent Harness"])
app.include_router(eval_router,      prefix="/eval",      tags=["Evaluation"])
