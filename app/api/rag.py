"""
api/rag.py — RAG API routes.

Endpoints:
  POST /rag/ingest   — Add text or a file to the knowledge base
  POST /rag/query    — Ask a question against the knowledge base
  GET  /rag/status   — Check how many chunks are indexed
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-initialised pipeline (created on first request)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


# ── Request / Response models ─────────────────────────────────────────────────
class IngestTextRequest(BaseModel):
    text: str = Field(..., description="The document text to add to the knowledge base")
    source: str = Field("api", description="A label for this document, e.g. a filename or URL")


class IngestResponse(BaseModel):
    chunks_added: int
    source: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the knowledge base")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    response: str
    sources: list[str]
    scores: list[float]
    llm_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int


class StatusResponse(BaseModel):
    chunks_indexed: int
    vector_store: str


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/ingest", response_model=IngestResponse, summary="Add text to knowledge base")
async def ingest_text(request: IngestTextRequest):
    """
    Chunk, embed, and store a piece of text in the vector database.

    The text will be available for retrieval in subsequent /rag/query calls.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")

    pipeline = get_pipeline()
    try:
        chunks_added = pipeline.ingest_text(request.text, source=request.source)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(chunks_added=chunks_added, source=request.source)


@router.post("/ingest/file", response_model=IngestResponse, summary="Upload a .txt file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a plain-text (.txt) file and add it to the knowledge base.

    STUDENT TODO: extend this to accept .pdf files using PyPDF2.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported for now.")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    pipeline = get_pipeline()
    try:
        chunks_added = pipeline.ingest_text(text, source=file.filename)
    except Exception as exc:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(chunks_added=chunks_added, source=file.filename)


@router.post("/query", response_model=QueryResponse, summary="Ask the knowledge base")
async def query(request: QueryRequest):
    """
    Retrieve relevant chunks and generate a grounded answer using Claude.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")

    pipeline = get_pipeline()
    try:
        result = pipeline.query(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        response=result.response,
        sources=result.sources,
        scores=result.scores,
        llm_latency_ms=result.llm_latency_ms,
        total_latency_ms=result.total_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@router.get("/status", response_model=StatusResponse, summary="Knowledge base status")
async def status():
    """Returns how many chunks are currently indexed."""
    from app.config import settings
    pipeline = get_pipeline()
    return StatusResponse(
        chunks_indexed=pipeline._vector_store.count()
            if hasattr(pipeline._vector_store, "count") else -1,
        vector_store=settings.vector_store,
    )
