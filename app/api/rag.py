"""
api/rag.py — RAG API routes.

Endpoints (standard RAG):
  POST /rag/ingest          — Add text to the knowledge base
  POST /rag/ingest/file     — Upload a .txt file
  POST /rag/query           — Ask a question
  GET  /rag/status          — Chunk count and vector store type

Endpoints (chunking):
  POST /rag/chunking/compare — Run all four strategies on a text, return chunk counts

Endpoints (metadata):
  POST /rag/metadata/extract — Extract structured metadata from a text or file path

Endpoints (page index — two-tier retrieval):
  POST /rag/page-index/build  — Build a hierarchical page index from text
  POST /rag/page-index/query  — Query using the page index
  GET  /rag/page-index/toc    — Return table of contents

Endpoints (agentic RAG):
  POST /rag/agentic/query    — Query with decompose → retrieve → reflect loop

Endpoints (graph RAG):
  POST /rag/graph-rag/build  — Build entity graph and community summaries
  POST /rag/graph-rag/query  — Query via community summaries or chunk retrieval
  GET  /rag/graph-rag/stats  — Entity/relation/community counts

Endpoints (corrective RAG):
  POST /rag/corrective/query — Query with grade → correct → refine → evaluate loop
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Lazy-initialised pipeline singletons ─────────────────────────────────────
_pipeline      = None
_page_index    = None
_agentic_rag   = None
_graph_rag     = None
_corrective_rag = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.rag.pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


def get_page_index():
    global _page_index
    if _page_index is None:
        from app.rag.page_index import PageIndex
        from app.config import settings
        _page_index = PageIndex(page_size_words=settings.page_index_page_size)
    return _page_index


def get_agentic_rag():
    global _agentic_rag
    if _agentic_rag is None:
        from app.rag.agentic_rag import AgenticRAGPipeline
        from app.config import settings
        _agentic_rag = AgenticRAGPipeline(
            rag=get_pipeline(),
            max_iterations=settings.agentic_max_iterations,
            reflection_threshold=settings.agentic_reflection_threshold,
        )
    return _agentic_rag


def get_graph_rag():
    global _graph_rag
    if _graph_rag is None:
        from app.rag.graph_rag import GraphRAGPipeline
        _graph_rag = GraphRAGPipeline(rag=get_pipeline())
    return _graph_rag


def get_corrective_rag():
    global _corrective_rag
    if _corrective_rag is None:
        from app.rag.corrective_rag import CorrectiveRAGPipeline
        _corrective_rag = CorrectiveRAGPipeline(rag=get_pipeline())
    return _corrective_rag


# ── Request / Response models ─────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str   = Field(..., description="Document text to add to the knowledge base")
    source: str = Field("api", description="Label for this document (filename, URL, etc.)")


class IngestResponse(BaseModel):
    chunks_added: int
    source: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the knowledge base")
    top_k: int    = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


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


# ── Standard RAG ──────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Add text to knowledge base")
async def ingest_text(request: IngestTextRequest):
    """Chunk, embed, and store text in the vector database."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    try:
        chunks_added = get_pipeline().ingest_text(request.text, source=request.source)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(chunks_added=chunks_added, source=request.source)


@router.post("/ingest/file", response_model=IngestResponse, summary="Upload a .txt file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a plain-text file and add it to the knowledge base.
    STUDENT TODO: extend to .pdf using pypdf (pip install pypdf).
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    try:
        chunks_added = get_pipeline().ingest_text(text, source=file.filename)
    except Exception as exc:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(chunks_added=chunks_added, source=file.filename)


@router.post("/query", response_model=QueryResponse, summary="Ask the knowledge base")
async def query(request: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer using Claude."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_pipeline().query(request.question, top_k=request.top_k)
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
    """How many chunks are indexed and which vector store is active."""
    from app.config import settings
    pipeline = get_pipeline()
    return StatusResponse(
        chunks_indexed=pipeline._vector_store.count()
            if hasattr(pipeline._vector_store, "count") else -1,
        vector_store=settings.vector_store,
    )


# ── Chunking ──────────────────────────────────────────────────────────────────

class ChunkingCompareRequest(BaseModel):
    text: str   = Field(..., description="Text to chunk")
    source: str = Field("compare", description="Source label")


@router.post("/chunking/compare", summary="Compare all four chunking strategies")
async def chunking_compare(request: ChunkingCompareRequest):
    """
    Run fixed, recursive, sentence, and semantic chunking on the same text
    and return chunk counts and first chunk previews for comparison.

    Semantic chunking requires an embedding API call — if no API key is
    configured it will return an error for that strategy only.

    Systems lesson: benchmark different strategies on YOUR data before
    committing to one in production.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")

    from app.rag.chunking import get_chunker

    results = {}
    for strategy in ("fixed", "recursive", "sentence"):
        try:
            chunker = get_chunker(strategy)
            chunks = chunker.chunk(request.text, source=request.source)
            results[strategy] = {
                "chunk_count": len(chunks),
                "avg_words": round(
                    sum(len(c.text.split()) for c in chunks) / max(len(chunks), 1), 1
                ),
                "first_chunk_preview": chunks[0].text[:200] if chunks else "",
            }
        except Exception as exc:
            results[strategy] = {"error": str(exc)}

    # Semantic is expensive — try it but catch gracefully
    try:
        chunker = get_chunker("semantic")
        chunks = chunker.chunk(request.text, source=request.source)
        results["semantic"] = {
            "chunk_count": len(chunks),
            "avg_words": round(
                sum(len(c.text.split()) for c in chunks) / max(len(chunks), 1), 1
            ),
            "first_chunk_preview": chunks[0].text[:200] if chunks else "",
        }
    except Exception as exc:
        results["semantic"] = {
            "error": str(exc),
            "note": "Semantic chunking requires an embedding API call.",
        }

    return {
        "input_word_count": len(request.text.split()),
        "strategies": results,
        "teaching_note": (
            "Fixed: fast, ignores structure. "
            "Recursive: respects paragraphs. "
            "Sentence: never cuts sentences. "
            "Semantic: splits at topic changes (needs API)."
        ),
    }


# ── Metadata ──────────────────────────────────────────────────────────────────

class MetadataRequest(BaseModel):
    text: str       = Field(..., description="Raw document text")
    source: str     = Field("api", description="Document identifier")
    doc_type: str   = Field("text", description="'text' or 'markdown'")


@router.post("/metadata/extract", summary="Extract document metadata")
async def extract_metadata(request: MetadataRequest):
    """
    Extract structured metadata (title, author, date, language, word count)
    from a text string using heuristic parsing.

    For Markdown, detects YAML front matter and H1 headings.
    For plain text, detects common patterns like 'Author:' and ISO dates.
    """
    from app.rag.metadata import MetadataExtractor
    extractor = MetadataExtractor()
    meta = extractor.extract_from_string(
        request.text, source=request.source, doc_type=request.doc_type
    )
    return meta.to_dict()


# ── Page Index ────────────────────────────────────────────────────────────────

class PageIndexBuildRequest(BaseModel):
    text: str   = Field(..., description="Full document text to index")
    source: str = Field("document", description="Document identifier")


class PageIndexQueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer using the page index")


@router.post("/page-index/build", summary="Build a hierarchical page index")
async def page_index_build(request: PageIndexBuildRequest):
    """
    Split the document into pages, LLM-summarise each page, and chunk within pages.

    This is the build step — run once per document. Subsequent queries against
    the page index use two-tier retrieval (page summaries → chunks).

    Teaching demo:
      1. POST /rag/page-index/build  with a long document
      2. GET  /rag/page-index/toc    to see the page summaries
      3. POST /rag/page-index/query  to ask questions
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    try:
        stats = get_page_index().build_from_text(request.text, source=request.source)
    except Exception as exc:
        logger.exception("PageIndex build failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {**stats, "next": "POST /rag/page-index/query or GET /rag/page-index/toc"}


@router.post("/page-index/query", summary="Query using two-tier page index")
async def page_index_query(request: PageIndexQueryRequest):
    """
    Two-tier retrieval: find relevant pages via summary similarity, then
    retrieve chunks within those pages for precise context.

    Response includes pages_searched so you can see which pages were selected.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_page_index().query(request.question)
    except Exception as exc:
        logger.exception("PageIndex query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "answer": result.answer,
        "pages_searched": result.pages_searched,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "total_latency_ms": result.total_latency_ms,
    }


@router.get("/page-index/toc", summary="Return page-level table of contents")
async def page_index_toc():
    """Return each page's number, word count, and LLM-generated summary."""
    toc = get_page_index().table_of_contents()
    if not toc:
        return {"message": "No documents indexed. POST /rag/page-index/build first.", "pages": []}
    return {"pages": toc, "total": len(toc)}


# ── Agentic RAG ───────────────────────────────────────────────────────────────

class AgenticQueryRequest(BaseModel):
    question: str = Field(..., description="Question for the agentic RAG pipeline")


@router.post("/agentic/query", summary="Agentic RAG: decompose → retrieve → reflect")
async def agentic_query(request: AgenticQueryRequest):
    """
    Agentic RAG pipeline:
      1. Decompose complex questions into sub-questions
      2. Retrieve evidence for each sub-question
      3. Self-reflect: is evidence sufficient? (LLM-scored)
      4. Rewrite and retry if score is below threshold (up to max_iterations)
      5. Synthesise a final answer from all collected evidence

    The reasoning_trace shows every step the agent took — use it for teaching.
    Compare token usage vs standard /rag/query to show the cost/quality tradeoff.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_agentic_rag().query(request.question)
    except Exception as exc:
        logger.exception("Agentic RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "answer": result.answer,
        "sub_questions": [
            {
                "question": sq.question,
                "evidence_count": len(sq.evidence),
                "retrieval_score": sq.retrieval_score,
                "iterations": sq.iterations_used,
            }
            for sq in result.sub_questions
        ],
        "reasoning_trace": result.reasoning_trace,
        "total_iterations": result.total_iterations,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_latency_ms": result.total_latency_ms,
    }


# ── Graph RAG ─────────────────────────────────────────────────────────────────

class GraphRAGBuildRequest(BaseModel):
    text: str   = Field(..., description="Document text to build the entity graph from")
    source: str = Field("document", description="Document identifier")


class GraphRAGQueryRequest(BaseModel):
    question: str   = Field(..., description="Question to answer")
    query_type: str = Field("auto", description="'global' | 'local' | 'auto'")


@router.post("/graph-rag/build", summary="Build entity graph and community summaries")
async def graph_rag_build(request: GraphRAGBuildRequest):
    """
    Build a GraphRAG index:
      1. Extract entities and relations from each chunk (LLM)
      2. Cluster entities into communities (connected components)
      3. Summarise each community (LLM)

    This is the offline build step. Run once per document set.
    Community summaries enable global synthesis queries.

    Note: this makes many LLM calls (one per chunk + one per community).
    For a 10-chunk document, expect ~15-20 API calls.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    try:
        stats = get_graph_rag().build(request.text, source=request.source)
    except Exception as exc:
        logger.exception("GraphRAG build failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return stats


@router.post("/graph-rag/query", summary="Graph RAG: global synthesis or local fact lookup")
async def graph_rag_query(request: GraphRAGQueryRequest):
    """
    GraphRAG query with automatic routing:
      - Global queries ("summarise all themes", "what are the main patterns?")
        → use community summaries as context
      - Local queries (specific entities, facts)
        → use standard chunk retrieval

    Set query_type="auto" to let the pipeline classify the query.

    Teaching demo: run the same question with query_type="global" then "local"
    to see different answers from different retrieval strategies.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    if request.query_type not in ("global", "local", "auto"):
        raise HTTPException(status_code=400, detail="query_type must be 'global', 'local', or 'auto'")
    try:
        result = get_graph_rag().query(request.question, query_type=request.query_type)
    except Exception as exc:
        logger.exception("GraphRAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "answer": result.answer,
        "query_type": result.query_type,
        "communities_used": result.communities_used,
        "entities_found": result.entities_found,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
    }


@router.get("/graph-rag/stats", summary="Graph entity and community statistics")
async def graph_rag_stats():
    """Return entity count, relation count, community count, and type breakdown."""
    return get_graph_rag().graph_stats()


# ── Corrective RAG ────────────────────────────────────────────────────────────

class CorrectiveRAGRequest(BaseModel):
    question: str = Field(..., description="Question for the corrective RAG pipeline")
    top_k: int    = Field(3, ge=1, le=10, description="Number of chunks to initially retrieve")


@router.post("/corrective/query", summary="CRAG: retrieve → grade → correct → evaluate")
async def corrective_query(request: CorrectiveRAGRequest):
    """
    Corrective RAG pipeline:
      1. Retrieve chunks (standard vector search)
      2. Grade each chunk: CORRECT / AMBIGUOUS / INCORRECT
      3. Apply correction strategy:
           All CORRECT   → use as-is
           Any AMBIGUOUS → supplement with web search (stub)
           All INCORRECT → web search only (stub)
      4. Refine context (remove irrelevant sentences)
      5. Generate answer
      6. Evaluate faithfulness + relevance (LLM-as-judge)

    Key teaching fields:
      graded_documents  — see exactly what grade each retrieved chunk received
      correction_strategy — which path was taken (use_asis / supplement_web / web_only)
      eval_score         — 0-1 quality score from the LLM evaluator

    Systems lesson: CRAG detects retrieval failures rather than silently
    generating hallucinations. Compare eval_score with standard /rag/query.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_corrective_rag().query(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Corrective RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "answer": result.answer,
        "correction_strategy": result.correction_strategy,
        "web_search_used": result.web_search_used,
        "knowledge_refined": result.knowledge_refined,
        "eval_score": result.eval_score,
        "eval_passed": result.eval_passed,
        "graded_documents": [
            {
                "source": g.source,
                "grade": g.grade.value,
                "reasoning": g.grade_reasoning,
                "similarity_score": round(g.similarity_score, 3),
            }
            for g in result.graded_documents
        ],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
    }
