"""
api/anti_rag.py — Anti-RAG API routes.

Endpoints:
  POST /anti-rag/kag/extract     — Extract entities/relations into knowledge graph
  POST /anti-rag/kag/query       — Query via graph traversal (KAG)
  GET  /anti-rag/kag/stats       — Knowledge graph statistics

  POST /anti-rag/cag/ingest      — Add a document to the CAG knowledge base
  POST /anti-rag/cag/query       — Query via full context window (CAG)
  GET  /anti-rag/cag/stats       — CAG knowledge base statistics

  POST /anti-rag/sql/query       — Text-to-SQL query
  PUT  /anti-rag/sql/schema      — Update the database schema

  POST /anti-rag/route           — Show routing decision for a query
  POST /anti-rag/compare         — Run same question through KAG + CAG side by side
  POST /anti-rag/demo/seed       — Pre-load demo data into KAG and CAG (no API call needed)

  POST /anti-rag/fine-tuning/generate-data — Generate synthetic training data
  GET  /anti-rag/fine-tuning/guide         — Fine-tuning decision framework
  POST /anti-rag/fine-tuning/lora-config   — Recommended LoRA config
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-initialised pipelines (one shared instance per process)
_kag_pipeline = None
_cag_pipeline = None
_sql_pipeline = None
_router       = None
_ft_guide     = None
_ft_builder   = None


def get_kag():
    global _kag_pipeline
    if _kag_pipeline is None:
        from app.anti_rag.kag import KAGPipeline
        _kag_pipeline = KAGPipeline()
    return _kag_pipeline


def get_cag():
    global _cag_pipeline
    if _cag_pipeline is None:
        from app.anti_rag.cag import CAGPipeline
        _cag_pipeline = CAGPipeline()
    return _cag_pipeline


def get_sql():
    global _sql_pipeline
    if _sql_pipeline is None:
        from app.anti_rag.structured_knowledge import StructuredKnowledgePipeline
        _sql_pipeline = StructuredKnowledgePipeline()
    return _sql_pipeline


def get_router():
    global _router
    if _router is None:
        from app.anti_rag.router import HybridStrategyRouter
        _router = HybridStrategyRouter()
    return _router


def get_ft_guide():
    global _ft_guide
    if _ft_guide is None:
        from app.anti_rag.fine_tuning import FineTuningGuide
        _ft_guide = FineTuningGuide()
    return _ft_guide


def get_ft_builder():
    global _ft_builder
    if _ft_builder is None:
        from app.anti_rag.fine_tuning import TrainingDataBuilder
        _ft_builder = TrainingDataBuilder()
    return _ft_builder


# ── KAG endpoints ─────────────────────────────────────────────────────────────

class KAGExtractRequest(BaseModel):
    text: str = Field(..., description="Text to extract entities and relations from")


class KAGQueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer using the knowledge graph")
    max_hops: int = Field(2, ge=1, le=4, description="Maximum graph traversal hops")


@router.post("/kag/extract", summary="Extract entities into knowledge graph")
async def kag_extract(request: KAGExtractRequest):
    """
    Parse text and extract named entities and relationships into the in-memory
    knowledge graph using Claude.

    Example text:
        "Alice manages the Platform team. Platform team owns the RAG Service.
         RAG Service depends on the Vector Database."
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    try:
        return get_kag().extract_and_store(request.text)
    except Exception as exc:
        logger.exception("KAG extract failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/kag/query", summary="Query the knowledge graph")
async def kag_query(request: KAGQueryRequest):
    """
    Answer a question by traversing the knowledge graph instead of using
    vector similarity search (the core KAG vs RAG distinction).

    The response includes a reasoning_path so students can see which graph
    edges were traversed to reach the answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_kag().query(request.question, max_hops=request.max_hops)
        return {
            "answer":         result.answer,
            "entities_used":  result.entities_used,
            "reasoning_path": result.reasoning_path,
            "input_tokens":   result.input_tokens,
            "output_tokens":  result.output_tokens,
        }
    except Exception as exc:
        logger.exception("KAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/kag/stats", summary="Knowledge graph statistics")
async def kag_stats():
    """Returns the number of entities and relations in the knowledge graph."""
    return get_kag().graph_stats()


# ── CAG endpoints ─────────────────────────────────────────────────────────────

class CAGIngestRequest(BaseModel):
    text:   str = Field(..., description="Full text of the document to add")
    source: str = Field("unknown", description="Document label (filename, URL, etc.)")


class CAGQueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer from the full document context")


@router.post("/cag/ingest", summary="Add a document to the CAG knowledge base")
async def cag_ingest(request: CAGIngestRequest):
    """
    Add a document to the Cache-Augmented Generation knowledge base.

    Unlike RAG, there is no chunking or embedding step — the full text is stored
    and will be included in the context of every future query.

    Watch the context_window_pct field to understand how much of the model's
    context window your knowledge base is consuming.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    try:
        return get_cag().ingest(request.text, source=request.source)
    except Exception as exc:
        logger.exception("CAG ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/cag/query", summary="Query using full context window (CAG)")
async def cag_query(request: CAGQueryRequest):
    """
    Answer a question by loading ALL ingested documents into a single context window.

    Key things to observe in the response:
      - cache_creation_input_tokens: non-zero on the FIRST query (cache is being built)
      - cache_read_input_tokens: non-zero on SUBSEQUENT queries (cache hit, ~90% cheaper)
      - context_window_pct: how much of the 200K token window the documents consume

    Compare this with KAG:
      - KAG: traverses a graph — fast, precise, but only knows what was explicitly extracted
      - CAG: reads everything — slower first call, but never misses implicit connections
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_cag().query(request.question)
        return {
            "answer":                      result.answer,
            "documents_used":              result.documents_used,
            "estimated_context_tokens":    result.estimated_context_tokens,
            "context_window_pct":          result.context_window_pct,
            "input_tokens":                result.input_tokens,
            "output_tokens":               result.output_tokens,
            "cache_creation_input_tokens": result.cache_creation_input_tokens,
            "cache_read_input_tokens":     result.cache_read_input_tokens,
        }
    except Exception as exc:
        logger.exception("CAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/cag/stats", summary="CAG knowledge base statistics")
async def cag_stats():
    """Returns document count, total size, and context window utilisation."""
    return get_cag().stats()


# ── SQL endpoints ─────────────────────────────────────────────────────────────

class SQLQueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about your data")


class SQLSchemaRequest(BaseModel):
    schema: str = Field(..., description="SQL schema DDL (CREATE TABLE statements)")


@router.post("/sql/query", summary="Text-to-SQL query")
async def sql_query(request: SQLQueryRequest):
    """
    Translate a natural language question into SQL, execute it (stub),
    and return an LLM-interpreted plain-English answer.

    Example: "How many orders were placed this month?"
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")
    try:
        result = get_sql().query(request.question)
        return {
            "answer":          result.answer,
            "query_generated": result.query_generated,
            "raw_results":     result.raw_results,
            "query_type":      result.query_type,
        }
    except Exception as exc:
        logger.exception("SQL query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/sql/schema", summary="Update database schema")
async def sql_update_schema(request: SQLSchemaRequest):
    """Update the SQL schema the Text-to-SQL pipeline reasons over."""
    get_sql().update_schema(request.schema)
    return {"status": "ok", "message": "Schema updated."}


# ── Router endpoint ───────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    query: str = Field(..., description="The question you want to route")


@router.post("/route", summary="Show strategy routing decision")
async def route_query(request: RouteRequest):
    """
    Show which retrieval strategy (KAG / CAG / SQL / RAG / direct) the hybrid
    router would select for a given query, and why.

    Useful for teaching students to understand the routing logic and tradeoffs.
    """
    return get_router().explain(request.query)


# ── Compare endpoint ──────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    question: str = Field(..., description="Question to run through both KAG and CAG")
    max_hops: int = Field(2, ge=1, le=4, description="Max graph hops for KAG")


@router.post("/compare", summary="Run same question through KAG and CAG side by side")
async def compare_strategies(request: CompareRequest):
    """
    Run the same question through both KAG (graph traversal) and CAG (full context)
    and return the results side by side.

    This is the core classroom comparison endpoint. Call /demo/seed first to
    pre-populate both pipelines with identical data, then ask the same question
    here to see:

      KAG: reasoning_path shows which graph edges were traversed
      CAG: cache_creation/read tokens show the caching benefit
      Both: answer quality, token cost, approach tradeoffs

    Demo questions that show the contrast well:
      - "Who manages the team that owns the RAG Service?"  (KAG wins: multi-hop)
      - "List everything owned by each team."              (CAG wins: synthesis)
      - "What infrastructure does the RAG Service use?"    (both work equally)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")

    kag_result = None
    cag_result = None
    kag_error  = None
    cag_error  = None

    try:
        r = get_kag().query(request.question, max_hops=request.max_hops)
        kag_result = {
            "answer":         r.answer,
            "entities_used":  r.entities_used,
            "reasoning_path": r.reasoning_path,
            "input_tokens":   r.input_tokens,
            "output_tokens":  r.output_tokens,
        }
    except Exception as exc:
        logger.exception("Compare: KAG failed")
        kag_error = str(exc)

    try:
        r = get_cag().query(request.question)
        cag_result = {
            "answer":                      r.answer,
            "documents_used":              r.documents_used,
            "context_window_pct":          r.context_window_pct,
            "input_tokens":                r.input_tokens,
            "output_tokens":               r.output_tokens,
            "cache_creation_input_tokens": r.cache_creation_input_tokens,
            "cache_read_input_tokens":     r.cache_read_input_tokens,
        }
    except Exception as exc:
        logger.exception("Compare: CAG failed")
        cag_error = str(exc)

    return {
        "question": request.question,
        "kag": kag_result if kag_result is not None else {"error": kag_error},
        "cag": cag_result if cag_result is not None else {"error": cag_error},
        "teaching_note": (
            "KAG traverses explicit graph edges (fast, precise, misses implicit links). "
            "CAG reads all documents (slower first call, cached on repeat, never misses). "
            "Call this endpoint twice on the same question to observe CAG's cache_read_input_tokens."
        ),
    }


# ── Demo seed endpoint ────────────────────────────────────────────────────────

@router.post("/demo/seed", summary="Pre-load demo data into KAG and CAG")
async def demo_seed():
    """
    Pre-populate both KAG and CAG with identical org chart demo data.
    No API call required — data is added directly.

    Call this once at the start of a classroom demo, then use:
      POST /anti-rag/kag/query   — graph traversal
      POST /anti-rag/cag/query   — full context window
      POST /anti-rag/compare     — side-by-side comparison

    Demo knowledge base (same data in both pipelines):
      Alice (CTO) manages Bob (Backend Lead) and Carol (ML Lead).
      Bob leads the Platform Team, which owns the RAG Service and API Gateway.
      Carol leads the AI Team, which owns the Agent Service and Evaluation Harness.
      The RAG Service depends on VectorDB (PostgreSQL + pgvector).
      The Agent Service uses the RAG Service.

    Good demo questions:
      Single-hop: "What team does Bob lead?"
      Two-hop:    "What does Bob's team own?"
      Three-hop:  "What does Alice's backend team's main service depend on?"
      Synthesis:  "List everything owned by each team."  (CAG shines here)
      Multi-hop:  "Who manages the team that owns the RAG Service?"  (KAG shines)
    """
    kag_stats = get_kag().seed_demo_graph()
    cag_stats = get_cag().seed_demo_documents()

    return {
        "status": "seeded",
        "kag": kag_stats,
        "cag": cag_stats,
        "next_steps": [
            "POST /anti-rag/kag/query  {\"question\": \"What does Bob's team own?\"}",
            "POST /anti-rag/cag/query  {\"question\": \"What does Bob's team own?\"}",
            "POST /anti-rag/compare    {\"question\": \"Who manages the team that owns the RAG Service?\"}",
            "POST /anti-rag/route      {\"query\": \"List everything owned by each team\"}",
        ],
    }


# ── Fine-tuning endpoints ─────────────────────────────────────────────────────

class GenerateDataRequest(BaseModel):
    domain:       str = Field(..., description="Domain description, e.g. 'customer support for SaaS'")
    instruction:  str = Field(..., description="System instruction for the fine-tuned model")
    n_examples:   int = Field(10, ge=1, le=50, description="Number of examples to generate")
    dataset_name: str = Field("generated_dataset", description="Name for the dataset")


class LoRAConfigRequest(BaseModel):
    model_size: str = Field("7b", description="Model size: '7b', '13b', or '70b'")
    task:       str = Field("qa", description="Task type: 'qa', 'summarization', 'classification', 'chat'")


@router.post("/fine-tuning/generate-data", summary="Generate synthetic fine-tuning training data")
async def generate_training_data(request: GenerateDataRequest):
    """
    Generate a synthetic training dataset for fine-tuning using Claude.
    Returns examples in Alpaca format (instruction, input, output).
    """
    try:
        dataset = get_ft_builder().generate_from_description(
            domain=request.domain,
            instruction=request.instruction,
            n_examples=request.n_examples,
            dataset_name=request.dataset_name,
        )
        return {
            "dataset_name":      dataset.name,
            "examples_generated": len(dataset),
            "examples":          [ex.to_alpaca_format() for ex in dataset.examples],
            "jsonl_preview":     dataset.to_jsonl()[:500] + "...",
        }
    except Exception as exc:
        logger.exception("Training data generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fine-tuning/guide", summary="Fine-tuning vs RAG decision framework")
async def fine_tuning_guide():
    """Returns the decision framework for choosing between fine-tuning and RAG."""
    return {
        "framework": get_ft_guide().decision_framework(),
        "note": (
            "This course covers KAG and CAG as alternatives to RAG, plus fine-tuning "
            "concepts (LoRA/PEFT). The hands-on focus is on dataset preparation and "
            "knowing when each approach is appropriate."
        ),
    }


@router.post("/fine-tuning/lora-config", summary="Get recommended LoRA config")
async def lora_config(request: LoRAConfigRequest):
    """Returns a recommended LoRA configuration for a given model size and task."""
    valid_sizes = ["7b", "13b", "70b"]
    valid_tasks = ["qa", "summarization", "classification", "chat"]
    if request.model_size not in valid_sizes:
        raise HTTPException(status_code=400, detail=f"model_size must be one of {valid_sizes}")
    if request.task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"task must be one of {valid_tasks}")
    return get_ft_guide().lora_config_recommendation(request.model_size, request.task)
