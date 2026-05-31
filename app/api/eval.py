"""
api/eval.py — RAG evaluation and observability API routes.

Endpoints:
  POST /eval/evaluate          Evaluate a single (question, context, answer) triple
  POST /eval/evaluate/batch    Evaluate a batch of test cases
  GET  /eval/tracer/stats      Aggregated tracing statistics
  GET  /eval/tracer/recent     Recent request traces
  POST /eval/prompts/register  Register a prompt version
  GET  /eval/prompts/{name}    Get the active version of a named prompt
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

_evaluator = None
_tracer    = None
_registry  = None


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        from app.eval.evaluator import LLMEvaluator
        _evaluator = LLMEvaluator(pass_threshold=0.70)
    return _evaluator


def get_tracer():
    global _tracer
    if _tracer is None:
        from app.eval.tracer import global_tracer
        _tracer = global_tracer
    return _tracer


def get_registry():
    global _registry
    if _registry is None:
        from app.eval.tracer import prompt_registry
        _registry = prompt_registry
    return _registry


# ── Evaluator endpoints ───────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    question:     str       = Field(..., description="The user's question")
    answer:       str       = Field(..., description="The system's generated answer")
    context:      str       = Field("",  description="Retrieved context used to generate the answer")
    ideal_answer: str       = Field("",  description="Optional reference answer for recall scoring")
    metrics:      list[str] = Field(
        default=["faithfulness", "relevance"],
        description="Metrics: faithfulness | relevance | context_recall | completeness"
    )


class TestCaseModel(BaseModel):
    question:     str
    context:      str
    answer:       str
    ideal_answer: str = ""


class BatchEvaluateRequest(BaseModel):
    test_cases: list[TestCaseModel] = Field(..., description="Test cases to evaluate")
    metrics: list[str] = Field(default=["faithfulness", "relevance"])


@router.post("/evaluate", summary="Evaluate a single RAG response")
async def evaluate(request: EvaluateRequest):
    """
    LLM-as-judge evaluation on a (question, context, answer) triple.
    Returns per-metric scores (0-1), overall score, and pass/fail verdict.
    """
    try:
        result = get_evaluator().evaluate(
            question=request.question,
            answer=request.answer,
            context=request.context,
            ideal_answer=request.ideal_answer,
            metrics=request.metrics,
        )
        return {
            "overall_score": result.overall_score,
            "passed": result.passed,
            "latency_ms": result.latency_ms,
            "scores": {
                name: {
                    "score":       score.score,
                    "level":       score.level,
                    "reasoning":   score.reasoning,
                    "suggestions": score.suggestions,
                }
                for name, score in result.scores.items()
            },
        }
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/evaluate/batch", summary="Evaluate a batch and return regression report")
async def evaluate_batch(request: BatchEvaluateRequest):
    """
    Evaluate multiple (question, context, answer) triples and return
    a regression report with pass_rate, per-metric averages, and GREEN/YELLOW/RED status.
    """
    from app.eval.evaluator import TestCase
    test_cases = [
        TestCase(question=tc.question, context=tc.context,
                 answer=tc.answer, ideal_answer=tc.ideal_answer)
        for tc in request.test_cases
    ]
    try:
        evaluator = get_evaluator()
        results   = evaluator.evaluate_batch(test_cases)
        return evaluator.regression_report(results)
    except Exception as exc:
        logger.exception("Batch evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tracer endpoints ──────────────────────────────────────────────────────────
@router.get("/tracer/stats", summary="Aggregated tracing statistics")
async def tracer_stats():
    """Aggregated latency, token, and cost statistics across all traced requests."""
    return get_tracer().get_stats()


@router.get("/tracer/recent", summary="Recent request traces")
async def tracer_recent(n: int = 10):
    """Returns the n most recent request traces."""
    return [t.to_dict() for t in get_tracer().get_recent(n=n)]


# ── Prompt version registry ───────────────────────────────────────────────────
class RegisterPromptRequest(BaseModel):
    name:     str = Field(..., description="Prompt name, e.g. 'rag_system'")
    template: str = Field(..., description="Full prompt template text")
    notes:    str = Field("",  description="What changed and why")


@router.post("/prompts/register", summary="Register a prompt version")
async def register_prompt(request: RegisterPromptRequest):
    """Register a new version of a named prompt template (auto-versioned v1, v2, ...)."""
    version_id = get_registry().register(
        name=request.name, template=request.template, notes=request.notes,
    )
    return {"version_id": version_id, "status": "registered"}


@router.get("/prompts/{name}", summary="Get active prompt version")
async def get_prompt(name: str):
    """Returns the latest (active) version of a named prompt."""
    active = get_registry().get_active(name)
    if not active:
        raise HTTPException(status_code=404, detail=f"No prompt named '{name}' found.")
    return active


@router.get("/prompts/{name}/history", summary="All versions of a prompt")
async def prompt_history(name: str):
    """Returns all registered versions of a named prompt."""
    return {"name": name, "versions": get_registry().list_versions(name)}
