"""
api/harness.py — Evaluation harness API routes.

Endpoints:
  POST /harness/evaluate          — Evaluate a single (question, context, answer) triple
  POST /harness/evaluate/batch    — Evaluate a batch of test cases
  GET  /harness/tracer/stats      — Aggregated tracing statistics
  GET  /harness/tracer/recent     — Recent request traces
  POST /harness/prompts/register  — Register a prompt version
  GET  /harness/prompts/{name}    — Get the active version of a named prompt
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
        from app.harness.evaluator import LLMEvaluator
        _evaluator = LLMEvaluator(pass_threshold=0.70)
    return _evaluator


def get_tracer():
    global _tracer
    if _tracer is None:
        from app.harness.tracer import global_tracer
        _tracer = global_tracer
    return _tracer


def get_registry():
    global _registry
    if _registry is None:
        from app.harness.tracer import prompt_registry
        _registry = prompt_registry
    return _registry


# ── Evaluator endpoints ───────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    question:     str          = Field(..., description="The user's question")
    answer:       str          = Field(..., description="The system's generated answer")
    context:      str          = Field("",  description="Retrieved context used to generate the answer")
    ideal_answer: str          = Field("",  description="Optional reference answer for recall scoring")
    metrics:      list[str]    = Field(
        default=["faithfulness", "relevance"],
        description="Metrics to evaluate: faithfulness, relevance, context_recall, completeness"
    )


class TestCaseModel(BaseModel):
    question:     str
    context:      str
    answer:       str
    ideal_answer: str = ""


class BatchEvaluateRequest(BaseModel):
    test_cases: list[TestCaseModel] = Field(..., description="List of test cases to evaluate")
    metrics: list[str] = Field(default=["faithfulness", "relevance"])


@router.post("/evaluate", summary="Evaluate a single response")
async def evaluate(request: EvaluateRequest):
    """
    Run LLM-as-judge evaluation on a single (question, context, answer) triple.

    Returns per-metric scores (0.0-1.0), an overall score, pass/fail verdict,
    and detailed reasoning from the evaluator.

    Supported metrics:
      - faithfulness:     Are all claims in the answer supported by the context?
      - relevance:        Does the answer actually address the question?
      - context_recall:   Do the retrieved chunks contain the needed information?
      - completeness:     Does the answer cover all parts of the question?
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


@router.post("/evaluate/batch", summary="Evaluate a batch of test cases")
async def evaluate_batch(request: BatchEvaluateRequest):
    """
    Run evaluation across multiple test cases and return a regression report.
    Use this to check for regressions before deploying changes.
    """
    from app.harness.evaluator import TestCase
    test_cases = [
        TestCase(
            question=tc.question,
            context=tc.context,
            answer=tc.answer,
            ideal_answer=tc.ideal_answer,
        )
        for tc in request.test_cases
    ]
    try:
        evaluator = get_evaluator()
        results = evaluator.evaluate_batch(test_cases)
        report  = evaluator.regression_report(results)
        return report
    except Exception as exc:
        logger.exception("Batch evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tracer endpoints ──────────────────────────────────────────────────────────
@router.get("/tracer/stats", summary="Aggregated tracing statistics")
async def tracer_stats():
    """Returns aggregated latency, token, and cost statistics across all traced requests."""
    return get_tracer().get_stats()


@router.get("/tracer/recent", summary="Recent request traces")
async def tracer_recent(n: int = 10):
    """Returns the n most recent request traces."""
    traces = get_tracer().get_recent(n=n)
    return [t.to_dict() for t in traces]


# ── Prompt version registry ───────────────────────────────────────────────────
class RegisterPromptRequest(BaseModel):
    name:     str = Field(..., description="Name of the prompt template, e.g. 'rag_system'")
    template: str = Field(..., description="The full prompt template text")
    notes:    str = Field("",  description="Changelog note: what changed and why")


@router.post("/prompts/register", summary="Register a prompt version")
async def register_prompt(request: RegisterPromptRequest):
    """
    Register a new version of a named prompt template.
    Versions are numbered automatically (v1, v2, ...).
    Use this to track prompt changes and correlate them with eval score changes.
    """
    version_id = get_registry().register(
        name=request.name,
        template=request.template,
        notes=request.notes,
    )
    return {"version_id": version_id, "status": "registered"}


@router.get("/prompts/{name}", summary="Get active prompt version")
async def get_prompt(name: str):
    """Returns the latest (active) version of a named prompt."""
    active = get_registry().get_active(name)
    if not active:
        raise HTTPException(status_code=404, detail=f"No prompt named '{name}' found.")
    return active


@router.get("/prompts/{name}/history", summary="List all versions of a prompt")
async def prompt_history(name: str):
    """Returns all versions of a named prompt template."""
    versions = get_registry().list_versions(name)
    return {"name": name, "versions": versions}
