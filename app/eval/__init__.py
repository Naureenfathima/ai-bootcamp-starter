"""
eval/ — RAG evaluation and observability.

Components:
  evaluator.py   LLM-as-judge: score RAG responses on faithfulness, relevance, etc.
  metrics.py     Metric definitions and scoring rubrics
  tracer.py      Request tracing, cost tracking, prompt version registry
"""

from app.eval.evaluator import LLMEvaluator, EvaluationResult
from app.eval.metrics import RAGMetrics, MetricScore
from app.eval.tracer import RequestTracer, Span

__all__ = [
    "LLMEvaluator",
    "EvaluationResult",
    "RAGMetrics",
    "MetricScore",
    "RequestTracer",
    "Span",
]
