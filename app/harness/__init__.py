"""
harness/ — Evaluation harness for LLM-powered systems.

"You can't improve what you don't measure."

This module provides structured evaluation tooling for RAG pipelines,
agents, and general LLM outputs — the foundation of responsible AI engineering.

Components:
  evaluator.py   — LLM-as-judge: score responses on faithfulness, relevance, etc.
  metrics.py     — Metric definitions and scoring rubrics
  tracer.py      — Request tracing and observability

Why harness engineering matters:
  Without evaluation, you are flying blind. Small prompt changes can
  silently degrade performance. The harness lets you:
    - Measure quality before and after changes
    - Catch regressions in CI
    - Build confidence that your system works

STUDENT TODO:
  - Write a test suite for your RAG pipeline using the evaluator.
  - Add harness checks to the GitHub Actions CI workflow.
  - Integrate with LangSmith or Langfuse for a production dashboard.
"""

from app.harness.evaluator import LLMEvaluator, EvaluationResult
from app.harness.metrics import RAGMetrics, MetricScore
from app.harness.tracer import RequestTracer, Span

__all__ = [
    "LLMEvaluator",
    "EvaluationResult",
    "RAGMetrics",
    "MetricScore",
    "RequestTracer",
    "Span",
]
