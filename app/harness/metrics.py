"""
harness/metrics.py — Metric definitions and scoring rubrics for LLM evaluation.

Three core RAG metrics (from RAGAS and the broader research literature):

  Faithfulness   — Is every claim in the answer supported by the context?
                   (detects hallucination)

  Answer Relevance — Does the answer actually address the question?
                   (detects irrelevant or off-topic responses)

  Context Recall — Do the retrieved chunks contain the information needed
                   to answer the question?
                   (detects retrieval quality problems)

Additional agent metrics:
  Tool Precision — Did the agent call the right tools?
  Step Efficiency — Did the agent use the minimum steps needed?

STUDENT TODO:
  - Add a "Completeness" metric: does the answer cover all parts of the question?
  - Add a "Toxicity" metric: detect harmful outputs.
  - Calibrate your rubrics against human annotations for your specific domain.
"""

from dataclasses import dataclass, field
from typing import Literal

ScoreLevel = Literal["excellent", "good", "fair", "poor"]


@dataclass
class MetricScore:
    """The result of evaluating a single metric."""
    metric_name: str
    score: float           # 0.0 – 1.0
    level: ScoreLevel
    reasoning: str         # human-readable justification from the LLM judge
    suggestions: list[str] = field(default_factory=list)

    @classmethod
    def from_score(cls, metric_name: str, score: float, reasoning: str,
                   suggestions: list[str] | None = None) -> "MetricScore":
        if score >= 0.85:
            level = "excellent"
        elif score >= 0.65:
            level = "good"
        elif score >= 0.40:
            level = "fair"
        else:
            level = "poor"
        return cls(
            metric_name=metric_name,
            score=score,
            level=level,
            reasoning=reasoning,
            suggestions=suggestions or [],
        )


# ── RAG Metrics ───────────────────────────────────────────────────────────────
class RAGMetrics:
    """
    Collection of metric rubrics for RAG evaluation.

    These are the prompts used by the LLM-as-judge evaluator.
    Customise the rubrics for your domain.
    """

    FAITHFULNESS_RUBRIC = """
Evaluate the FAITHFULNESS of an AI answer against the provided context.

Faithfulness measures: Does every factual claim in the answer come from the context?
A faithful answer NEVER introduces facts not present in the context.

Score 1.0 (Excellent): Every claim is directly supported by the context. No hallucinations.
Score 0.7 (Good):      Most claims are supported; 1-2 minor additions that don't mislead.
Score 0.4 (Fair):      Some claims are unsupported; the answer mixes retrieved facts with invented ones.
Score 0.1 (Poor):      Most claims are not in the context. Significant hallucination.

Context:
{context}

Question: {question}
Answer: {answer}

Respond with JSON:
{{"score": <0.0-1.0>, "reasoning": "<one sentence>", "unsupported_claims": ["<claim1>", ...]}}
"""

    RELEVANCE_RUBRIC = """
Evaluate the ANSWER RELEVANCE of an AI answer to the question asked.

Relevance measures: Does the answer directly address the user's question?
An irrelevant answer might be factually correct but fail to answer what was asked.

Score 1.0 (Excellent): Directly and completely answers the question.
Score 0.7 (Good):      Addresses the question but with some tangential content.
Score 0.4 (Fair):      Partially answers but misses key aspects of the question.
Score 0.1 (Poor):      Fails to answer the question; off-topic or evasive.

Question: {question}
Answer: {answer}

Respond with JSON:
{{"score": <0.0-1.0>, "reasoning": "<one sentence>", "missing_aspects": ["<aspect1>", ...]}}
"""

    CONTEXT_RECALL_RUBRIC = """
Evaluate the CONTEXT RECALL for a RAG retrieval.

Context Recall measures: Do the retrieved chunks contain the information
needed to correctly answer the question?

Score 1.0 (Excellent): Context fully covers everything needed to answer.
Score 0.7 (Good):      Context covers most of the needed information.
Score 0.4 (Fair):      Context is partially relevant; key facts are missing.
Score 0.1 (Poor):      Context is largely irrelevant or missing critical information.

Question: {question}
Retrieved Context:
{context}

Expected/Ideal Answer (for reference): {ideal_answer}

Respond with JSON:
{{"score": <0.0-1.0>, "reasoning": "<one sentence>", "missing_information": ["<info1>", ...]}}
"""

    COMPLETENESS_RUBRIC = """
Evaluate the COMPLETENESS of an AI answer to the question.

Completeness measures: Does the answer cover all parts of a multi-part question?

Score 1.0: All aspects of the question are addressed.
Score 0.7: Most aspects covered; 1-2 minor omissions.
Score 0.4: About half the question is answered.
Score 0.1: Only a small portion of the question is addressed.

Question: {question}
Answer: {answer}

Respond with JSON:
{{"score": <0.0-1.0>, "reasoning": "<one sentence>", "omitted_aspects": ["<aspect1>", ...]}}
"""

    @classmethod
    def all_rubrics(cls) -> dict[str, str]:
        return {
            "faithfulness": cls.FAITHFULNESS_RUBRIC,
            "relevance": cls.RELEVANCE_RUBRIC,
            "context_recall": cls.CONTEXT_RECALL_RUBRIC,
            "completeness": cls.COMPLETENESS_RUBRIC,
        }
