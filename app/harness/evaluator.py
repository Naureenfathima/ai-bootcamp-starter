"""
harness/evaluator.py — LLM-as-judge evaluation engine.

The idea: use a powerful LLM (Claude) to grade the output of another LLM call.
This is the standard approach for automated RAG and agent evaluation.

Pipeline:
  1. Run your system (RAG, agent, etc.) to get a response
  2. Pass the (question, context, answer) triple to the evaluator
  3. Evaluator prompts Claude with a scoring rubric
  4. Claude returns a structured score + reasoning
  5. Log the result, alert on regressions

STUDENT TODO:
  - Build an evaluation dataset from your domain (10-50 gold-standard Q&A pairs).
  - Run the evaluator in CI to catch regressions before deploying.
  - Integrate with LangSmith (https://smith.langchain.com/) or
    Langfuse (https://langfuse.com/) for a full observability dashboard.
  - Add a batch runner that evaluates your entire test set and reports averages.
"""

import logging
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from app.config import settings
from app.harness.metrics import RAGMetrics, MetricScore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation of a single response."""
    question: str
    answer: str
    context: str
    scores: dict[str, MetricScore] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = False
    latency_ms: float = 0.0
    evaluator_tokens: int = 0

    def summary(self) -> str:
        lines = [
            f"Evaluation Summary",
            f"  Question:      {self.question[:80]}...",
            f"  Overall Score: {self.overall_score:.2f} ({'PASS' if self.passed else 'FAIL'})",
        ]
        for name, score in self.scores.items():
            lines.append(f"  {name:<20} {score.score:.2f} [{score.level}]  — {score.reasoning}")
        return "\n".join(lines)


@dataclass
class TestCase:
    """A single test case for the evaluation harness."""
    question: str
    context: str
    answer: str
    ideal_answer: str = ""   # optional reference answer
    tags: list[str] = field(default_factory=list)


class LLMEvaluator:
    """
    Evaluates LLM outputs using Claude as an impartial judge.

    Usage:
        evaluator = LLMEvaluator(pass_threshold=0.70)

        result = evaluator.evaluate(
            question="What is the return policy?",
            context="Our return policy allows returns within 30 days...",
            answer="You can return items within 30 days of purchase.",
        )

        print(result.summary())
        if not result.passed:
            print("REGRESSION — score below threshold!")
    """

    def __init__(
        self,
        pass_threshold: float = 0.70,
        metrics: list[str] | None = None,
    ):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._threshold = pass_threshold
        # Default to faithfulness + relevance; completeness optional
        self._metrics = metrics or ["faithfulness", "relevance"]
        logger.info("LLMEvaluator: metrics=%s threshold=%.2f", self._metrics, pass_threshold)

    def _score_metric(
        self,
        metric_name: str,
        question: str,
        answer: str,
        context: str,
        ideal_answer: str = "",
    ) -> MetricScore:
        """Run a single metric evaluation using the LLM-as-judge pattern."""
        rubrics = RAGMetrics.all_rubrics()
        rubric_template = rubrics.get(metric_name)
        if not rubric_template:
            raise ValueError(f"Unknown metric: {metric_name!r}")

        prompt = rubric_template.format(
            question=question,
            answer=answer,
            context=context,
            ideal_answer=ideal_answer or "(not provided)",
        )

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
            score_val = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "")
            suggestions = (
                data.get("unsupported_claims", [])
                or data.get("missing_aspects", [])
                or data.get("missing_information", [])
                or data.get("omitted_aspects", [])
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse evaluator response for %s: %s", metric_name, e)
            score_val = 0.5
            reasoning = "Parse error — check evaluator output"
            suggestions = []

        return MetricScore.from_score(
            metric_name=metric_name,
            score=score_val,
            reasoning=reasoning,
            suggestions=suggestions,
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        ideal_answer: str = "",
        metrics: list[str] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a single (question, context, answer) triple.

        Args:
            question:     The user's question.
            answer:       The system's generated answer.
            context:      The retrieved context used to generate the answer.
            ideal_answer: Optional reference answer for context_recall scoring.
            metrics:      Override the default metrics list for this evaluation.

        Returns:
            EvaluationResult with per-metric scores and overall pass/fail.
        """
        t0 = time.perf_counter()
        metrics_to_run = metrics or self._metrics
        scores: dict[str, MetricScore] = {}
        total_tokens = 0

        for metric_name in metrics_to_run:
            try:
                score = self._score_metric(
                    metric_name=metric_name,
                    question=question,
                    answer=answer,
                    context=context,
                    ideal_answer=ideal_answer,
                )
                scores[metric_name] = score
                logger.info("Metric %s: %.2f [%s]", metric_name, score.score, score.level)
            except Exception as e:
                logger.error("Failed to evaluate metric %s: %s", metric_name, e)

        overall = sum(s.score for s in scores.values()) / max(len(scores), 1)
        passed   = overall >= self._threshold
        latency  = round((time.perf_counter() - t0) * 1000, 1)

        result = EvaluationResult(
            question=question,
            answer=answer,
            context=context,
            scores=scores,
            overall_score=round(overall, 3),
            passed=passed,
            latency_ms=latency,
        )

        logger.info("Evaluation: overall=%.2f (%s) in %sms",
                    overall, "PASS" if passed else "FAIL", latency)
        return result

    def evaluate_batch(self, test_cases: list[TestCase]) -> list[EvaluationResult]:
        """
        Evaluate a batch of test cases and return all results.

        Use this to run a full regression suite before deploying changes.
        """
        results = []
        for i, tc in enumerate(test_cases, 1):
            logger.info("Evaluating test case %d/%d", i, len(test_cases))
            result = self.evaluate(
                question=tc.question,
                answer=tc.answer,
                context=tc.context,
                ideal_answer=tc.ideal_answer,
            )
            results.append(result)

        passed    = sum(1 for r in results if r.passed)
        avg_score = sum(r.overall_score for r in results) / max(len(results), 1)
        logger.info(
            "Batch evaluation: %d/%d passed | avg_score=%.2f",
            passed, len(results), avg_score
        )
        return results

    def regression_report(self, results: list[EvaluationResult]) -> dict:
        """
        Summarise a batch evaluation into a regression report.
        Suitable for CI output or a Slack notification.
        """
        if not results:
            return {"error": "No results to report"}

        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        avg_score = sum(r.overall_score for r in results) / len(results)

        metric_averages = {}
        all_metrics = set(m for r in results for m in r.scores)
        for m in all_metrics:
            vals = [r.scores[m].score for r in results if m in r.scores]
            metric_averages[m] = round(sum(vals) / len(vals), 3) if vals else 0.0

        return {
            "total_cases": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(len(passed) / len(results), 3),
            "average_score": round(avg_score, 3),
            "metric_averages": metric_averages,
            "threshold": self._threshold,
            "status": "GREEN" if len(failed) == 0 else ("YELLOW" if avg_score >= 0.6 else "RED"),
            "failed_cases": [
                {"question": r.question[:60], "score": r.overall_score}
                for r in failed
            ],
        }
