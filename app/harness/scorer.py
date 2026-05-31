"""
harness/scorer.py — Task outcome scoring.

The scorer answers: did the agent complete the task?

This is fundamentally different from RAG evaluation (app/eval/evaluator.py),
which asks "is the answer faithful to the retrieved context?". The task scorer
asks three questions:

  1. Outcome   — Did the final response satisfy the success criteria?   (LLM judge, 0-1)
  2. Tool use  — Did the agent call every expected tool?                (deterministic, 0-1)
  3. Efficiency — Was the step count reasonable, or did it loop?       (deterministic, 0-1)

Overall = weighted average (outcome 60%, tool use 25%, efficiency 15%).

Systems lesson:
  Scoring separates into two types:
  - Deterministic scoring (tool use, efficiency): fast, free, always consistent.
  - LLM-as-judge (outcome): slower, costs tokens, but can evaluate open-ended goals.
  Maximise deterministic scoring first — use LLM judge only where it's necessary.

STUDENT TODO:
  - Add a "safety" dimension: flag responses that include harmful content.
  - Cache outcome scores: identical (task_id, response) pairs should not re-call the LLM.
  - Tune pass_threshold per task difficulty: easy tasks should pass at 0.85+.
"""

import json
import logging
from dataclasses import dataclass

import anthropic
from app.config import settings
from app.harness.tasks import Task, TaskResult

logger = logging.getLogger(__name__)

_OUTCOME_RUBRIC = """
You are evaluating whether an AI agent successfully completed a task.

Task: {description}
Success criteria: {success_criteria}

Agent's final response:
{agent_response}

Tools called (in order): {tools_called}

Trajectory (each tool call with inputs and outputs):
{trajectory}

Score the agent's outcome 0.0 – 1.0:
  1.0  Task fully completed. All success criteria met.
  0.7  Task mostly completed. Minor gaps or imprecision.
  0.4  Task partially completed. Key parts missing or incorrect.
  0.1  Task attempted but failed. Response is wrong or off-target.
  0.0  Complete failure or crash. Agent did not attempt the task.

Respond with JSON only — no explanation outside the JSON:
{{"score": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""


@dataclass
class TaskScore:
    """Per-dimension score breakdown for one TaskResult."""
    outcome_score: float       # LLM judge: did it achieve the goal?
    tool_use_score: float      # deterministic: did it call the right tools?
    efficiency_score: float    # deterministic: was the step count reasonable?
    overall_score: float       # weighted average
    passed: bool
    outcome_reasoning: str


class TaskScorer:
    """
    Scores a TaskResult across three dimensions and writes scores back to the result.

    Usage:
        scorer = TaskScorer(pass_threshold=0.70)
        task_score = scorer.score(task_result)

        print(task_score.overall_score)
        print(task_score.passed)
        print(task_score.outcome_reasoning)

        # Scores are also written back to task_result:
        print(task_result.success)
        print(task_result.score)
    """

    def __init__(
        self,
        pass_threshold: float = 0.70,
        outcome_weight: float = 0.60,
        tool_use_weight: float = 0.25,
        efficiency_weight: float = 0.15,
    ):
        self._client       = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._threshold    = pass_threshold
        self._w_outcome    = outcome_weight
        self._w_tool       = tool_use_weight
        self._w_efficiency = efficiency_weight

    def score(self, result: TaskResult) -> TaskScore:
        """Score a TaskResult and write scores back into the result object."""
        outcome_score, reasoning = self._score_outcome(result)
        tool_use_score           = self._score_tool_use(result)
        efficiency_score         = self._score_efficiency(result)

        overall = round(
            self._w_outcome    * outcome_score
            + self._w_tool     * tool_use_score
            + self._w_efficiency * efficiency_score,
            3,
        )
        passed = overall >= self._threshold

        # Write back so TaskResult is self-contained
        result.success         = passed
        result.score           = overall
        result.score_reasoning = reasoning

        logger.info(
            "Scored '%s': outcome=%.2f tool=%.2f eff=%.2f → overall=%.2f [%s]",
            result.task.id, outcome_score, tool_use_score, efficiency_score,
            overall, "PASS" if passed else "FAIL",
        )

        return TaskScore(
            outcome_score=outcome_score,
            tool_use_score=tool_use_score,
            efficiency_score=efficiency_score,
            overall_score=overall,
            passed=passed,
            outcome_reasoning=reasoning,
        )

    # ── Dimension scorers ──────────────────────────────────────────────────────

    def _score_outcome(self, result: TaskResult) -> tuple[float, str]:
        """LLM-as-judge: did the agent achieve the goal?"""
        trajectory_str = "\n".join(
            f"  Step {s['step']}: {s['tool_name']}({s['tool_input']}) → {str(s['tool_output'])[:200]}"
            for s in result.trajectory
        ) or "  (no tool calls made)"

        prompt = _OUTCOME_RUBRIC.format(
            description=result.task.description,
            success_criteria=result.task.success_criteria,
            agent_response=result.agent_response[:1000],
            tools_called=result.tools_called or ["(none)"],
            trajectory=trajectory_str,
        )

        try:
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            data    = json.loads(raw)
            score   = max(0.0, min(1.0, float(data.get("score", 0.5))))
            reason  = data.get("reasoning", "")
            return score, reason
        except Exception as exc:
            logger.warning("Outcome scoring error for '%s': %s", result.task.id, exc)
            return 0.5, f"Scoring error: {exc}"

    def _score_tool_use(self, result: TaskResult) -> float:
        """Partial credit: fraction of expected tools that were called."""
        expected = set(result.task.expected_tools)
        if not expected:
            return 1.0
        called = set(result.tools_called)
        return len(called & expected) / len(expected)

    def _score_efficiency(self, result: TaskResult) -> float:
        """
        Penalise excessive looping.

        Expected steps ≈ number of expected tools.
        ≤ 1.5× expected: full score.
        > 3× expected:   low score.
        """
        expected = max(len(result.task.expected_tools), 1)
        actual   = max(result.steps_taken, 1)
        ratio    = actual / expected
        if ratio <= 1.5:
            return 1.0
        elif ratio <= 2.0:
            return 0.8
        elif ratio <= 3.0:
            return 0.5
        else:
            return 0.2
