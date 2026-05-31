"""
harness/runner.py — Batch task runner and regression reporter.

The runner orchestrates a full TaskSuite run: execute every task, score it,
and produce a HarnessReport. Run the suite before and after a code change —
a drop in pass_rate or a status change from GREEN → YELLOW is a regression.

Status thresholds:
  GREEN  — 100% pass rate
  YELLOW — 70-99% pass rate
  RED    — below 70% pass rate

Usage:
    from app.harness.runner import HarnessRunner
    from app.harness.tasks import default_suite

    runner = HarnessRunner()
    report = runner.run(default_suite())

    print(report.status)          # "GREEN" | "YELLOW" | "RED"
    print(report.pass_rate)       # e.g. 0.8
    for r in report.results:
        if not r.success:
            print(f"FAIL: {r.task.description}")
            print(f"  Response:   {r.agent_response}")
            print(f"  Trajectory: {r.trajectory}")

STUDENT TODO:
  - Parallelise task execution: each task is independent and uses a separate agent.
    Use asyncio.gather() or concurrent.futures.ThreadPoolExecutor.
  - Persist reports to JSON so you can compare across deploys.
  - Integrate into CI: fail the pipeline if status == "RED".
  - Add per-tag breakdowns: which tags have the lowest pass rates?
"""

import logging
import time
from dataclasses import dataclass, field

from app.harness.tasks import Task, TaskSuite, TaskResult
from app.harness.executor import AgentExecutor
from app.harness.scorer import TaskScorer, TaskScore

logger = logging.getLogger(__name__)


@dataclass
class HarnessReport:
    """Aggregate report from a full TaskSuite run."""
    suite_name: str
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    avg_latency_ms: float
    avg_steps: float
    total_input_tokens: int
    total_output_tokens: int
    status: str                              # "GREEN" | "YELLOW" | "RED"
    results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "total_tasks": self.total_tasks,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 3),
            "avg_score": round(self.avg_score, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_steps": round(self.avg_steps, 1),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "status": self.status,
            "failed_tasks": [
                {
                    "task_id": r.task.id,
                    "description": r.task.description,
                    "score": round(r.score, 3),
                    "reasoning": r.score_reasoning,
                    "tools_expected": r.task.expected_tools,
                    "tools_called": r.tools_called,
                }
                for r in self.results if not r.success
            ],
            "results": [r.to_dict() for r in self.results],
        }


class HarnessRunner:
    """
    Runs a TaskSuite end-to-end: execute each task, score it, build a report.

    Each task gets a fresh agent instance (trajectory and memory fully isolated).
    Tasks run sequentially — see STUDENT TODO for parallelism.
    """

    def __init__(self, pass_threshold: float = 0.70):
        self._executor  = AgentExecutor()
        self._scorer    = TaskScorer(pass_threshold=pass_threshold)
        self._threshold = pass_threshold

    def run(self, suite: TaskSuite) -> HarnessReport:
        """
        Execute and score every task in the suite, return a HarnessReport.

        Tasks run in suite order. A fresh agent is created per task.
        """
        logger.info(
            "HarnessRunner: starting suite '%s' (%d tasks)", suite.name, len(suite)
        )
        t0 = time.perf_counter()

        results: list[TaskResult] = []
        scores:  list[TaskScore]  = []

        for i, task in enumerate(suite.tasks, 1):
            logger.info(
                "Task %d/%d: '%s' [%s]", i, len(suite), task.description, task.difficulty
            )
            result = self._executor.execute(task)
            score  = self._scorer.score(result)
            results.append(result)
            scores.append(score)

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        report   = self._build_report(suite.name, results, scores)
        logger.info(
            "Suite '%s' complete | %d/%d passed | status=%s | %.1fs",
            suite.name, report.passed, report.total_tasks, report.status, total_ms / 1000,
        )
        return report

    def run_task(self, task: Task) -> tuple[TaskResult, TaskScore]:
        """Execute and score a single task. Useful for interactive debugging."""
        result = self._executor.execute(task)
        score  = self._scorer.score(result)
        return result, score

    def _build_report(
        self,
        suite_name: str,
        results: list[TaskResult],
        scores: list[TaskScore],
    ) -> HarnessReport:
        if not results:
            return HarnessReport(
                suite_name=suite_name, total_tasks=0, passed=0, failed=0,
                pass_rate=0.0, avg_score=0.0, avg_latency_ms=0.0, avg_steps=0.0,
                total_input_tokens=0, total_output_tokens=0, status="RED",
            )

        passed    = sum(1 for r in results if r.success)
        failed    = len(results) - passed
        pass_rate = passed / len(results)

        avg_score   = sum(s.overall_score for s in scores) / len(scores)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_steps   = sum(r.steps_taken for r in results) / len(results)
        total_in    = sum(r.input_tokens for r in results)
        total_out   = sum(r.output_tokens for r in results)

        if pass_rate == 1.0:
            status = "GREEN"
        elif pass_rate >= 0.70:
            status = "YELLOW"
        else:
            status = "RED"

        return HarnessReport(
            suite_name=suite_name,
            total_tasks=len(results),
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            avg_score=avg_score,
            avg_latency_ms=avg_latency,
            avg_steps=avg_steps,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            status=status,
            results=results,
        )
