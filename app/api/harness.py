"""
api/harness.py — Agent harness API routes.

Endpoints:
  GET  /harness/tasks               List all tasks in the default suite
  POST /harness/run/task            Run and score a single task by ID
  POST /harness/run/suite           Run the full default suite
  POST /harness/run/custom          Run and score a custom task (define inline)
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

_runner = None


def get_runner():
    global _runner
    if _runner is None:
        from app.harness.runner import HarnessRunner
        _runner = HarnessRunner(pass_threshold=0.70)
    return _runner


# ── Schemas ───────────────────────────────────────────────────────────────────
class RunTaskRequest(BaseModel):
    task_id: str = Field(..., description="Task ID from GET /harness/tasks")


class CustomTaskRequest(BaseModel):
    description:      str       = Field(..., description="Human-readable goal label")
    input:            str       = Field(..., description="The message to send to the agent")
    expected_tools:   list[str] = Field(default=[], description="Tool names that should be called")
    success_criteria: str       = Field(..., description="What does a passing response look like?")
    difficulty:       str       = Field(default="medium", description="easy | medium | hard")
    tags:             list[str] = Field(default=[])


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/tasks", summary="List tasks in the default suite")
async def list_tasks():
    """
    Returns all tasks in the default suite with their IDs, descriptions,
    expected tools, and difficulty levels.
    """
    from app.harness.tasks import default_suite
    suite = default_suite()
    return {
        "suite": suite.name,
        "total": len(suite),
        "tasks": [
            {
                "id":               t.id,
                "description":      t.description,
                "input":            t.input,
                "expected_tools":   t.expected_tools,
                "difficulty":       t.difficulty,
                "tags":             t.tags,
                "success_criteria": t.success_criteria,
            }
            for t in suite.tasks
        ],
    }


@router.post("/run/task", summary="Run and score a single task")
async def run_task(request: RunTaskRequest):
    """
    Run the agent on one task from the default suite and return the full result:
    trajectory (every tool call with inputs + outputs), score breakdown, pass/fail.
    """
    from app.harness.tasks import default_suite
    suite = default_suite()
    task  = next((t for t in suite.tasks if t.id == request.task_id), None)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{request.task_id}' not found. Call GET /harness/tasks to list available IDs."
        )
    try:
        result, score = get_runner().run_task(task)
        return {
            **result.to_dict(),
            "score_breakdown": {
                "outcome":    round(score.outcome_score, 3),
                "tool_use":   round(score.tool_use_score, 3),
                "efficiency": round(score.efficiency_score, 3),
                "overall":    round(score.overall_score, 3),
            },
        }
    except Exception as exc:
        logger.exception("run_task failed for '%s'", request.task_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/run/suite", summary="Run the full default task suite")
async def run_suite():
    """
    Run every task in the default suite sequentially.
    Returns a HarnessReport with pass_rate, avg_score, per-task results,
    and a GREEN / YELLOW / RED status signal.

    GREEN  = 100% pass rate
    YELLOW = 70-99% pass rate
    RED    = below 70%

    Use this as a regression check before deploying changes.
    """
    try:
        from app.harness.tasks import default_suite
        report = get_runner().run(default_suite())
        return report.to_dict()
    except Exception as exc:
        logger.exception("run_suite failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/run/custom", summary="Run and score a custom task defined inline")
async def run_custom_task(request: CustomTaskRequest):
    """
    Define and run a one-off task without adding it to the default suite.
    Useful for testing specific agent behaviours during development.
    """
    from app.harness.tasks import Task
    task = Task.create(
        description=request.description,
        input=request.input,
        expected_tools=request.expected_tools,
        success_criteria=request.success_criteria,
        tags=request.tags,
        difficulty=request.difficulty,
    )
    try:
        result, score = get_runner().run_task(task)
        return {
            **result.to_dict(),
            "score_breakdown": {
                "outcome":    round(score.outcome_score, 3),
                "tool_use":   round(score.tool_use_score, 3),
                "efficiency": round(score.efficiency_score, 3),
                "overall":    round(score.overall_score, 3),
            },
        }
    except Exception as exc:
        logger.exception("run_custom_task failed")
        raise HTTPException(status_code=500, detail=str(exc))
