"""
harness/ — Agent harness: run, observe, and score autonomous agents.

An agent harness is the infrastructure for testing agents against defined tasks.
It is NOT an evaluation module (that lives in app/eval/) — it is the scaffolding
that wraps agent execution, captures trajectories, and scores task completion.

Components:
  tasks.py     Task and TaskSuite definitions — what the agent should accomplish
  executor.py  Runs the agent, captures the full trajectory (every tool call)
  scorer.py    Scores task outcomes: did the agent complete the goal?
  runner.py    Batch runner — runs a TaskSuite and produces a HarnessReport

Flow:
  TaskSuite → HarnessRunner → AgentExecutor → TrajectoryCapturingAgent
                           → TaskScorer (outcome + tool_use + efficiency)
                           → HarnessReport (pass_rate, status, per-task breakdown)
"""

from app.harness.tasks import Task, TaskSuite, TaskResult, default_suite
from app.harness.executor import AgentExecutor
from app.harness.scorer import TaskScorer, TaskScore
from app.harness.runner import HarnessRunner, HarnessReport

__all__ = [
    "Task",
    "TaskSuite",
    "TaskResult",
    "default_suite",
    "AgentExecutor",
    "TaskScorer",
    "TaskScore",
    "HarnessRunner",
    "HarnessReport",
]
