from __future__ import annotations
from typing import Optional
"""
harness/tasks.py — Task definitions for the agent harness.

A Task is the atomic unit of the harness: a structured goal with a defined
input, expected tools, and success criteria. The harness runs tasks, captures
what the agent actually did (trajectory), and scores the outcome.

A TaskSuite groups related tasks into a named benchmark. Run a suite before
and after a code change — a drop in pass_rate is a regression signal.
"""

import uuid
from dataclasses import dataclass, field


@dataclass
class Task:
    """One agent task: a goal with defined inputs and success criteria."""
    id: str
    description: str           # human-readable label for reports
    input: str                 # the message sent to agent.run()
    expected_tools: list[str]  # tools that must be called for a correct solution
    success_criteria: str      # plain English — what does passing look like?
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium" # "easy" | "medium" | "hard"

    @classmethod
    def create(
        cls,
        description: str,
        input: str,
        expected_tools: list[str],
        success_criteria: str,
        tags:Optional[ list[str]] = None,
        difficulty: str = "medium",
    ) -> "Task":
        return cls(
            id=str(uuid.uuid4())[:8],
            description=description,
            input=input,
            expected_tools=expected_tools,
            success_criteria=success_criteria,
            tags=tags or [],
            difficulty=difficulty,
        )


@dataclass
class TaskResult:
    """The full outcome of running one task against an agent."""
    task: Task
    agent_response: str
    tools_called: list[str]
    steps_taken: int
    trajectory: list[dict]     # [{step, tool_name, tool_input, tool_output}]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool = False
    score: float = 0.0
    score_reasoning: str = ""

    def tool_use_correct(self) -> bool:
        """Did the agent call every expected tool?"""
        if not self.task.expected_tools:
            return True
        return all(t in self.tools_called for t in self.task.expected_tools)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.id,
            "task_description": self.task.description,
            "difficulty": self.task.difficulty,
            "tags": self.task.tags,
            "success": self.success,
            "score": round(self.score, 3),
            "score_reasoning": self.score_reasoning,
            "tool_use_correct": self.tool_use_correct(),
            "tools_expected": self.task.expected_tools,
            "tools_called": self.tools_called,
            "steps_taken": self.steps_taken,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "trajectory": self.trajectory,
            "agent_response": self.agent_response,
        }


class TaskSuite:
    """A named collection of tasks run as a single benchmark."""

    def __init__(self, name: str):
        self.name = name
        self._tasks: list[Task] = []

    def add(self, task: Task) -> "TaskSuite":
        self._tasks.append(task)
        return self

    @property
    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def filter_by_tag(self, tag: str) -> list[Task]:
        return [t for t in self._tasks if tag in t.tags]

    def filter_by_difficulty(self, difficulty: str) -> list[Task]:
        return [t for t in self._tasks if t.difficulty == difficulty]

    def __len__(self) -> int:
        return len(self._tasks)


# ── Built-in suite ─────────────────────────────────────────────────────────────
def default_suite() -> TaskSuite:
    """
    Starter suite covering the three built-in tools.
    Use this as a baseline regression suite during development.
    Extend it with domain-specific tasks for your capstone project.
    """
    suite = TaskSuite(name="default")

    suite.add(Task.create(
        description="Single arithmetic calculation",
        input="What is 15% of 840?",
        expected_tools=["calculator"],
        success_criteria="The answer is 126. The agent must use the calculator tool and state a number close to 126.",
        tags=["math", "single-tool"],
        difficulty="easy",
    ))

    suite.add(Task.create(
        description="Current time retrieval",
        input="What is the current UTC time?",
        expected_tools=["get_current_time"],
        success_criteria="The agent returns the current time in a readable format. It must use the get_current_time tool.",
        tags=["time", "single-tool"],
        difficulty="easy",
    ))

    suite.add(Task.create(
        description="Multi-tool: math + time",
        input="What is sqrt(144) added to the current UTC hour (as a number)?",
        expected_tools=["calculator", "get_current_time"],
        success_criteria=(
            "The agent uses both the calculator and get_current_time tools. "
            "sqrt(144)=12. The final answer should be 12 + current_hour."
        ),
        tags=["math", "time", "multi-tool"],
        difficulty="medium",
    ))

    suite.add(Task.create(
        description="Multi-step compound interest calculation",
        input="If I invest $5,000 at 7% annual interest compounded yearly, how much will I have after 3 years?",
        expected_tools=["calculator"],
        success_criteria=(
            "The agent computes 5000 * (1.07 ** 3) ≈ 6125.22 using the calculator. "
            "The final answer should be close to $6,125."
        ),
        tags=["math", "multi-step", "finance"],
        difficulty="medium",
    ))

    suite.add(Task.create(
        description="Error recovery — imaginary number",
        input="What is the square root of -1?",
        expected_tools=["calculator"],
        success_criteria=(
            "The agent attempts the calculation, handles the error gracefully, "
            "and explains that sqrt(-1) is imaginary (not a real number)."
        ),
        tags=["math", "error-handling"],
        difficulty="hard",
    ))

    return suite
