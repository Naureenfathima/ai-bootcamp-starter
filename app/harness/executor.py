"""
harness/executor.py — Runs an agent on a task and captures the full trajectory.

The executor is the bridge between a Task definition and a TaskResult.
It runs the agent, records every tool call with its inputs and outputs,
and packages everything as a TaskResult ready for scoring.

Why trajectory capture matters:
  "The answer was wrong" tells you nothing.
  "The agent called calculator('15 * 84') instead of calculator('0.15 * 840')"
  tells you exactly what to fix.

The TrajectoryCapturingAgent subclasses Agent without modifying it, intercepting
tool execution to record inputs and outputs at each step.

Systems lesson:
  Instrumentation without modifying production code is a standard observability
  pattern. The harness adds observation capability as a wrapper (decorator pattern)
  rather than coupling it to the core Agent class.

STUDENT TODO:
  - Run tasks in parallel using asyncio or ThreadPoolExecutor (each task is independent).
  - Add a timeout: tasks that run too long should fail with a clear error.
  - Stream trajectory steps in real time so you can watch the agent "think".
"""

import logging
import time

from app.harness.tasks import Task, TaskResult
from app.agent.agent import Agent, AgentResult
from app.agent.tools import DEFAULT_TOOLS
from app.config import settings

logger = logging.getLogger(__name__)


class TrajectoryCapturingAgent(Agent):
    """
    Agent subclass that records every tool call + result as a trajectory entry.

    Runs the full agentic loop exactly as Agent does, but intercepts each
    tool execution to append {step, tool_name, tool_input, tool_output} to
    self._trajectory.

    A fresh instance is created per task so trajectory state never leaks
    between tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trajectory: list[dict] = []

    def run_captured(self, user_message: str) -> tuple[AgentResult, list[dict]]:
        """
        Run the agent on a message and return (AgentResult, trajectory).

        Replaces the standard tool execution path with one that records
        inputs and outputs before returning the result to the agent loop.
        """
        import anthropic as _anthropic

        self._trajectory = []
        self._memory.add("user", user_message)

        steps_taken = 0
        tools_called: list[str] = []
        total_input = total_output = 0
        t0 = time.perf_counter()

        messages = self._memory.as_claude_messages()

        while steps_taken < settings.max_agent_steps:
            steps_taken += 1
            logger.debug("Harness step %d/%d", steps_taken, settings.max_agent_steps)

            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                system=self._system,
                tools=self._tool_schemas,
                messages=messages,
            )
            total_input  += response.usage.input_tokens
            total_output += response.usage.output_tokens

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")),
                    "Task completed — no text output.",
                )
                self._memory.add("assistant", final_text)
                total_ms = round((time.perf_counter() - t0) * 1000, 1)
                return AgentResult(
                    response=final_text,
                    steps_taken=steps_taken,
                    tools_called=tools_called,
                    total_latency_ms=total_ms,
                    input_tokens=total_input,
                    output_tokens=total_output,
                ), list(self._trajectory)

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name  = block.name
                    tool_input = block.input
                    tools_called.append(tool_name)

                    tool = self._tool_map.get(tool_name)
                    if tool is None:
                        output = f"Error: tool '{tool_name}' not found."
                        logger.error("Harness: unknown tool '%s'", tool_name)
                    else:
                        try:
                            output = tool.execute(**tool_input)
                        except Exception as exc:
                            output = f"Error in {tool_name}: {exc}"
                            logger.warning("Harness: tool '%s' raised: %s", tool_name, exc)

                    # Record this step to the trajectory
                    self._trajectory.append({
                        "step": steps_taken,
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_output": str(output)[:500],
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })

                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user",      "content": tool_results},
                ]
            else:
                logger.warning("Harness: unexpected stop_reason '%s'", response.stop_reason)
                break

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        return AgentResult(
            response="Max steps reached without completing the task.",
            steps_taken=steps_taken,
            tools_called=tools_called,
            total_latency_ms=total_ms,
            input_tokens=total_input,
            output_tokens=total_output,
        ), list(self._trajectory)


class AgentExecutor:
    """
    Runs a Task against a fresh agent instance and returns a TaskResult.

    A new agent is created per task so memory from previous tasks cannot
    influence the next one — each task is fully isolated.

    Usage:
        executor = AgentExecutor()
        result = executor.execute(task)

        print(result.trajectory)          # every tool call with inputs + outputs
        print(result.tool_use_correct())  # did it call the right tools?
        print(result.steps_taken)         # how many loop iterations?
    """

    def __init__(self, tools=None):
        self._tools = tools or DEFAULT_TOOLS

    def execute(self, task: Task) -> TaskResult:
        """Run a task and return a TaskResult (unscored — call scorer.score() next)."""
        logger.info(
            "Executor: task '%s' | '%s'", task.id, task.description[:60]
        )
        agent = TrajectoryCapturingAgent(tools=self._tools)

        try:
            agent_result, trajectory = agent.run_captured(task.input)
        except Exception as exc:
            logger.exception("Executor: task '%s' crashed", task.id)
            return TaskResult(
                task=task,
                agent_response=f"Execution crashed: {exc}",
                tools_called=[],
                steps_taken=0,
                trajectory=[],
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                success=False,
                score=0.0,
                score_reasoning=f"Agent raised an unhandled exception: {exc}",
            )

        logger.info(
            "Executor: task '%s' done | steps=%d tools=%s latency=%sms",
            task.id, agent_result.steps_taken,
            agent_result.tools_called, agent_result.total_latency_ms,
        )

        return TaskResult(
            task=task,
            agent_response=agent_result.response,
            tools_called=agent_result.tools_called,
            steps_taken=agent_result.steps_taken,
            trajectory=trajectory,
            latency_ms=agent_result.total_latency_ms,
            input_tokens=agent_result.input_tokens,
            output_tokens=agent_result.output_tokens,
        )
