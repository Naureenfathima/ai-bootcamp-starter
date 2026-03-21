"""
agent.py — Autonomous agent with tool use and memory.

The agent follows an agentic loop:
  1. Receive user message
  2. Call Claude with tools
  3. If Claude calls a tool → execute it → feed result back
  4. Repeat until Claude gives a final text response

STUDENT TODO:
  - Add multi-agent support: spawn sub-agents for parallel sub-tasks.
  - Add chain-of-thought prompting: ask the agent to plan before acting.
  - Add structured output: return results as JSON for downstream processing.
  - Implement a step budget: warn the user if the agent is taking many steps.
"""

import logging
import time
from dataclasses import dataclass, field

import anthropic

from app.config import settings
from app.agent.tools import Tool, DEFAULT_TOOLS
from app.agent.memory import EpisodicMemory, WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    response: str
    steps_taken: int
    tools_called: list[str]
    total_latency_ms: float
    input_tokens: int
    output_tokens: int


class Agent:
    """
    An LLM-powered agent that can use tools to complete multi-step tasks.

    Usage:
        agent = Agent()
        result = agent.run("What is 25% of 840, and what is today's date?")
        print(result.response)

    The agent will automatically call the 'calculator' and 'get_current_time'
    tools as needed, then combine the results into a final answer.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ):
        self._client  = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._tools   = tools or DEFAULT_TOOLS
        self._memory  = EpisodicMemory()
        self._working = WorkingMemory()

        self._system = system_prompt or (
            "You are a helpful AI assistant with access to tools. "
            "When you need to perform a calculation, search for information, or check the time, "
            "use the appropriate tool. Think step by step before acting. "
            "Be concise and direct in your final responses."
        )

        # Build tool schemas for Claude
        self._tool_schemas = [t.as_claude_tool() for t in self._tools]
        self._tool_map     = {t.name: t for t in self._tools}

        logger.info("Agent initialised with %d tools: %s",
                    len(self._tools), [t.name for t in self._tools])

    def run(self, user_message: str) -> AgentResult:
        """
        Run the agent on a user message.

        The agent will loop until Claude produces a final text response
        (i.e. stops calling tools) or until max_agent_steps is reached.

        Args:
            user_message: The user's request in plain text.

        Returns:
            AgentResult with the final response and execution metadata.
        """
        t_total = time.perf_counter()

        self._memory.add("user", user_message)

        steps_taken  = 0
        tools_called = []
        total_input_tokens  = 0
        total_output_tokens = 0

        messages = self._memory.as_claude_messages()

        while steps_taken < settings.max_agent_steps:
            steps_taken += 1
            logger.info("Agent step %d/%d", steps_taken, settings.max_agent_steps)

            # ── Call Claude ────────────────────────────────────────────────────
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                system=self._system,
                tools=self._tool_schemas,
                messages=messages,
            )

            total_input_tokens  += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # ── Check stop reason ──────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                # Claude is done — extract the final text response
                final_text = next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "I completed the task but could not generate a text response."
                )
                self._memory.add("assistant", final_text)

                total_ms = round((time.perf_counter() - t_total) * 1000, 1)
                logger.info(
                    "Agent finished | steps=%d | tools=%s | total=%sms | tokens=%d",
                    steps_taken, tools_called, total_ms,
                    total_input_tokens + total_output_tokens,
                )

                return AgentResult(
                    response=final_text,
                    steps_taken=steps_taken,
                    tools_called=tools_called,
                    total_latency_ms=total_ms,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )

            elif response.stop_reason == "tool_use":
                # Claude wants to call one or more tools
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input
                    tools_called.append(tool_name)

                    logger.info("Tool call: %s(%s)", tool_name, tool_input)

                    # Find and execute the tool
                    tool = self._tool_map.get(tool_name)
                    if tool is None:
                        tool_output = f"Error: tool '{tool_name}' not found."
                        logger.error("Unknown tool: %s", tool_name)
                    else:
                        try:
                            tool_output = tool.execute(**tool_input)
                        except Exception as exc:
                            tool_output = f"Error executing {tool_name}: {exc}"
                            logger.exception("Tool execution failed: %s", tool_name)

                    logger.info("Tool result: %s → %r", tool_name, tool_output[:200])

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(tool_output),
                    })

                # Feed tool results back to Claude
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user",      "content": tool_results},
                ]

            else:
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

        # Safety: max steps reached
        logger.warning("Agent hit max_agent_steps=%d without finishing", settings.max_agent_steps)
        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        return AgentResult(
            response="I ran into a limit while completing this task. Please try a simpler request.",
            steps_taken=steps_taken,
            tools_called=tools_called,
            total_latency_ms=total_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def reset(self) -> None:
        """Clear all memory and start a fresh session."""
        self._memory.clear()
        self._working.clear()
        logger.info("Agent memory cleared")
