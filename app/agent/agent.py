from __future__ import annotations
from typing import Optional
"""
agent.py — Autonomous agent with tool use and memory.

The agent follows an agentic loop:
  1. Receive user message
  2. Call the LLM with tools (provider set by LLM_PROVIDER in .env)
  3. If the LLM calls a tool → execute it → feed result back
  4. Repeat until the LLM gives a final text response

STUDENT TODO:
  - Add multi-agent support: spawn sub-agents for parallel sub-tasks.
  - Add chain-of-thought prompting: ask the agent to plan before acting.
  - Add structured output: return results as JSON for downstream processing.
  - Implement a step budget: warn the user if the agent is taking many steps.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from app.config import settings
from app.llm import get_llm_client
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
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
    ):
        self._llm     = get_llm_client()
        self._tools   = tools or DEFAULT_TOOLS
        self._memory  = EpisodicMemory()
        self._working = WorkingMemory()

        self._system = system_prompt or (
            "You are a helpful AI assistant with access to tools. "
            "When you need to perform a calculation, search for information, or check the time, "
            "use the appropriate tool. Think step by step before acting. "
            "Be concise and direct in your final responses."
        )

        # Tool schemas in OpenAI function-calling format (universal)
        self._tool_schemas = [t.as_openai_tool() for t in self._tools]
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

        # Messages in OpenAI format — provider-agnostic throughout the loop
        messages = self._memory.as_messages()

        while steps_taken < settings.max_agent_steps:
            steps_taken += 1
            logger.info("Agent step %d/%d", steps_taken, settings.max_agent_steps)

            turn = self._llm.chat_with_tools(
                messages=messages,
                tools=self._tool_schemas,
                system=self._system,
                max_tokens=1024,
            )

            total_input_tokens  += turn.input_tokens
            total_output_tokens += turn.output_tokens

            if turn.stop_reason == "end_turn":
                final_text = turn.text or "I completed the task but could not generate a text response."
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

            elif turn.stop_reason == "tool_calls":
                # Append the assistant's tool-call message (OpenAI format)
                messages = messages + [{
                    "role": "assistant",
                    "content": turn.text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in turn.tool_calls
                    ],
                }]

                # Execute each tool and append results
                for tc in turn.tool_calls:
                    tools_called.append(tc.name)
                    logger.info("Tool call: %s(%s)", tc.name, tc.arguments)

                    tool = self._tool_map.get(tc.name)
                    if tool is None:
                        tool_output = f"Error: tool '{tc.name}' not found."
                        logger.error("Unknown tool: %s", tc.name)
                    else:
                        try:
                            tool_output = tool.execute(**tc.arguments)
                        except Exception as exc:
                            tool_output = f"Error executing {tc.name}: {exc}"
                            logger.exception("Tool execution failed: %s", tc.name)

                    logger.info("Tool result: %s → %r", tc.name, str(tool_output)[:200])

                    messages = messages + [{
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_output),
                    }]

            else:
                logger.warning("Unexpected stop_reason: %s", turn.stop_reason)
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
