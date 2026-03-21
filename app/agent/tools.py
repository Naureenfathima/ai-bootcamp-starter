"""
tools.py — Tool definitions for the autonomous agent.

Each tool has:
  1. A schema dict  (tells the LLM what the tool is and how to call it)
  2. An execute()   (the Python code that actually runs when the LLM calls it)

STUDENT TODO:
  - Add a 'search_web' tool using the Serper or Tavily API.
  - Add a 'read_file' tool so the agent can access local documents.
  - Add a 'send_email' tool using SendGrid or Resend.
  - Implement proper error handling: what should the agent do if a tool fails?
"""

import math
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ── Tool base ─────────────────────────────────────────────────────────────────
class Tool:
    """
    Base class for all agent tools.

    Subclass this and implement:
      - name       (str)
      - description(str)
      - input_schema (dict following Claude's tool-use schema format)
      - execute(**kwargs) → str
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def as_claude_tool(self) -> dict:
        """Return the tool definition in Claude's format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


# ── Calculator ────────────────────────────────────────────────────────────────
class CalculatorTool(Tool):
    """
    Evaluates a mathematical expression safely.

    Example call from Claude:
        {"expression": "sqrt(16) + 2 * (3 + 4)"}
    """

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Use this for any calculation. "
        "Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(). "
        "Example: 'sqrt(144) + 2 ** 10'"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A valid mathematical expression, e.g. 'sqrt(16) * 3'",
            }
        },
        "required": ["expression"],
    }

    # Safe math names available in expressions
    _SAFE_NAMES = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "abs": abs, "round": round, "pi": math.pi, "e": math.e,
    }

    def execute(self, expression: str) -> str:
        try:
            # eval with restricted globals — never exec arbitrary code
            result = eval(expression, {"__builtins__": {}}, self._SAFE_NAMES)  # noqa: S307
            logger.info("Calculator: %s = %s", expression, result)
            return str(result)
        except Exception as exc:
            logger.warning("Calculator error: %s", exc)
            return f"Error evaluating '{expression}': {exc}"


# ── Current Time ──────────────────────────────────────────────────────────────
class CurrentTimeTool(Tool):
    """Returns the current UTC date and time."""

    name = "get_current_time"
    description = "Returns the current UTC date and time. Use this when the user asks about the current time or date."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info("CurrentTimeTool called → %s", now)
        return now


# ── Web Search (stub) ─────────────────────────────────────────────────────────
class WebSearchTool(Tool):
    """
    Searches the web for information.

    STUDENT TODO: Implement this using Tavily or Serper API.
      - Tavily: https://docs.tavily.com/  (has a free tier)
      - Serper: https://serper.dev/       (100 free searches/month)
    """

    name = "search_web"
    description = (
        "Search the internet for current information. Use this when the user asks about "
        "recent events, facts you are unsure about, or anything that may have changed recently."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, e.g. 'latest AI news 2025'",
            }
        },
        "required": ["query"],
    }

    def execute(self, query: str) -> str:
        # STUDENT TODO: Replace this stub with a real API call.
        # Example with Tavily:
        #   from tavily import TavilyClient
        #   client = TavilyClient(api_key=settings.tavily_api_key)
        #   results = client.search(query, max_results=3)
        #   return json.dumps(results["results"])
        logger.warning("WebSearchTool is a stub. Implement it with Tavily or Serper.")
        return (
            f"[STUB] Search results for '{query}' would appear here. "
            "Implement this tool with the Tavily or Serper API to get real results."
        )


# ── Registry ──────────────────────────────────────────────────────────────────
DEFAULT_TOOLS: list[Tool] = [
    CalculatorTool(),
    CurrentTimeTool(),
    WebSearchTool(),
]
