"""
api/agent.py — Agent API routes.

Endpoints:
  POST /agent/run    — Run the agent on a user message
  POST /agent/reset  — Clear the agent's conversation memory
  GET  /agent/tools  — List available tools
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from app.agent.agent import Agent
        _agent = Agent()
    return _agent


# ── Request / Response models ─────────────────────────────────────────────────
class RunRequest(BaseModel):
    message: str = Field(..., description="The user's request for the agent to complete")


class RunResponse(BaseModel):
    response: str
    steps_taken: int
    tools_called: list[str]
    total_latency_ms: float
    input_tokens: int
    output_tokens: int


class ToolInfo(BaseModel):
    name: str
    description: str


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/run", response_model=RunResponse, summary="Run the agent")
async def run_agent(request: RunRequest):
    """
    Send a message to the agent. It will use tools as needed and return
    a final response.

    Example requests:
      - "What is 15% of 2340?"
      - "What time is it right now?"
      - "Search for the latest news about AI and summarise it."
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="'message' must not be empty")

    agent = get_agent()
    try:
        result = agent.run(request.message)
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return RunResponse(
        response=result.response,
        steps_taken=result.steps_taken,
        tools_called=result.tools_called,
        total_latency_ms=result.total_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@router.post("/reset", summary="Reset agent memory")
async def reset_agent():
    """Clear the agent's conversation history and working memory."""
    get_agent().reset()
    return {"status": "ok", "message": "Agent memory cleared."}


@router.get("/tools", response_model=list[ToolInfo], summary="List available tools")
async def list_tools():
    """Returns the tools available to the agent."""
    agent = get_agent()
    return [
        ToolInfo(name=t.name, description=t.description)
        for t in agent._tools
    ]
