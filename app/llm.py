from __future__ import annotations
from typing import Optional
import json
"""
llm.py — Pluggable LLM provider abstraction.

All LLM calls in this project go through get_llm_client(), which returns
a thin wrapper with a single consistent interface:

    client = get_llm_client()
    response = client.chat("What is RAG?", system="You are helpful.", max_tokens=256)
    print(response.text)
    print(response.input_tokens, response.output_tokens)

Supported providers (set LLM_PROVIDER in .env):

  anthropic   — Claude via Anthropic SDK          (default)
  openai      — GPT-4o / GPT-4o-mini via OpenAI
  ollama      — Any model running locally in Ollama (free, no API key)
  groq        — Ultra-fast cloud inference, free tier (llama, mixtral, gemma)
  together    — Together AI, $25 free credit, has Qwen models
  openrouter  — Routes to 200+ models, pay-per-token

Providers marked "OpenAI-compatible" use the openai SDK with a custom base_url.
This means the same code works for Ollama, Groq, Together AI, and OpenRouter
without any extra dependencies.

Setup quick-start:

  Ollama (local, free):
    brew install ollama          # macOS
    ollama pull qwen2.5:7b      # download ~4GB model
    ollama serve                 # starts on http://localhost:11434
    # .env: LLM_PROVIDER=ollama  LLM_MODEL=qwen2.5:7b

  Groq (cloud, free tier):
    Sign up → https://console.groq.com → copy API key
    # .env: LLM_PROVIDER=groq  GROQ_API_KEY=gsk_...  LLM_MODEL=llama-3.1-8b-instant

  Together AI (cloud, $25 free):
    Sign up → https://api.together.ai → copy API key
    # .env: LLM_PROVIDER=together  TOGETHER_API_KEY=...
    #        LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

Systems lesson:
  This abstraction follows the Adapter pattern: each provider speaks a different
  "dialect" (Anthropic vs OpenAI format), but the adapter translates them all
  to the same LLMResponse. Downstream code never needs to know which provider
  is active — it just calls client.chat().

STUDENT TODO:
  - Add HuggingFace Inference Endpoints as a provider.
  - Add response caching: identical (system, user) pairs return cached responses.
  - Add latency + cost logging per provider so students can compare.
"""

import logging
import time
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class AgentTurnResponse:
    """
    Normalised response from chat_with_tools().

    stop_reason is one of:
      "end_turn"   — LLM produced a final text answer, no more tool calls
      "tool_calls" — LLM wants to call one or more tools (see tool_calls list)
    """
    stop_reason: str
    text: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# ── Message format conversion helpers ────────────────────────────────────────
# The agent maintains messages in OpenAI format internally.
# These helpers translate to Anthropic format when needed.

def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        fn = t["function"]
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        })
    return result


def _openai_messages_to_anthropic(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI message list to Anthropic format.

    Key differences:
    - Tool results use role "tool" in OpenAI; Anthropic wraps them in a "user" message.
    - Tool calls live in message["tool_calls"] in OpenAI; Anthropic uses content blocks.
    - Consecutive tool-result messages are grouped into one Anthropic "user" message.
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "system":
            i += 1
            continue  # Anthropic takes system as a separate param

        if role in ("user", "assistant") and not msg.get("tool_calls"):
            result.append({"role": role, "content": msg.get("content") or ""})
            i += 1

        elif role == "assistant" and msg.get("tool_calls"):
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                })
            result.append({"role": "assistant", "content": content})
            i += 1

        elif role == "tool":
            # Collect consecutive tool results into one Anthropic user message
            tool_results = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": messages[i]["tool_call_id"],
                    "content": messages[i]["content"],
                })
                i += 1
            result.append({"role": "user", "content": tool_results})

        else:
            i += 1

    return result


class LLMClient:
    """
    Unified interface for calling any configured LLM provider.

    Usage:
        client = get_llm_client()
        resp = client.chat("Summarise this text: ...", max_tokens=256)
        print(resp.text)
    """

    def chat(
        self,
        user_message: str,
        system: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        raise NotImplementedError

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        max_tokens: int = 1024,
    ) -> AgentTurnResponse:
        """
        One turn of an agentic loop with tool calling.

        Args:
            messages: Conversation so far in OpenAI message format.
            tools:    Available tools in OpenAI function-calling format.
            system:   System prompt.
            max_tokens: Max tokens for this response.

        Returns:
            AgentTurnResponse — either a final text answer or a list of tool calls.
        """
        raise NotImplementedError


# ── Anthropic (Claude) ────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """Claude via the Anthropic SDK."""

    def __init__(self):
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model  = settings.claude_model
        logger.info("LLM: Anthropic / %s", self._model)

    def chat(self, user_message: str, system: str = "", max_tokens: int = 1024) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_message}],
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._model,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        max_tokens: int = 1024,
    ) -> AgentTurnResponse:
        anthropic_messages = _openai_messages_to_anthropic(messages)
        anthropic_tools    = _openai_tools_to_anthropic(tools)

        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            tools=anthropic_tools,
            messages=anthropic_messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        if response.stop_reason == "tool_use":
            tool_calls = [
                ToolCall(id=b.id, name=b.name, arguments=b.input)
                for b in response.content
                if b.type == "tool_use"
            ]
            return AgentTurnResponse(
                stop_reason="tool_calls",
                text=None,
                tool_calls=tool_calls,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        text = next(
            (b.text for b in response.content if hasattr(b, "text")), ""
        )
        return AgentTurnResponse(
            stop_reason="end_turn",
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ── OpenAI-compatible (OpenAI, Ollama, Groq, Together, OpenRouter) ────────────

class OpenAICompatibleClient(LLMClient):
    """
    Covers any provider that speaks the OpenAI Chat Completions API.

    All of these work with the same openai SDK — just different base_url + api_key:
      - OpenAI:     base_url=None (default)
      - Ollama:     base_url=http://localhost:11434/v1  api_key=ollama
      - Groq:       base_url=https://api.groq.com/openai/v1
      - Together:   base_url=https://api.together.xyz/v1
      - OpenRouter: base_url=https://openrouter.ai/api/v1
    """

    def __init__(self, base_url:Optional[ str], api_key: str, model: str, provider_name: str):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model  = model
        self._name   = provider_name
        logger.info("LLM: %s / %s (base_url=%s)", provider_name, model, base_url or "default")

    def chat(self, user_message: str, system: str = "", max_tokens: int = 1024) -> LLMResponse:
        t0 = time.perf_counter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=self._model,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        max_tokens: int = 1024,
    ) -> AgentTurnResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=all_messages,
            tools=tools,
        )

        choice  = response.choices[0]
        message = choice.message

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            return AgentTurnResponse(
                stop_reason="tool_calls",
                text=message.content,
                tool_calls=tool_calls,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return AgentTurnResponse(
            stop_reason="end_turn",
            text=message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client() -> LLMClient:
    """
    Return the LLM client configured by LLM_PROVIDER in .env.

    Reads these settings from .env (see app/config.py):
      LLM_PROVIDER   — which provider to use (default: "anthropic")
      LLM_MODEL      — override the model name
      GROQ_API_KEY   — required when LLM_PROVIDER=groq
      TOGETHER_API_KEY — required when LLM_PROVIDER=together
      OPENROUTER_API_KEY — required when LLM_PROVIDER=openrouter
      OLLAMA_BASE_URL  — Ollama server URL (default: http://localhost:11434/v1)
    """
    provider = getattr(settings, "llm_provider", "anthropic").lower()

    if provider == "anthropic":
        return AnthropicClient()

    if provider == "openai":
        model = getattr(settings, "llm_model", None) or settings.openai_chat_model
        return OpenAICompatibleClient(
            base_url=None,
            api_key=settings.openai_api_key,
            model=model,
            provider_name="OpenAI",
        )

    if provider == "ollama":
        base_url = getattr(settings, "ollama_base_url", "http://localhost:11434/v1")
        model    = getattr(settings, "llm_model", "qwen2.5:7b")
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key="ollama",           # Ollama ignores the key but SDK requires one
            model=model,
            provider_name="Ollama",
        )

    if provider == "groq":
        model = getattr(settings, "llm_model", "llama-3.1-8b-instant")
        api_key = getattr(settings, "groq_api_key", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        return OpenAICompatibleClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model,
            provider_name="Groq",
        )

    if provider == "together":
        model = getattr(settings, "llm_model", "Qwen/Qwen2.5-7B-Instruct")
        api_key = getattr(settings, "together_api_key", "")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is not set in .env")
        return OpenAICompatibleClient(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model=model,
            provider_name="Together AI",
        )

    if provider == "openrouter":
        model = getattr(settings, "llm_model", "qwen/qwen-2.5-7b-instruct")
        api_key = getattr(settings, "openrouter_api_key", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        return OpenAICompatibleClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            provider_name="OpenRouter",
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Choose: anthropic | openai | ollama | groq | together | openrouter"
    )
