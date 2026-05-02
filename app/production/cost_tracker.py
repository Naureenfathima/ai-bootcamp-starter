"""
production/cost_tracker.py — Real-time LLM cost tracking and budget controls.

LLM costs are easy to underestimate. A single agent with a loop bug can
run up hundreds of dollars in minutes. This module:

  1. Tracks token usage per user, per endpoint, per model in real-time
  2. Enforces budget caps (hard limit = reject; soft limit = warn)
  3. Provides cost optimisation recommendations

Cost optimisation strategies covered in this course:
  - Semantic caching (30-50% cost reduction for repetitive queries)
  - Model selection: use Haiku for classification, Sonnet for generation
  - Prompt compression: remove redundant context before sending
  - Retrieval optimisation: retrieve fewer but higher-quality chunks
  - Response length control: set appropriate max_tokens

STUDENT TODO:
  - Persist usage data to a database for billing and analytics.
  - Add Slack/email alerts when a user exceeds 80% of their budget.
  - Build a cost dashboard using the data:build-dashboard skill.
  - Implement prompt compression: measure tokens before/after and log savings.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Pricing (as of 2025 — update as models change) ───────────────────────────
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {
        "input_per_1k":  0.003,    # $3 per 1M input tokens
        "output_per_1k": 0.015,    # $15 per 1M output tokens
    },
    "claude-3-haiku-20240307": {
        "input_per_1k":  0.00025,
        "output_per_1k": 0.00125,
    },
    "claude-opus-4-6": {
        "input_per_1k":  0.015,
        "output_per_1k": 0.075,
    },
    "text-embedding-3-small": {
        "input_per_1k":  0.00002,
        "output_per_1k": 0.0,
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost in USD for a single LLM call."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-3-5-sonnet-20241022"])
    return (
        (input_tokens  / 1000) * pricing["input_per_1k"]
        + (output_tokens / 1000) * pricing["output_per_1k"]
    )


@dataclass
class UsageRecord:
    """A single API call usage record."""
    user_id: str
    model: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CostBudget:
    """Budget configuration for a user or project."""
    user_id: str
    daily_limit_usd: float    # hard limit: requests rejected above this
    soft_limit_usd: float     # soft limit: alert triggered at this level
    monthly_limit_usd: float = 0.0   # 0 = no monthly limit


class CostTracker:
    """
    Real-time cost tracker for LLM API usage.

    Usage:
        tracker = CostTracker()
        tracker.set_budget("user_123", CostBudget(
            user_id="user_123",
            daily_limit_usd=1.00,
            soft_limit_usd=0.80,
        ))

        # After each LLM call:
        tracker.record(
            user_id="user_123",
            model="claude-3-5-sonnet-20241022",
            endpoint="/rag/query",
            input_tokens=500,
            output_tokens=200,
        )

        # Check if the user is within budget:
        tracker.check_budget("user_123")   # raises if over hard limit
    """

    def __init__(self):
        self._records: list[UsageRecord] = []
        self._budgets: dict[str, CostBudget] = {}
        logger.info("CostTracker initialised")

    def set_budget(self, user_id: str, budget: CostBudget) -> None:
        """Set budget limits for a user."""
        self._budgets[user_id] = budget
        logger.info("CostTracker: set budget for %s (daily=$%.2f)",
                    user_id, budget.daily_limit_usd)

    def record(
        self,
        user_id: str,
        model: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record a usage event and return the cost in USD.

        Call this after every LLM API call.
        """
        cost = calculate_cost(model, input_tokens, output_tokens)
        record = UsageRecord(
            user_id=user_id,
            model=model,
            endpoint=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._records.append(record)
        logger.debug("Cost recorded: user=%s model=%s cost=$%.5f tokens=%d+%d",
                     user_id, model, cost, input_tokens, output_tokens)

        # Check soft limit
        budget = self._budgets.get(user_id)
        if budget:
            daily_cost = self.daily_cost(user_id)
            if daily_cost >= budget.soft_limit_usd:
                logger.warning(
                    "SOFT LIMIT: %s reached $%.4f (soft=$%.2f daily=$%.2f)",
                    user_id, daily_cost, budget.soft_limit_usd, budget.daily_limit_usd
                )
        return cost

    def check_budget(self, user_id: str) -> None:
        """
        Check if a user is within their daily budget.
        Raises RuntimeError if the hard limit is exceeded.
        Call this BEFORE making an LLM call.
        """
        budget = self._budgets.get(user_id)
        if not budget:
            return  # no budget set → unlimited

        daily_cost = self.daily_cost(user_id)
        if daily_cost >= budget.daily_limit_usd:
            raise RuntimeError(
                f"Daily budget exceeded for {user_id}: "
                f"${daily_cost:.4f} >= ${budget.daily_limit_usd:.2f}. "
                f"Resets at midnight UTC."
            )

    def daily_cost(self, user_id: str) -> float:
        """Total cost for a user today."""
        today_start = time.time() - (time.time() % 86400)
        return sum(
            r.cost_usd for r in self._records
            if r.user_id == user_id and r.timestamp >= today_start
        )

    def user_summary(self, user_id: str) -> dict:
        """Cost summary for a specific user."""
        user_records = [r for r in self._records if r.user_id == user_id]
        budget = self._budgets.get(user_id)
        daily = self.daily_cost(user_id)

        by_model   = {}
        by_endpoint = {}
        for r in user_records:
            by_model[r.model]       = by_model.get(r.model, 0) + r.cost_usd
            by_endpoint[r.endpoint] = by_endpoint.get(r.endpoint, 0) + r.cost_usd

        return {
            "user_id": user_id,
            "daily_cost_usd": round(daily, 5),
            "daily_limit_usd": budget.daily_limit_usd if budget else None,
            "budget_used_pct": round(daily / budget.daily_limit_usd * 100, 1) if budget else None,
            "total_calls": len(user_records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in user_records),
            "cost_by_model":    {k: round(v, 5) for k, v in by_model.items()},
            "cost_by_endpoint": {k: round(v, 5) for k, v in by_endpoint.items()},
        }

    def optimisation_tips(self, user_id: str) -> list[str]:
        """
        Generate cost optimisation recommendations based on usage patterns.
        These are the strategies we teach in the course.
        """
        tips = []
        user_records = [r for r in self._records if r.user_id == user_id]
        if not user_records:
            return ["No usage data available yet."]

        avg_output = sum(r.output_tokens for r in user_records) / len(user_records)
        avg_input  = sum(r.input_tokens  for r in user_records) / len(user_records)

        if avg_output > 800:
            tips.append(
                "Your average output is high (>800 tokens). Consider setting a lower "
                "max_tokens limit or asking the model to be more concise."
            )
        if avg_input > 3000:
            tips.append(
                "Your average input is large (>3000 tokens). Consider: (1) retrieving "
                "fewer RAG chunks, (2) compressing your system prompt, or (3) using "
                "semantic caching for repeated queries."
            )

        models_used = set(r.model for r in user_records)
        if "claude-opus-4-6" in models_used:
            tips.append(
                "You are using Claude Opus, which costs 5x more than Sonnet. "
                "Route simple classification or summarisation tasks to Claude Haiku "
                "and reserve Opus for complex reasoning tasks."
            )

        if not tips:
            tips.append(
                "Your usage looks efficient. Keep an eye on cache hit rate — "
                "semantic caching can reduce costs by 30-50% for repetitive workloads."
            )
        return tips

    def global_stats(self) -> dict:
        """Aggregate stats across all users."""
        if not self._records:
            return {"total_cost_usd": 0.0, "total_calls": 0}

        return {
            "total_cost_usd": round(sum(r.cost_usd for r in self._records), 4),
            "total_calls": len(self._records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in self._records),
            "unique_users": len(set(r.user_id for r in self._records)),
            "cost_by_model": {
                model: round(sum(r.cost_usd for r in self._records if r.model == model), 4)
                for model in set(r.model for r in self._records)
            },
        }
