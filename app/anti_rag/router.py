"""
anti_rag/router.py — Hybrid strategy router.

Not every question deserves the same retrieval strategy. This router
classifies each incoming query and routes it to the right pipeline:

  ┌──────────────────────────────────────┬────────────────────┐
  │ Query type                           │ Route to           │
  ├──────────────────────────────────────┼────────────────────┤
  │ "Who manages X?" / "What owns Y?"   │ KAG (graph)        │
  │ "How many orders..." / "Total rev"  │ SQL (structured)   │
  │ "Compare all / synthesize across"   │ CAG (full context) │
  │ "What does doc say about X?"        │ RAG (vector)       │
  │ "Explain concept X"                 │ LLM directly       │
  └──────────────────────────────────────┴────────────────────┘

STUDENT TODO:
  - Replace keyword routing with a classifier (fine-tuned or few-shot prompted).
  - Add confidence scores so the router can escalate to a fallback.
  - Add A/B testing: route 10% of queries to an alternative strategy and compare.
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Strategy = Literal["kag", "sql", "cag", "rag", "direct"]


@dataclass
class RoutingDecision:
    strategy: Strategy
    confidence: float   # 0.0 – 1.0
    reason: str


# Keyword patterns for each strategy
_KAG_PATTERNS = [
    r"\bwho\b.*(manage|own|lead|report|depend)",
    r"\bwhat.*(team|service|component|system)\b",
    r"\brelationship\b",
    r"\bdepend(s|ency|encies)?\b",
    r"\bhierarch",
    r"\borgani[sz]ation",
]

_SQL_PATTERNS = [
    r"\bhow many\b",
    r"\btotal\b",
    r"\bsum\b",
    r"\baverage\b|\bavg\b",
    r"\bcount\b",
    r"\bmost (recent|popular|common)\b",
    r"\blatest\b",
    r"\btrend\b",
    r"\bcompare\b.*\bperiod",
    r"\bQ[1-4]\b",
    r"\brevenue\b|\bsales\b|\border",
]

_CAG_PATTERNS = [
    r"\bcompare\b.*(all|every|each)",
    r"\bacross all\b",
    r"\bfrom all\b",
    r"\beverything about\b",
    r"\bsynthesi[sz]e\b",
    r"\boverview of all\b",
    r"\bsummar(y|ise|ize) (all|every|the (whole|entire|full))",
    r"\blist all\b",
]

_RAG_PATTERNS = [
    r"\baccording to\b",
    r"\bin the doc",
    r"\bwhat does.*(say|mention|state|describe)",
    r"\bsummarise\b|\bsummarize\b",
    r"\bexplain.*document",
    r"\bfind in\b",
]


class HybridStrategyRouter:
    """
    Routes an incoming query to the most appropriate retrieval strategy.

    This is the "meta-level" of the anti-RAG system — the decision layer
    that decides which approach to use before any retrieval happens.

    Usage:
        router = HybridStrategyRouter()
        decision = router.route("How many orders came in last week?")
        print(decision.strategy)   # "sql"
        print(decision.reason)     # "Aggregation query detected → SQL"
    """

    def route(self, query: str) -> RoutingDecision:
        """
        Classify a query and return the recommended strategy.

        Args:
            query: The user's natural language question.

        Returns:
            RoutingDecision with strategy, confidence, and reason.
        """
        q = query.lower()

        # Check KAG patterns (entity relationship questions)
        for pattern in _KAG_PATTERNS:
            if re.search(pattern, q):
                logger.info("Router: KAG ← '%s'", query[:60])
                return RoutingDecision(
                    strategy="kag",
                    confidence=0.8,
                    reason=f"Entity/relationship query detected → Knowledge Graph"
                )

        # Check SQL patterns (aggregation / metric questions)
        for pattern in _SQL_PATTERNS:
            if re.search(pattern, q):
                logger.info("Router: SQL ← '%s'", query[:60])
                return RoutingDecision(
                    strategy="sql",
                    confidence=0.85,
                    reason="Aggregation/metric query detected → SQL lookup"
                )

        # Check CAG patterns (cross-document synthesis / full-context questions)
        for pattern in _CAG_PATTERNS:
            if re.search(pattern, q):
                logger.info("Router: CAG ← '%s'", query[:60])
                return RoutingDecision(
                    strategy="cag",
                    confidence=0.75,
                    reason="Cross-document synthesis detected → CAG (full context)"
                )

        # Check RAG patterns (document-grounded questions)
        for pattern in _RAG_PATTERNS:
            if re.search(pattern, q):
                logger.info("Router: RAG ← '%s'", query[:60])
                return RoutingDecision(
                    strategy="rag",
                    confidence=0.75,
                    reason="Document-grounded query detected → Vector RAG"
                )

        # Default: direct LLM answer (general knowledge / reasoning)
        logger.info("Router: DIRECT ← '%s'", query[:60])
        return RoutingDecision(
            strategy="direct",
            confidence=0.6,
            reason="General knowledge or reasoning query → direct LLM"
        )

    def explain(self, query: str) -> dict:
        """
        Return a detailed explanation of the routing decision for debugging.
        Useful for teaching students how the router works.
        """
        decision = self.route(query)
        return {
            "query": query,
            "strategy": decision.strategy,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "strategy_descriptions": {
                "kag":    "Knowledge Augmented Generation — graph traversal for entity/relationship questions",
                "sql":    "Text-to-SQL — precise structured queries for metrics and aggregations",
                "cag":    "Cache-Augmented Generation — full context window for cross-document synthesis",
                "rag":    "Vector RAG — semantic retrieval for document-grounded questions",
                "direct": "Direct LLM — no retrieval needed, answer from model knowledge",
            }
        }
