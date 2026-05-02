"""
anti_rag/ — Alternative approaches to Retrieval-Augmented Generation.

RAG is not always the right tool. This module covers four "anti-RAG" strategies:

  CAG  (Cache-Augmented Generation)    — load ALL docs into context, no retrieval step
  KAG  (Knowledge Augmented Generation) — structured knowledge graphs + LLMs
  StructuredKnowledge                  — SQL/API lookups for structured data
  FineTuning                           — bake knowledge into model weights
  HybridRouter                         — pick the right strategy per query

Strategy comparison:

  ┌──────────────┬─────────────────┬──────────────┬───────────────────┐
  │ Strategy     │ Retrieval step  │ Can miss ctx │ Best for          │
  ├──────────────┼─────────────────┼──────────────┼───────────────────┤
  │ RAG          │ Vector search   │ Yes          │ Large unstructured │
  │ CAG          │ None            │ Never        │ Small-med corpora  │
  │ KAG          │ Graph traversal │ Yes (edges)  │ Multi-hop reasoning│
  │ SQL          │ Generated query │ No           │ Structured data    │
  │ Fine-tuning  │ None (baked in) │ Never        │ Stable domain      │
  └──────────────┴─────────────────┴──────────────┴───────────────────┘
"""

from app.anti_rag.cag import CAGPipeline
from app.anti_rag.kag import KAGPipeline
from app.anti_rag.structured_knowledge import StructuredKnowledgePipeline
from app.anti_rag.fine_tuning import FineTuningGuide
from app.anti_rag.router import HybridStrategyRouter

__all__ = [
    "CAGPipeline",
    "KAGPipeline",
    "StructuredKnowledgePipeline",
    "FineTuningGuide",
    "HybridStrategyRouter",
]
