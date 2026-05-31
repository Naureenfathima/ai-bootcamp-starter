"""
graph_rag.py — Document-level Graph RAG (Microsoft GraphRAG pattern).

Standard RAG retrieves a few similar chunks. This is fine for local, specific
questions but fails on global synthesis questions like "What are the main themes
across all our documentation?" — you cannot answer that by retrieving 3 chunks.

GraphRAG (Edge et al., 2024) solves this by:
  1. Building a knowledge graph FROM the document corpus (entities + relations)
  2. Grouping entities into communities (clusters of related nodes)
  3. Writing LLM summaries for each community
  4. At query time: routing global queries to community summaries
                    routing local queries to standard chunk retrieval

Difference from KAG (app/anti_rag/kag.py):
  KAG:      you PRE-DEFINE a structured graph, then query it.
  GraphRAG: you BUILD the graph from unstructured documents, use it for synthesis.

Architecture:

  Documents
    │
    ▼ [Entity extraction per chunk]
  Entity Graph (nodes + edges)
    │
    ▼ [Connected-component clustering]
  Communities
    │
    ▼ [LLM summarisation per community]
  Community Summaries (Tier 1 — global queries)
    │
    ┌─────────────┴──────────────────────────┐
    │ global query                           │ local query
    ▼                                        ▼
  Community summaries as context          Chunk retrieval (standard RAG)

Systems lesson:
  GraphRAG exemplifies "pay at index time, save at query time". Building the
  graph and community summaries is expensive, but it is done once offline.
  Every query then benefits from pre-computed, semantically rich summaries.

Reference: https://arxiv.org/abs/2404.16130

STUDENT TODO:
  - Replace connected-component clustering with Louvain community detection
    (pip install networkx python-louvain) for true community structure.
  - Add entity disambiguation: merge "GPT-4" and "gpt4" into one node.
  - Persist the graph to JSON/disk so you don't rebuild on every restart.
  - Add a query classifier that uses an LLM (not just keyword heuristics)
    to decide local vs global routing.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from app.config import settings
from app.rag.pipeline import RAGPipeline
from app.rag.ingestion import chunk_text

logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    """A node in the document-derived knowledge graph."""
    name: str
    entity_type: str       # Person, Technology, Concept, Team, Service, …
    description: str       # extracted from source text
    source_chunks: list[int] = field(default_factory=list)


@dataclass
class GraphRelation:
    """A directed edge between two entities."""
    source: str
    target: str
    relation: str          # uses, manages, depends_on, …


@dataclass
class Community:
    """A cluster of related entities with a synthesised summary."""
    community_id: int
    entities: list[str]    # entity names
    summary: str           # LLM-generated summary of this cluster


@dataclass
class GraphRAGResult:
    """Result from GraphRAGPipeline.query()."""
    question: str
    answer: str
    query_type: str                # "local" | "global"
    communities_used: list[int]
    entities_found: list[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float


class GraphRAGPipeline:
    """
    Document-derived Graph RAG.

    Build phase (offline, run once per document set):
        pipeline = GraphRAGPipeline(rag=base_rag)
        stats = pipeline.build(text, source="architecture.txt")

    Query phase (online, fast):
        result = pipeline.query("Summarise all the key components mentioned.")
        result = pipeline.query("What does the RAG Service depend on?")
    """

    def __init__(self, rag: RAGPipeline):
        self._rag         = rag
        self._client      = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._entities:   dict[str, GraphEntity] = {}
        self._relations:  list[GraphRelation]    = []
        self._communities: list[Community]       = []
        logger.info("GraphRAGPipeline initialised")

    # ── Build Phase ────────────────────────────────────────────────────────────

    def build(self, text: str, source: str) -> dict:
        """
        Full offline build pipeline:
          1. Ingest into base RAG (enables local queries)
          2. Extract entities and relations from each chunk
          3. Cluster entities into communities
          4. Summarise each community

        This is the expensive step — run it once, query many times.
        """
        t0 = time.perf_counter()

        # Step 1: base RAG ingestion (for local query path)
        self._rag.ingest_text(text, source=source)

        # Step 2: entity extraction
        chunks = chunk_text(text, source=source)
        for idx, chunk in enumerate(chunks):
            self._extract_entities(chunk.text, chunk_index=idx)

        # Step 3: community detection
        self._build_communities()

        # Step 4: summarise communities
        self._summarise_communities()

        build_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("GraphRAG build: %d entities, %d relations, %d communities, %sms",
                    len(self._entities), len(self._relations), len(self._communities), build_ms)
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "communities": len(self._communities),
            "build_latency_ms": build_ms,
        }

    def _extract_entities(self, text: str, chunk_index: int) -> None:
        """One LLM call per chunk to extract entities and relations."""
        prompt = f"""Extract named entities and relationships from the text.

Return this exact JSON structure:
{{
  "entities": [
    {{"name": "EntityName", "type": "Person|Technology|Concept|Team|Service|Other", "description": "one sentence"}}
  ],
  "relations": [
    {{"source": "EntityA", "target": "EntityB", "relation": "uses|manages|depends_on|creates|owns"}}
  ]
}}

Rules:
- Entity names must be exact proper nouns or technical terms from the text
- Only include entities EXPLICITLY mentioned — do not infer
- Relations must be directed (source acts on target)

Text:
{text[:1500]}"""

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("GraphRAG: entity parse error for chunk %d", chunk_index)
            return

        for e in data.get("entities", []):
            name = e.get("name", "").strip()
            if not name:
                continue
            if name in self._entities:
                self._entities[name].source_chunks.append(chunk_index)
            else:
                self._entities[name] = GraphEntity(
                    name=name,
                    entity_type=e.get("type", "Other"),
                    description=e.get("description", ""),
                    source_chunks=[chunk_index],
                )

        for r in data.get("relations", []):
            src, tgt, rel = r.get("source", ""), r.get("target", ""), r.get("relation", "")
            if src and tgt and rel:
                self._relations.append(GraphRelation(source=src, target=tgt, relation=rel))

    def _build_communities(self) -> None:
        """
        Group entities into communities via BFS-based connected components.

        STUDENT TODO: replace with Louvain community detection (networkx) for
        proper hierarchical clustering on large graphs.
        """
        adjacency: dict[str, set[str]] = {e: set() for e in self._entities}
        for rel in self._relations:
            if rel.source in adjacency:
                adjacency[rel.source].add(rel.target)
            if rel.target in adjacency:
                adjacency[rel.target].add(rel.source)

        visited: set[str] = set()
        components: list[list[str]] = []

        for entity in self._entities:
            if entity in visited:
                continue
            component: list[str] = []
            queue = [entity]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(adjacency.get(node, set()) - visited)
            components.append(component)

        self._communities = [
            Community(community_id=i, entities=comp, summary="")
            for i, comp in enumerate(components)
        ]
        logger.info("GraphRAG: %d communities from %d entities", len(self._communities), len(self._entities))

    def _summarise_communities(self) -> None:
        """Generate one LLM summary per community."""
        for community in self._communities:
            entity_lines = [
                f"- {n} ({self._entities[n].entity_type}): {self._entities[n].description}"
                for n in community.entities
                if n in self._entities
            ]
            community_set = set(community.entities)
            relation_lines = [
                f"  {r.source} -{r.relation}→ {r.target}"
                for r in self._relations
                if r.source in community_set and r.target in community_set
            ]
            context = "\n".join(entity_lines)
            if relation_lines:
                context += "\nRelationships:\n" + "\n".join(relation_lines[:10])

            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Summarise this cluster of related entities in 2-3 sentences. "
                        f"Explain what they represent together and how they relate.\n\n{context}"
                    ),
                }],
            )
            community.summary = response.content[0].text.strip()
            logger.debug("GraphRAG: community %d summarised", community.community_id)

    # ── Query Phase ────────────────────────────────────────────────────────────

    def query(self, question: str, query_type: str = "auto") -> GraphRAGResult:
        """
        Answer using community summaries (global) or chunk retrieval (local).

        Global queries: "What are the main themes?", "Summarise all components."
        Local queries:  "What does ServiceX depend on?", "Who manages TeamY?"

        Args:
            question:   The user's question.
            query_type: "global" | "local" | "auto" (heuristic classification).
        """
        t0 = time.perf_counter()

        if query_type == "auto":
            query_type = self._classify(question)

        logger.info("GraphRAG query: type=%s question='%s'", query_type, question[:60])

        if query_type == "global":
            answer, communities_used, entities, in_tok, out_tok = self._global_query(question)
        else:
            answer, communities_used, entities, in_tok, out_tok = self._local_query(question)

        return GraphRAGResult(
            question=question,
            answer=answer,
            query_type=query_type,
            communities_used=communities_used,
            entities_found=entities,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def _classify(self, question: str) -> str:
        """
        Heuristic: detect global vs local query intent.

        Global indicators: summarise, overview, main, theme, pattern, across, all.
        Local indicators: specific entity names present in the question.

        STUDENT TODO: replace with an LLM classifier for higher accuracy.
        """
        global_words = {
            "summarise", "summary", "overview", "main", "theme", "pattern",
            "across", "all", "overall", "generally", "compare", "list all",
        }
        q_lower = question.lower()
        if any(kw in q_lower for kw in global_words):
            return "global"
        for entity_name in self._entities:
            if entity_name.lower() in q_lower:
                return "local"
        return "global"

    def _global_query(self, question: str) -> tuple[str, list[int], list[str], int, int]:
        """Answer using community summaries — for synthesis/overview questions."""
        if not self._communities:
            return ("Build the graph first with build().", [], [], 0, 0)

        context = "\n\n".join(
            f"[Community {c.community_id}]: {c.summary}"
            for c in self._communities if c.summary
        )
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=(
                "You synthesise information from multiple community summaries. "
                "Be comprehensive and reference community IDs when relevant."
            ),
            messages=[{
                "role": "user",
                "content": f"Community summaries:\n{context}\n\nQuestion: {question}",
            }],
        )
        return (
            response.content[0].text,
            [c.community_id for c in self._communities],
            list(self._entities)[:10],
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _local_query(self, question: str) -> tuple[str, list[int], list[str], int, int]:
        """Answer using chunk retrieval — for specific fact questions."""
        results = self._rag.retrieve(question)
        context = "\n---\n".join(r.chunk.text for r in results)
        q_lower = question.lower()
        found_entities = [n for n in self._entities if n.lower() in q_lower]

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            system="Answer precisely using only the provided context chunks.",
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }],
        )
        return (
            response.content[0].text,
            [],
            found_entities,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    def graph_stats(self) -> dict:
        type_counts: dict[str, int] = {}
        for e in self._entities.values():
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "communities": len(self._communities),
            "entity_types": type_counts,
        }
