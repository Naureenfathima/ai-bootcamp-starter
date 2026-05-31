"""
anti_rag/kag.py — Knowledge Augmented Generation (KAG).

KAG is an "anti-RAG" approach that replaces unstructured vector search with
a structured knowledge graph. Instead of embedding chunks and doing cosine
similarity search, we build a graph of entities and relationships that the
LLM can traverse logically.

Why KAG instead of RAG?
  - RAG retrieves chunks that are *similar* to the query (fuzzy, statistical).
  - KAG retrieves facts that are *logically connected* to the query (precise).
  - KAG excels at multi-hop reasoning: "Who manages the team that owns service X?"
  - RAG excels at open-domain Q&A over large unstructured corpora.

Architecture:
  Documents → Entity Extraction (Claude) → Knowledge Graph
  Query     → Entity Recognition → Graph Traversal → Context → LLM

STUDENT TODO:
  - Replace the in-memory graph with a real graph DB (Neo4j, Amazon Neptune).
  - Add relation extraction: identify typed edges (e.g. "works_at", "owns").
  - Add entity disambiguation: merge "GPT-4" and "gpt4" into one node.
  - Combine KAG with RAG (hybrid): use graph for structured facts,
    vector search for unstructured prose.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import openai
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)


# ── Graph Data Structures ─────────────────────────────────────────────────────
@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    label: str          # e.g. "Person", "Service", "Concept"
    properties: dict = field(default_factory=dict)


@dataclass
class Relation:
    """A directed edge between two entities."""
    source_id: str
    target_id: str
    relation_type: str  # e.g. "MANAGES", "DEPENDS_ON", "IS_A"
    properties: dict = field(default_factory=dict)


@dataclass
class KAGResult:
    """Result returned by the KAG pipeline."""
    answer: str
    entities_used: list[str]
    reasoning_path: list[str]   # the chain of graph hops taken
    input_tokens: int
    output_tokens: int


# ── In-Memory Knowledge Graph ─────────────────────────────────────────────────
class InMemoryKnowledgeGraph:
    """
    Minimal knowledge graph backed by plain Python dicts.
    Good for demos and testing — swap for Neo4j in production.

    Production note:
        Neo4j Cypher query equivalent of a 2-hop traversal:
            MATCH (start {id: $entity_id})-[*1..2]->(related)
            RETURN related, relationships(path)
    """

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity
        logger.debug("KG: added entity %s (%s)", entity.id, entity.label)

    def add_relation(self, relation: Relation) -> None:
        self._relations.append(relation)
        logger.debug("KG: added relation %s -[%s]-> %s",
                     relation.source_id, relation.relation_type, relation.target_id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_neighbours(self, entity_id: str, max_hops: int = 2) -> list[tuple[Entity, str]]:
        """
        Return all entities reachable within max_hops from entity_id.
        Returns list of (entity, relation_type) tuples.
        """
        visited = set()
        frontier = [entity_id]
        result = []

        for _ in range(max_hops):
            next_frontier = []
            for src in frontier:
                for rel in self._relations:
                    if rel.source_id == src and rel.target_id not in visited:
                        tgt = self._entities.get(rel.target_id)
                        if tgt:
                            result.append((tgt, rel.relation_type))
                            visited.add(rel.target_id)
                            next_frontier.append(rel.target_id)
            frontier = next_frontier

        return result

    def search_entities(self, keyword: str) -> list[Entity]:
        """Simple keyword search over entity IDs and properties."""
        keyword = keyword.lower()
        return [
            e for e in self._entities.values()
            if keyword in e.id.lower()
            or keyword in str(e.properties).lower()
        ]

    def to_context_string(self, entity_ids: list[str]) -> str:
        """Format a subgraph as a readable context string for the LLM."""
        lines = []
        for eid in entity_ids:
            entity = self._entities.get(eid)
            if not entity:
                continue
            lines.append(f"Entity: {entity.id} [{entity.label}]")
            for k, v in entity.properties.items():
                lines.append(f"  {k}: {v}")
            neighbours = self.get_neighbours(eid, max_hops=1)
            for neighbour, rel_type in neighbours:
                lines.append(f"  -{rel_type}-> {neighbour.id}")
            lines.append("")
        return "\n".join(lines)

    def entity_count(self) -> int:
        return len(self._entities)

    def relation_count(self) -> int:
        return len(self._relations)


# ── KAG Pipeline ──────────────────────────────────────────────────────────────
class KAGPipeline:
    """
    Knowledge Augmented Generation pipeline.

    Two-phase workflow:
      Phase 1 — Build: extract entities and relations from documents.
      Phase 2 — Query: identify relevant entities, traverse the graph, generate answer.

    Usage:
        pipeline = KAGPipeline()

        # Build the knowledge graph from documents
        pipeline.extract_and_store(
            "Alice manages the Platform team. Platform team owns the RAG Service. "
            "RAG Service depends on the Vector Database."
        )

        # Query using graph traversal instead of vector search
        result = pipeline.query("What does Alice's team own?")
        print(result.answer)
    """

    ENTITY_EXTRACTION_PROMPT = """
You are a knowledge graph builder. Extract entities and relationships from the text.

Return JSON in this exact format:
{{
  "entities": [
    {{"id": "alice", "label": "Person", "properties": {{"name": "Alice", "role": "Manager"}}}},
    {{"id": "platform_team", "label": "Team", "properties": {{"name": "Platform Team"}}}}
  ],
  "relations": [
    {{"source_id": "alice", "target_id": "platform_team", "relation_type": "MANAGES"}}
  ]
}}

Rules:
- IDs must be lowercase_snake_case, no spaces
- Relation types must be UPPERCASE_SNAKE_CASE
- Only extract what is explicitly stated in the text
- Common relation types: MANAGES, OWNS, DEPENDS_ON, IS_A, PART_OF, USES, CREATED_BY

Text to extract from:
{text}
"""

    def __init__(self):
        # Prefer OpenAI if a key is present; fall back to Anthropic (Claude).
        # The else branch was previously None, causing AttributeError at query time.
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._graph  = InMemoryKnowledgeGraph()
        logger.info("KAGPipeline initialised (provider=%s)",
                    "openai" if isinstance(self._client, OpenAI) else "anthropic")

    # ── Phase 1: Build ────────────────────────────────────────────────────────
    def extract_and_store(self, text: str) -> dict:
        """
        Use Claude to extract entities and relations from text, then store
        them in the knowledge graph.

        Returns:
            Summary of entities and relations added.
        """
        logger.info("KAG: extracting entities from %d chars", len(text))

        if isinstance(self._client, OpenAI):
            response = self._client.chat.completions.create(
                model=settings.openai_chat_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": self.ENTITY_EXTRACTION_PROMPT.format(text=text)}]
            )
            raw = response.choices[0].message.content.strip()
        else:
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": self.ENTITY_EXTRACTION_PROMPT.format(text=text)
                }]
            )
            raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("KAG: failed to parse entity extraction JSON: %s", e)
            return {"entities_added": 0, "relations_added": 0, "error": str(e)}

        entities_added = 0
        for e in data.get("entities", []):
            entity = Entity(
                id=e["id"],
                label=e.get("label", "Unknown"),
                properties=e.get("properties", {})
            )
            self._graph.add_entity(entity)
            entities_added += 1

        relations_added = 0
        for r in data.get("relations", []):
            relation = Relation(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                properties=r.get("properties", {})
            )
            self._graph.add_relation(relation)
            relations_added += 1

        logger.info("KAG: added %d entities, %d relations", entities_added, relations_added)
        return {
            "entities_added": entities_added,
            "relations_added": relations_added,
            "total_entities": self._graph.entity_count(),
            "total_relations": self._graph.relation_count(),
        }

    # ── Phase 2: Query ────────────────────────────────────────────────────────
    def query(self, question: str, max_hops: int = 2) -> KAGResult:
        """
        Answer a question using graph traversal instead of vector similarity.

        Steps:
          1. Identify which entities in the question are in the graph
          2. Traverse the graph from those entities (up to max_hops)
          3. Format the subgraph as context
          4. Generate an answer with Claude

        Args:
            question:  The user's question.
            max_hops:  How many relationship hops to traverse from seed entities.

        Returns:
            KAGResult with answer and reasoning path.
        """
        # Step 1: Find seed entities mentioned in the question
        words = question.lower().split()
        seed_entities = []
        reasoning_path = []

        for word in words:
            matches = self._graph.search_entities(word)
            for match in matches:
                if match.id not in seed_entities:
                    seed_entities.append(match.id)
                    reasoning_path.append(f"Found entity: {match.id} [{match.label}]")

        logger.info("KAG: question='%s' → seed_entities=%s", question[:60], seed_entities)

        if not seed_entities:
            return KAGResult(
                answer="I couldn't find any relevant entities in the knowledge graph for this question.",
                entities_used=[],
                reasoning_path=["No matching entities found in graph"],
                input_tokens=0,
                output_tokens=0,
            )

        # Step 2: Expand to neighbours
        all_entity_ids = list(seed_entities)
        for eid in seed_entities:
            neighbours = self._graph.get_neighbours(eid, max_hops=max_hops)
            for entity, rel_type in neighbours:
                if entity.id not in all_entity_ids:
                    all_entity_ids.append(entity.id)
                    reasoning_path.append(f"Traversed: {eid} -[{rel_type}]-> {entity.id}")

        # Step 3: Build context from the subgraph
        context = self._graph.to_context_string(all_entity_ids)
        reasoning_path.append(f"Assembled context from {len(all_entity_ids)} entities")

        # Step 4: Generate answer
        system_prompt = (
            "You are a precise assistant answering questions using a structured knowledge graph. "
            "Use ONLY the provided graph context to answer. Be direct and cite the entity names. "
            "If the answer cannot be determined from the graph, say so clearly."
        )
        user_prompt = f"Knowledge Graph Context:\n{context}\n\nQuestion: {question}"

        if isinstance(self._client, OpenAI):
            response = self._client.chat.completions.create(
                model=settings.openai_chat_model,
                max_tokens=512,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            answer = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        else:
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            answer = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        reasoning_path.append("Generated answer from graph context")
        logger.info("KAG: answered question with %d entities, %d hops", len(all_entity_ids), max_hops)
        return KAGResult(
            answer=answer,
            entities_used=all_entity_ids,
            reasoning_path=reasoning_path,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def seed_demo_graph(self) -> dict:
        """
        Pre-populate the knowledge graph with demo data for classroom demos.
        No API call — adds entities and relations directly.

        The demo data is a tech company org chart that supports multi-hop questions:
          Alice -MANAGES-> Bob  -LEADS-> Platform Team -OWNS-> RAG Service -DEPENDS_ON-> VectorDB
          Alice -MANAGES-> Carol -LEADS-> AI Team       -OWNS-> Agent Service -USES-> RAG Service

        Demo questions to show students:
          Single-hop: "What team does Bob lead?"
          Two-hop:    "What services does Bob's team own?"
          Three-hop:  "What does Alice's backend team's RAG Service depend on?"
          Compare:    "What does Alice manage?" (try same question on CAG for comparison)
        """
        entities = [
            Entity("alice",          "Person",         {"name": "Alice", "role": "CTO"}),
            Entity("bob",            "Person",         {"name": "Bob",   "role": "Backend Lead"}),
            Entity("carol",          "Person",         {"name": "Carol", "role": "ML Lead"}),
            Entity("platform_team",  "Team",           {"name": "Platform Team"}),
            Entity("ai_team",        "Team",           {"name": "AI Team"}),
            Entity("rag_service",    "Service",        {"name": "RAG Service"}),
            Entity("agent_service",  "Service",        {"name": "Agent Service"}),
            Entity("api_gateway",    "Service",        {"name": "API Gateway"}),
            Entity("eval_harness",   "Service",        {"name": "Evaluation Harness"}),
            Entity("vectordb",       "Infrastructure", {"name": "VectorDB",
                                                        "type": "PostgreSQL + pgvector"}),
        ]

        relations = [
            Relation("alice",         "bob",           "MANAGES"),
            Relation("alice",         "carol",         "MANAGES"),
            Relation("bob",           "platform_team", "LEADS"),
            Relation("carol",         "ai_team",       "LEADS"),
            Relation("platform_team", "rag_service",   "OWNS"),
            Relation("platform_team", "api_gateway",   "OWNS"),
            Relation("ai_team",       "agent_service", "OWNS"),
            Relation("ai_team",       "eval_harness",  "OWNS"),
            Relation("rag_service",   "vectordb",      "DEPENDS_ON"),
            Relation("agent_service", "rag_service",   "USES"),
        ]

        for entity in entities:
            self._graph.add_entity(entity)
        for relation in relations:
            self._graph.add_relation(relation)

        logger.info(
            "KAGPipeline: seeded demo graph (%d entities, %d relations)",
            len(entities), len(relations),
        )
        return self.graph_stats()

    def graph_stats(self) -> dict:
        """Return current graph statistics."""
        return {
            "entities": self._graph.entity_count(),
            "relations": self._graph.relation_count(),
        }
