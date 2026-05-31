# Anti-RAG Module — `app/anti_rag/`

RAG is not always the right tool. This module covers four alternative strategies for grounding LLM responses in data, and a hybrid router that automatically picks the best one per query.

**Module:** 3 (extended) | **API prefix:** `/anti-rag`

---

## When NOT to Use RAG

| Situation | Problem with RAG | Better approach |
|-----------|-----------------|-----------------|
| Multi-hop entity reasoning ("Who manages the team that owns Service X?") | Vector similarity can't follow logic chains | KAG |
| Small, stable document set (< 200K tokens) | Retrieval adds latency with no benefit | CAG |
| Structured data in a database | Vector search is fuzzy; aggregations are always wrong | Text-to-SQL |
| Repeated, domain-specific queries at scale | Retrieval cost and latency on every request | Fine-tuning |

---

## KAG — Knowledge Augmented Generation (`kag.py`)

Replaces vector similarity with graph traversal. You define entities (nodes) and relationships (edges); queries walk the graph to find logically connected facts.

**Best for:** Multi-hop questions
```
"What depends on Alice's team's RAG Service?"
→ Alice -MANAGES→ Bob -LEADS→ Platform Team -OWNS→ RAG Service -DEPENDS_ON→ VectorDB
```

```python
pipeline = KAGPipeline()
pipeline.extract_and_store("Alice manages the Platform team...")
result = pipeline.query("What does Bob's team own?", max_hops=2)
print(result.reasoning_path)  # every graph hop — shows students the "thinking"
```

**KAG vs RAG:**
- KAG wins: precise multi-hop entity chains
- RAG wins: open-domain Q&A over large unstructured corpora
- KAG misses facts not in the graph; RAG hallucinates for facts outside its retrieval

**TODO:** Replace `InMemoryKnowledgeGraph` with Neo4j for production-scale graphs.

---

## CAG — Cache Augmented Generation (`cag.py`)

Puts the entire document corpus into a single Claude context window (up to 200K tokens). No retrieval step — the model reads everything every time. Prompt caching makes repeat calls cheap.

**Best for:** Synthesis queries, small corpora, "compare/list all" questions

```
POST /anti-rag/cag/query  {"question": "List everything owned by each team."}
```

- First call: `cache_creation_input_tokens > 0` (building the cache)
- Same call again: `cache_read_input_tokens > 0` (reading from cache — much cheaper)

**CAG vs KAG:**

| | CAG | KAG |
|---|---|---|
| Misses context | Never | Yes (missing graph edge) |
| Multi-hop reasoning | Inferred by LLM | Explicit graph traversal |
| First call cost | High | Medium |
| Repeat call cost | Low (cache hit) | Medium |
| Best for | Synthesis, small corpora | Entity relationship queries |

---

## Text-to-SQL (`structured_knowledge.py`)

If data lives in a relational database, SQL is more precise than vector search for any query involving filters, aggregations, or joins.

```
POST /anti-rag/sql/query  {"question": "How many active users signed up in 2024?"}
```

**When SQL beats RAG:** aggregations, sorting, exact lookups, schema-bound data.

**TODO:** Connect to a real PostgreSQL or SQLite database.

---

## Fine-Tuning Concepts (`fine_tuning.py`)

Fine-tuning trains a model on your domain data so it answers by weight, not retrieval. Appropriate when:
- You have thousands of high-quality examples
- The task is narrow and well-defined
- Retrieval latency/cost is unacceptable
- The knowledge is stable (doesn't change frequently)

The file includes: a **decision framework** (fine-tune vs RAG), a **training data builder** (generate (instruction, output) pairs with Claude), and **recommended LoRA configs**.

```
GET  /anti-rag/fine-tuning/guide
POST /anti-rag/fine-tuning/generate-data
POST /anti-rag/fine-tuning/lora-config
```

**Fine-tune wins:** classification, extraction with fixed schema, code generation in a specific style, format compliance.
**RAG wins:** frequently changing knowledge, long-tail queries, source citation required.

---

## Hybrid Router (`router.py`)

Classifies each incoming query and routes to the most appropriate strategy.

```
POST /anti-rag/route  {"query": "Summarize everything about each team's services"}
→ {"strategy": "cag", "reasoning": "synthesis query over small corpus"}
```

**TODO:** Replace keyword heuristics with an LLM classifier for higher accuracy.

---

## Classroom Demo Script

### Step 1 — Seed demo data (no API key needed)
```
POST /anti-rag/demo/seed
```
Loads an identical org-chart into both KAG and CAG:
Alice (CTO) → Bob (Backend Lead) → Platform Team → RAG Service → VectorDB
Alice → Carol (ML Lead) → AI Team → Agent Service

### Step 2 — Run the same question through both
```
POST /anti-rag/kag/query  {"question": "What does Bob's team own?", "max_hops": 2}
POST /anti-rag/cag/query  {"question": "What does Bob's team own?"}
```

### Step 3 — Side-by-side comparison
```
POST /anti-rag/compare  {"question": "Who manages the team that owns the RAG Service?"}
```

### What to highlight

- `reasoning_path` in KAG — the explicit graph traversal students can follow
- `cache_creation_input_tokens` → `cache_read_input_tokens` between two CAG calls — the caching benefit
- `context_window_pct` in CAG — what percentage of Claude's 200K window the corpus uses
- Token cost comparison: KAG cheaper on large corpora; CAG amortises cost over repeat queries
