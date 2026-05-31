# RAG Module — `app/rag/`

Retrieval-Augmented Generation grounds LLM responses in your documents. Instead of relying on training-time knowledge, the system embeds your documents, finds the most relevant chunks at query time, and passes them as context to the LLM.

**Module:** 3 | **API prefix:** `/rag`

---

## Standard Pipeline (`pipeline.py`)

Four stages, one class:

```
Text → [chunk] → [embed] → [store]          ← ingestion
Query → [embed] → [similarity search] → [LLM generation]  ← retrieval + generation
```

```python
pipeline = RAGPipeline()
pipeline.ingest_text("Alice manages the Platform team...", source="org.txt")
result = pipeline.query("Who manages the Platform team?")
# result.response, result.sources, result.scores
```

The LLM provider is controlled by `LLM_PROVIDER` in `.env` — swap between Anthropic, OpenAI, Groq, Ollama without changing code.

---

## File Map

| File | Stage | What it does |
|------|-------|-------------|
| `ingestion.py` | 1 | Word-window chunking with configurable overlap |
| `embeddings.py` | 2 | OpenAI or local `sentence-transformers` |
| `retrieval.py` | 3 | Cosine similarity search (`InMemoryVectorStore` or `PgVectorStore`) |
| `pipeline.py` | 4 | End-to-end `RAGPipeline` — ties stages 1–4 together |
| `chunking.py` | 1b | Four pluggable chunking strategies |
| `metadata.py` | 1c | Extract structured metadata from text, Markdown, and PDFs |
| `page_index.py` | adv | Two-tier hierarchical index (page summaries + chunk retrieval) |
| `agentic_rag.py` | adv | Iterative decompose → retrieve → reflect → rewrite loop |
| `corrective_rag.py` | adv | Grade retrieved docs, web search fallback if they're bad |
| `graph_rag.py` | adv | Entity graph from documents, community summaries for synthesis queries |

---

## Chunking Strategies (`chunking.py`)

Chunking is one of the highest-leverage decisions in RAG. Bad chunking cuts sentences in half or mixes unrelated topics.

| Strategy | When to use |
|----------|------------|
| `FixedSize` | Rapid prototyping, structured data (logs, CSV) |
| `Recursive` | General-purpose default — respects paragraphs and lines |
| `Sentence` | Q&A over prose — no sentence ever cut in half |
| `Semantic` | High-precision retrieval — splits at meaning boundaries (costs embedding API calls) |

```python
from app.rag.chunking import get_chunker
chunker = get_chunker("sentence", chunk_size=300, chunk_overlap=30)
chunks  = chunker.chunk(document_text, source="whitepaper.pdf")
```

All four implement the same `ChunkingStrategy` protocol — they're drop-in replacements.

**TODO:** Benchmark all strategies on the same query set and plot retrieval quality vs chunk size.

---

## Document Metadata (`metadata.py`)

Metadata unlocks filtered retrieval: "only search documents from 2024" or "only documents tagged `security`".

```python
extractor = MetadataExtractor()
meta = extractor.extract("reports/q3_2024.pdf")
# meta.title, meta.author, meta.date, meta.tags
```

Supports `.txt` (heuristics), `.md` (YAML front matter), `.pdf` (requires `pip install pypdf`).

**TODO:** Connect metadata to `PgVectorStore` as a JSONB column for SQL-level filtering.

---

## Page Index — Two-Tier Retrieval (`page_index.py`)

Flat chunk lists lose document structure. A 50-page whitepaper chunked into equal windows mixes unrelated sections. `PageIndex` mimics how humans read: scan summaries to find the right section, then read only that section.

```
Tier 1: embed page summaries → find relevant pages
Tier 2: within those pages, find the best chunks
```

Build once (expensive — LLM summarises every page), query many times (cheap).

**TODO:** Add `GET /rag/page-index/toc` and generate example questions per page for even better retrieval precision.

---

## Agentic RAG (`agentic_rag.py`)

Standard RAG fails on multi-part questions and can't recover from poor retrieval. Agentic RAG adds a reasoning loop:

```
Decompose question → sub-questions
For each sub-question:
    Retrieve → Reflect (score 0-1) → Rewrite if score < threshold → Retrieve again
                                 → Accept if score ≥ threshold
Synthesise final answer from all evidence
```

`reasoning_trace` logs every decision — use it to debug why the agent made particular choices.

**Systems lesson:** Always set a hard `max_iterations` ceiling. An unbounded LLM loop will drain your budget silently.

**TODO:** Add a web search tool as a last resort when max iterations is reached.

---

## Corrective RAG — CRAG (`corrective_rag.py`)

Standard RAG's silent failure mode: it uses retrieved documents confidently even when they're irrelevant. CRAG fixes this with explicit grading:

```
Retrieve → Grade each doc (CORRECT / AMBIGUOUS / INCORRECT)
         → CORRECT:   use as-is
         → AMBIGUOUS: supplement with web search
         → INCORRECT: discard, web search only
         → Refine context → Generate → Evaluate
```

`_web_search()` is a stub. The file contains Tavily and DuckDuckGo implementation examples.

**TODO:** Connect `_web_search()` to Tavily (`pip install tavily-python`) to see full CRAG behaviour.

---

## GraphRAG (`graph_rag.py`)

Standard RAG can't answer "What are the main themes across all our docs?" — no 3-chunk retrieval can synthesise the whole corpus. GraphRAG builds an entity graph from your documents and pre-computes community summaries.

```
Build (offline):  extract entities → cluster into communities → LLM-summarise each community
Query (online):   global question → community summaries as context
                  local question  → standard chunk retrieval
```

**Difference from KAG** (`anti_rag/kag.py`): KAG uses a pre-defined structured graph; GraphRAG builds the graph from unstructured documents automatically.

**TODO:** Replace connected-component clustering with Louvain community detection for proper hierarchical structure.

---

## Key Settings (`.env`)

```
EMBEDDING_PROVIDER=local        # "openai" | "cohere" | "local"
VECTOR_STORE=memory             # "memory" | "pgvector"
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RETRIEVAL_TOP_K=3
SIMILARITY_THRESHOLD=0.7
AGENTIC_MAX_ITERATIONS=3
AGENTIC_REFLECTION_THRESHOLD=0.7
PAGE_INDEX_PAGE_SIZE=800
```
