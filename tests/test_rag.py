"""
tests/test_rag.py — Unit tests for the RAG pipeline.

Run with:  pytest tests/ -v

STUDENT TODO:
  - Add tests for edge cases: empty documents, very short queries, non-English text.
  - Add an integration test that calls the actual Claude API (mark with @pytest.mark.integration).
  - Add tests for the API endpoints using FastAPI's TestClient.
"""

import pytest
from app.rag.ingestion import chunk_text, Chunk
from app.rag.retrieval import InMemoryVectorStore, cosine_similarity, SearchResult


# ── Ingestion tests ───────────────────────────────────────────────────────────
class TestChunking:

    def test_basic_chunking(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, source="test.txt", chunk_size=20, chunk_overlap=5)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_source_preserved(self):
        chunks = chunk_text("hello world foo bar baz", source="my_doc.txt", chunk_size=3, chunk_overlap=0)
        assert all(c.source == "my_doc.txt" for c in chunks)

    def test_chunk_indices_sequential(self):
        text = " ".join([f"w{i}" for i in range(50)])
        chunks = chunk_text(text, source="t", chunk_size=10, chunk_overlap=2)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_text("", source="empty.txt")
        assert chunks == []

    def test_single_word_text(self):
        chunks = chunk_text("hello", source="single.txt", chunk_size=10, chunk_overlap=2)
        assert len(chunks) == 1
        assert chunks[0].text == "hello"

    def test_overlap_creates_shared_words(self):
        # With overlap, adjacent chunks should share some words
        words = [f"word{i}" for i in range(30)]
        text = " ".join(words)
        chunks = chunk_text(text, source="t", chunk_size=10, chunk_overlap=5)
        if len(chunks) >= 2:
            set1 = set(chunks[0].text.split())
            set2 = set(chunks[1].text.split())
            assert len(set1 & set2) > 0


# ── Cosine similarity tests ───────────────────────────────────────────────────
class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


# ── Vector store tests ────────────────────────────────────────────────────────
class TestInMemoryVectorStore:

    def _make_store_with_items(self, n: int) -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        for i in range(n):
            chunk = Chunk(text=f"chunk {i}", source="test", chunk_index=i)
            # Simple distinct vectors
            vec = [0.0] * 10
            vec[i % 10] = 1.0
            store.add(chunk, vec)
        return store

    def test_add_and_count(self):
        store = InMemoryVectorStore()
        chunk = Chunk(text="hello", source="test", chunk_index=0)
        store.add(chunk, [1.0, 0.0, 0.0])
        assert store.count() == 1

    def test_search_returns_best_match(self):
        store = InMemoryVectorStore()
        c1 = Chunk(text="cat", source="t", chunk_index=0)
        c2 = Chunk(text="dog", source="t", chunk_index=1)
        store.add(c1, [1.0, 0.0])
        store.add(c2, [0.0, 1.0])

        results = store.search([1.0, 0.0], top_k=1, threshold=0.0)
        assert len(results) == 1
        assert results[0].chunk.text == "cat"

    def test_search_respects_top_k(self):
        store = self._make_store_with_items(5)
        results = store.search([1.0] + [0.0] * 9, top_k=2, threshold=0.0)
        assert len(results) <= 2

    def test_search_respects_threshold(self):
        store = InMemoryVectorStore()
        chunk = Chunk(text="test", source="t", chunk_index=0)
        store.add(chunk, [1.0, 0.0])
        # Query in opposite direction — similarity will be -1.0
        results = store.search([-1.0, 0.0], top_k=5, threshold=0.5)
        assert len(results) == 0

    def test_clear_empties_store(self):
        store = self._make_store_with_items(3)
        store.clear()
        assert store.count() == 0

    def test_search_empty_store(self):
        store = InMemoryVectorStore()
        results = store.search([1.0, 0.0], top_k=3, threshold=0.0)
        assert results == []

    def test_results_sorted_by_score_descending(self):
        store = InMemoryVectorStore()
        for i, weight in enumerate([0.9, 0.3, 0.7]):
            chunk = Chunk(text=f"chunk{i}", source="t", chunk_index=i)
            store.add(chunk, [weight, 1.0 - weight])
        results = store.search([1.0, 0.0], top_k=3, threshold=0.0)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
