"""
corrective_rag.py — Corrective RAG (CRAG).

Standard RAG has a dangerous silent failure mode: it retrieves and uses documents
confidently even when the retrieved context is irrelevant. The model then
generates a plausible-sounding but wrong answer with no indication that
retrieval failed. This is the "retrieval hallucination" problem.

CRAG (Yan et al., 2024) fixes this with an explicit retrieve → grade → correct loop:

  1. Retrieve   — standard vector similarity retrieval
  2. Grade      — LLM grades each document: CORRECT / AMBIGUOUS / INCORRECT
  3. Correct    — choose strategy based on grades:
                   All CORRECT   → use documents as-is
                   Any AMBIGUOUS → supplement with web search
                   All INCORRECT → discard, fall back to web search only
  4. Refine     — strip irrelevant sentences from the context ("knowledge distillation")
  5. Generate   — answer from corrected, refined context
  6. Evaluate   — faithfulness + relevance scores via LLM-as-judge

The result: CRAG is honest about retrieval failures and actively recovers from
them, rather than generating confident hallucinations.

Reference: https://arxiv.org/abs/2401.15884

Systems lesson:
  Defensive system design — assume failures WILL happen and build recovery paths.
  A system that detects when it doesn't know is more trustworthy than one that
  hallucinates confidently. This pattern generalises:
    stale cache → fetch from database
    bad retrieval → web search fallback
    low confidence STT → ask user to repeat

The web_search() method is a stub. STUDENT TODO: replace it with Tavily or
DuckDuckGo to see full CRAG behaviour on live queries.

STUDENT TODO:
  - Connect _web_search() to a real API (Tavily is purpose-built for RAG agents):
      pip install tavily-python
      client = TavilyClient(api_key=settings.tavily_api_key)
      return client.search(query, max_results=3)["results"][0]["content"]
  - Log grading decisions to build a dataset for improving the grader LLM.
  - Add a confidence score to the final answer based on the grade distribution.
  - Grade at the document level (group chunks by source before grading).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import anthropic

from app.config import settings
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval import SearchResult
from app.eval.evaluator import LLMEvaluator

logger = logging.getLogger(__name__)


class RetrievalGrade(str, Enum):
    """Quality grade assigned to a retrieved chunk by the LLM grader."""
    CORRECT   = "correct"    # Clearly relevant and accurate
    AMBIGUOUS = "ambiguous"  # Partially relevant or uncertain
    INCORRECT = "incorrect"  # Clearly off-topic or contradictory


@dataclass
class GradedDocument:
    """A retrieved chunk annotated with its quality grade."""
    chunk_text: str
    source: str
    similarity_score: float    # cosine similarity from retrieval
    grade: RetrievalGrade
    grade_reasoning: str       # why the grader assigned this grade


@dataclass
class CRAGResult:
    """Full result of a CRAG pipeline query."""
    question: str
    answer: str
    graded_documents: list[GradedDocument]
    correction_strategy: str   # "use_asis" | "supplement_web" | "web_only"
    web_search_used: bool
    knowledge_refined: bool    # whether the knowledge strip step changed the context
    eval_score: float          # faithfulness + relevance (0–1), 0 if eval disabled
    eval_passed: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float


class CorrectiveRAGPipeline:
    """
    Corrective RAG: retrieve → grade → correct → refine → generate → evaluate.

    Usage:
        base = RAGPipeline()
        base.ingest_text(document_text, source="knowledge_base.txt")

        crag = CorrectiveRAGPipeline(rag=base)
        result = crag.query("What is the refund policy for enterprise customers?")

        print(result.answer)
        print(f"Strategy: {result.correction_strategy}")
        print(f"Grades: {[(g.grade.value, g.source) for g in result.graded_documents]}")
        print(f"Eval: {result.eval_score:.2f} ({'PASS' if result.eval_passed else 'FAIL'})")
    """

    def __init__(
        self,
        rag: RAGPipeline,
        run_eval: bool = True,
    ):
        self._rag       = rag
        self._client    = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._run_eval  = run_eval
        self._evaluator = LLMEvaluator(metrics=["faithfulness", "relevance"]) if run_eval else None
        self._in_tok    = 0
        self._out_tok   = 0
        logger.info("CorrectiveRAGPipeline initialised (eval=%s)", run_eval)

    # ── Entry Point ────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 3) -> CRAGResult:
        """
        Run the full CRAG pipeline.

          1. Retrieve → 2. Grade → 3. Correct → 4. Refine → 5. Generate → 6. Evaluate
        """
        t0 = time.perf_counter()
        self._in_tok = self._out_tok = 0

        # 1. Retrieve
        results = self._rag.retrieve(question, top_k=top_k)

        # 2. Grade each document
        graded = self._grade_documents(question, results)

        # 3. Correction strategy
        strategy, context, web_used = self._correct(question, graded)

        # 4. Refine context
        refined = False
        if context and strategy != "web_only":
            context, refined = self._refine(question, context)

        # 5. Generate
        answer = self._generate(question, context)

        # 6. Evaluate
        eval_score = 0.0
        eval_passed = False
        if self._run_eval and self._evaluator and context:
            try:
                eval_result = self._evaluator.evaluate(
                    question=question, answer=answer, context=context
                )
                eval_score  = eval_result.overall_score
                eval_passed = eval_result.passed
            except Exception as exc:
                logger.warning("CRAG: evaluation failed: %s", exc)

        latency = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("CRAG: strategy=%s web=%s refined=%s eval=%.2f latency=%sms",
                    strategy, web_used, refined, eval_score, latency)

        return CRAGResult(
            question=question,
            answer=answer,
            graded_documents=graded,
            correction_strategy=strategy,
            web_search_used=web_used,
            knowledge_refined=refined,
            eval_score=eval_score,
            eval_passed=eval_passed,
            input_tokens=self._in_tok,
            output_tokens=self._out_tok,
            latency_ms=latency,
        )

    # ── Step 2: Grade ──────────────────────────────────────────────────────────

    def _grade_documents(
        self, question: str, results: list[SearchResult]
    ) -> list[GradedDocument]:
        """
        Grade each retrieved chunk's relevance to the question.

        One LLM call per chunk. Each grade is independent — the grader does
        not compare chunks against each other.
        """
        graded: list[GradedDocument] = []

        for result in results:
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=128,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Grade this document chunk's relevance to the question.\n\n"
                        f"Question: {question}\n\n"
                        f"Document: {result.chunk.text[:600]}\n\n"
                        f"Return JSON: {{\"grade\": \"correct|ambiguous|incorrect\", "
                        f"\"reasoning\": \"<one sentence>\"}}\n\n"
                        f"correct  = chunk directly answers the question\n"
                        f"ambiguous = chunk is partially relevant or uncertain\n"
                        f"incorrect = chunk is off-topic or contradicts the question"
                    ),
                }],
            )
            self._track(response)
            raw = response.content[0].text.strip()

            try:
                data = json.loads(self._strip_fences(raw))
                grade = RetrievalGrade(data.get("grade", "ambiguous").lower())
                reasoning = data.get("reasoning", "")
            except (json.JSONDecodeError, ValueError):
                grade = RetrievalGrade.AMBIGUOUS
                reasoning = "Parse error — defaulting to ambiguous"

            graded.append(GradedDocument(
                chunk_text=result.chunk.text,
                source=result.chunk.source,
                similarity_score=result.score,
                grade=grade,
                grade_reasoning=reasoning,
            ))
            logger.debug("CRAG grade: %s | %s", grade.value, reasoning[:60])

        counts = {g: sum(1 for d in graded if d.grade == g) for g in RetrievalGrade}
        logger.info("CRAG grades: correct=%d ambiguous=%d incorrect=%d",
                    counts[RetrievalGrade.CORRECT],
                    counts[RetrievalGrade.AMBIGUOUS],
                    counts[RetrievalGrade.INCORRECT])
        return graded

    # ── Step 3: Correct ────────────────────────────────────────────────────────

    def _correct(
        self, question: str, graded: list[GradedDocument]
    ) -> tuple[str, str, bool]:
        """
        Choose a correction strategy and return (strategy, context_text, web_used).

        Decision rules:
          Any CORRECT   → use those chunks as-is (happy path)
          Only AMBIGUOUS → supplement with web search
          All INCORRECT  → discard everything, use web search only
        """
        correct   = [d for d in graded if d.grade == RetrievalGrade.CORRECT]
        ambiguous = [d for d in graded if d.grade == RetrievalGrade.AMBIGUOUS]

        if correct:
            context = "\n---\n".join(d.chunk_text for d in correct)
            logger.info("CRAG: use_asis (%d correct chunks)", len(correct))
            return "use_asis", context, False

        if ambiguous:
            kb_ctx  = "\n---\n".join(d.chunk_text for d in ambiguous)
            web_ctx = self._web_search(question)
            context = f"Knowledge base:\n{kb_ctx}\n\nWeb search:\n{web_ctx}"
            logger.info("CRAG: supplement_web (%d ambiguous chunks)", len(ambiguous))
            return "supplement_web", context, True

        web_ctx = self._web_search(question)
        logger.info("CRAG: web_only (all %d chunks incorrect)", len(graded))
        return "web_only", web_ctx, True

    def _web_search(self, query: str) -> str:
        """
        Fallback web search when retrieved documents are insufficient.

        STUDENT TODO: replace stub with Tavily, DuckDuckGo, or SerpAPI.

        Tavily example (purpose-built for RAG agents):
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)
            result = client.search(query, max_results=3)
            return "\\n".join(r["content"] for r in result["results"])

        DuckDuckGo example (free, no key):
            from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=3))
            return "\\n".join(r["body"] for r in results)
        """
        logger.warning("CRAG: web_search stub called for '%s' — no search API configured", query)
        return (
            f"[Web search stub — implement _web_search() to get live results for: '{query}']\n"
            "Options: Tavily (pip install tavily-python), "
            "DuckDuckGo (pip install duckduckgo-search), SerpAPI."
        )

    # ── Step 4: Refine ─────────────────────────────────────────────────────────

    def _refine(self, question: str, context: str) -> tuple[str, bool]:
        """
        Strip irrelevant sentences from context ("knowledge distillation" from CRAG paper).

        Asks Claude to remove sentences that do not contribute to answering the
        question. Returns (refined_context, was_meaningfully_shorter).
        """
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Remove sentences that don't contribute to answering the question. "
                    f"Keep only sentences that directly help answer it. "
                    f"Return only the refined text, no explanation.\n\n"
                    f"Question: {question}\n\nContext:\n{context[:2000]}"
                ),
            }],
        )
        self._track(response)
        refined = response.content[0].text.strip()
        # Flag as "refined" if we trimmed at least 10%
        was_refined = len(refined) < len(context) * 0.9
        logger.debug("CRAG refine: %d → %d chars", len(context), len(refined))
        return refined, was_refined

    # ── Step 5: Generate ───────────────────────────────────────────────────────

    def _generate(self, question: str, context: str) -> str:
        """Generate a final answer from the corrected, refined context."""
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=(
                "You are a precise assistant. Answer using ONLY the provided context. "
                "If the context is insufficient, say so clearly — do not speculate."
            ),
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }],
        )
        self._track(response)
        return response.content[0].text

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _track(self, response) -> None:
        self._in_tok  += response.usage.input_tokens
        self._out_tok += response.usage.output_tokens

    @staticmethod
    def _strip_fences(text: str) -> str:
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        return text
