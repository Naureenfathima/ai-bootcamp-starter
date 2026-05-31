"""
agentic_rag.py — Agentic Retrieval-Augmented Generation.

Standard RAG is a fixed pipeline: retrieve once → generate. It cannot
recover from poor retrieval or handle multi-part questions that need
evidence from different parts of the knowledge base.

Agentic RAG adds a reasoning LOOP:
  1. Decompose — break complex questions into focused sub-questions
  2. Retrieve  — get evidence for each sub-question
  3. Reflect   — is this evidence sufficient? (LLM scores quality 0-1)
  4. Rewrite   — if not sufficient, rephrase the query and retry
  5. Synthesise — generate a final answer from all collected evidence

Related research:
  - Adaptive RAG (Jeong et al., 2024) — route queries to different strategies
  - Self-RAG (Asai et al., 2023) — generate retrieval decision tokens
  - FLARE (Jiang et al., 2023) — retrieve when the model is uncertain

Architecture (loop per sub-question):

  Question
    │
    ▼
  [Decompose] ──→ [sub-q1, sub-q2, ...]
                      │
                      ▼ (for each sub-question)
                  [Retrieve] ──→ chunks
                      │
                      ▼
                  [Reflect] ──→ score 0-1
                      │
                    ┌─┴─────────────────────────┐
                    │ score ≥ threshold          │ score < threshold
                    ▼                           ▼
                  [accept]               [Rewrite query]──→ [Retrieve] (repeat)
                                                           max_iterations times
    ┌─────────────────────────────────────────────┘
    ▼
  [Synthesise] ──→ Final answer

Systems lesson:
  The bounded loop (max_iterations) is critical. Never build an unbounded LLM
  loop in production — it can run up costs and fail silently. Always set a hard
  ceiling. The reasoning_trace field gives full observability into every decision
  the agent made, enabling debugging and quality analysis.

STUDENT TODO:
  - Add a tool-use path: if after max_iterations the agent still lacks evidence,
    call a web search tool instead of giving up.
  - Track marginal information gain per iteration: stop early if new evidence
    doesn't add anything above a similarity threshold to existing evidence.
  - Add streaming: yield partial answers as each sub-question is answered.
"""

import json
import logging
import time
from dataclasses import dataclass, field

import anthropic

from app.config import settings
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class SubQuestion:
    """A decomposed sub-question and the evidence collected for it."""
    question: str
    evidence: list[str]    = field(default_factory=list)
    sources: list[str]     = field(default_factory=list)
    retrieval_score: float = 0.0   # self-reflection quality score (0-1)
    iterations_used: int   = 0


@dataclass
class AgenticRAGResult:
    """Full result of an agentic RAG query."""
    question: str
    answer: str
    sub_questions: list[SubQuestion]
    reasoning_trace: list[str]      # step-by-step log of every agent decision
    total_iterations: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float


class AgenticRAGPipeline:
    """
    Agentic RAG with query decomposition, iterative retrieval, and self-reflection.

    Wraps an existing RAGPipeline — AgenticRAG is a reasoning layer on top of
    standard retrieval, not a replacement for it.

    Usage:
        base_rag = RAGPipeline()
        base_rag.ingest_text(document_text, source="whitepaper.pdf")

        agent = AgenticRAGPipeline(rag=base_rag, max_iterations=3)
        result = agent.query("Compare the scalability approach with the security model.")
        print(result.answer)
        print("\\n".join(result.reasoning_trace))  # see every step
    """

    def __init__(
        self,
        rag: RAGPipeline,
        max_iterations: int = 3,
        reflection_threshold: float = 0.7,
        top_k: int = 3,
    ):
        self._rag              = rag
        self._client           = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._max_iterations   = max_iterations
        self._threshold        = reflection_threshold
        self._top_k            = top_k
        logger.info("AgenticRAGPipeline: max_iterations=%d threshold=%.2f",
                    max_iterations, reflection_threshold)

    # ── Entry Point ────────────────────────────────────────────────────────────

    def query(self, question: str) -> AgenticRAGResult:
        """
        Run the full agentic loop.

        Phase 1: Decompose the question into sub-questions.
        Phase 2: For each sub-question, iterate retrieve → reflect → (rewrite if needed).
        Phase 3: Synthesise a final answer from all collected evidence.
        """
        t_total = time.perf_counter()
        trace: list[str] = []
        in_tokens = out_tokens = 0

        trace.append(f"START: question='{question[:80]}'")

        # Phase 1: Decompose
        sub_questions, tok_in, tok_out = self._decompose(question, trace)
        in_tokens += tok_in; out_tokens += tok_out

        # Phase 2: Retrieve + reflect for each sub-question
        total_iters = 0
        for sq in sub_questions:
            i, t_in, t_out = self._collect_evidence(sq, trace)
            in_tokens += t_in; out_tokens += t_out
            total_iters += i

        # Phase 3: Synthesise
        answer, t_in, t_out = self._synthesise(question, sub_questions, trace)
        in_tokens += t_in; out_tokens += t_out

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        trace.append(
            f"DONE: {total_ms}ms | in={in_tokens} out={out_tokens} tokens"
        )

        return AgenticRAGResult(
            question=question,
            answer=answer,
            sub_questions=sub_questions,
            reasoning_trace=trace,
            total_iterations=total_iters,
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            total_latency_ms=total_ms,
        )

    # ── Phase 1: Decompose ─────────────────────────────────────────────────────

    def _decompose(
        self, question: str, trace: list[str]
    ) -> tuple[list[SubQuestion], int, int]:
        """
        Break a complex question into 1-3 independently searchable sub-questions.

        Simple questions ("What is X?") → 1 sub-question (the original).
        Multi-part questions → multiple focused sub-questions.
        """
        prompt = f"""Break the following question into 1 to 3 focused sub-questions
that can each be independently answered by searching a document.

Rules:
- Each sub-question must be self-contained (no pronouns referencing other sub-questions)
- For a simple factual question, return just 1 sub-question identical to the original
- Return JSON only: {{"sub_questions": ["question 1", "question 2"]}}

Question: {question}"""

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(self._strip_fences(raw))
            qs = data.get("sub_questions", [question])
        except (json.JSONDecodeError, KeyError):
            qs = [question]

        sub_questions = [SubQuestion(question=q) for q in qs]
        trace.append(f"DECOMPOSE: {len(sub_questions)} sub-questions: {[sq.question for sq in sub_questions]}")
        return sub_questions, response.usage.input_tokens, response.usage.output_tokens

    # ── Phase 2: Retrieve + Reflect ────────────────────────────────────────────

    def _collect_evidence(
        self, sq: SubQuestion, trace: list[str]
    ) -> tuple[int, int, int]:
        """
        Iterative retrieve → reflect loop for one sub-question.

        Returns (iterations_used, input_tokens, output_tokens).
        """
        current_query = sq.question
        in_tok = out_tok = 0

        for iteration in range(1, self._max_iterations + 1):
            results = self._rag.retrieve(current_query, top_k=self._top_k)
            evidence = [r.chunk.text for r in results]
            sources  = [r.chunk.source for r in results]
            trace.append(f"RETRIEVE [{iteration}] '{current_query[:60]}' → {len(evidence)} chunks")

            if not evidence:
                trace.append(f"REFLECT [{iteration}]: no evidence found")
                sq.iterations_used = iteration
                break

            score, t_in, t_out = self._reflect(sq.question, evidence, trace, iteration)
            in_tok += t_in; out_tok += t_out
            sq.retrieval_score = score

            if score >= self._threshold:
                sq.evidence.extend(evidence)
                sq.sources.extend(sources)
                sq.iterations_used = iteration
                trace.append(f"REFLECT [{iteration}]: score={score:.2f} ≥ {self._threshold} → sufficient ✓")
                break
            else:
                trace.append(f"REFLECT [{iteration}]: score={score:.2f} < {self._threshold} → rewriting")
                if iteration < self._max_iterations:
                    current_query, t_in, t_out = self._rewrite(sq.question, evidence, trace)
                    in_tok += t_in; out_tok += t_out
                else:
                    # Last iteration: accept whatever we have
                    sq.evidence.extend(evidence)
                    sq.sources.extend(sources)
                    sq.iterations_used = iteration
                    trace.append(f"REFLECT [{iteration}]: max iterations reached — accepting available evidence")

        return sq.iterations_used, in_tok, out_tok

    def _reflect(
        self, question: str, evidence: list[str], trace: list[str], iteration: int
    ) -> tuple[float, int, int]:
        """Score how well evidence answers the question (0.0 – 1.0)."""
        context = "\n---\n".join(evidence[:3])
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": (
                    f"Score how well the evidence answers the question.\n\n"
                    f"Question: {question}\n\nEvidence:\n{context}\n\n"
                    f"Return JSON: {{\"score\": <0.0-1.0>, \"reason\": \"<one sentence>\"}}\n"
                    f"1.0 = fully answers the question. 0.0 = completely off-topic."
                ),
            }],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(self._strip_fences(raw))
            score = float(data.get("score", 0.5))
            trace.append(f"REFLECT [{iteration}] score={score:.2f}: {data.get('reason', '')[:80]}")
        except (json.JSONDecodeError, KeyError, ValueError):
            score = 0.5
        return score, response.usage.input_tokens, response.usage.output_tokens

    def _rewrite(
        self, original: str, failed_evidence: list[str], trace: list[str]
    ) -> tuple[str, int, int]:
        """Rephrase the query to find better evidence on the next attempt."""
        sample = failed_evidence[0][:200] if failed_evidence else "no evidence found"
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": (
                    f"The search query returned insufficient evidence. "
                    f"Rewrite the query with different phrasing to find better results.\n\n"
                    f"Original query: {original}\n"
                    f"What was found (not helpful enough): {sample}\n\n"
                    f"Respond with only the rewritten query."
                ),
            }],
        )
        rewritten = response.content[0].text.strip()
        trace.append(f"REWRITE: '{original[:50]}' → '{rewritten[:50]}'")
        return rewritten, response.usage.input_tokens, response.usage.output_tokens

    # ── Phase 3: Synthesise ────────────────────────────────────────────────────

    def _synthesise(
        self, question: str, sub_questions: list[SubQuestion], trace: list[str]
    ) -> tuple[str, int, int]:
        """Combine all collected evidence into a final answer."""
        context_blocks = []
        for i, sq in enumerate(sub_questions, 1):
            if sq.evidence:
                evidence_lines = "\n".join(f"  • {e[:300]}" for e in sq.evidence[:3])
                context_blocks.append(
                    f"Sub-question {i} (quality={sq.retrieval_score:.2f}): {sq.question}\n"
                    f"{evidence_lines}"
                )

        if not context_blocks:
            return (
                "I was unable to find sufficient evidence in the knowledge base to answer this question.",
                0, 0,
            )

        context = "\n\n".join(context_blocks)
        trace.append(f"SYNTHESISE: {len(context_blocks)} evidence blocks")

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=(
                "You are a precise research assistant. Answer the question using "
                "ONLY the provided evidence. Be comprehensive but concise. "
                "Acknowledge gaps explicitly if evidence is incomplete."
            ),
            messages=[{
                "role": "user",
                "content": f"Evidence:\n{context}\n\nOriginal question: {question}",
            }],
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        return text
