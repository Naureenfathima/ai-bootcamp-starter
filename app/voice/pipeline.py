from __future__ import annotations
from typing import Optional
"""
pipeline.py — End-to-end Voice AI pipeline.

Flow:  Audio bytes → STT → LLM → TTS → Audio bytes

Latency budget for a voice assistant (target: < 1.5s total):
  STT (Deepgram nova-2):   ~150-300ms
  LLM (Claude, 2-3 sent):  ~400-800ms
  TTS (ElevenLabs turbo):  ~200-400ms
  Network + overhead:       ~100-200ms

Systems lesson:
  Latency is the primary UX constraint in voice AI. The pipeline stage breakdown
  (stt / llm / tts) is the first tool for identifying your bottleneck.
  Common findings:
    LLM is slow → reduce max_tokens, use a smaller model, or cache common answers
    TTS is slow → switch to streaming TTS, cache fixed phrases
    STT is slow → move to streaming WebSocket instead of prerecorded upload

  Streaming is the path to < 500ms total latency. The current pipeline is
  "batch" — all three stages run sequentially. See STUDENT TODO below.

Architecture:

  ┌─────────────┐   Audio bytes
  │  Client     │ ───────────────────────────────────────▶ POST /voice/process
  └─────────────┘                                                │
                                                                 ▼
                                          ┌──────────────────────────────┐
                                          │        VoicePipeline          │
                                          │                              │
                                          │  [Stage 1] DeepgramSTT       │
                                          │    ↓ transcript              │
                                          │  [Stage 2] Claude LLM        │
                                          │    ↓ llm_text                │
                                          │  [Stage 3] ElevenLabsTTS     │
                                          │    ↓ audio bytes             │
                                          └──────────────────────────────┘
                                                         │
                                                         ▼  mp3 bytes + latency headers
                                          ┌─────────────┐
                                          │  Client      │
                                          └─────────────┘

STUDENT TODO:
  - Implement streaming: STT WebSocket → streaming LLM → streaming TTS → WebSocket out.
    This collapses three sequential wait times into a nearly continuous pipeline.
  - Add interruption detection: if new audio arrives while TTS is playing, cancel TTS.
  - Add context-aware responses: pass conversation history to the RAG pipeline
    so the voice assistant can answer knowledge-base questions accurately.
  - Implement session persistence: store history in PersistentStore (see memory.py).
"""

import logging
import time
from dataclasses import dataclass

import anthropic

from app.config import settings
from app.voice.stt import DeepgramSTT, LowConfidenceError
from app.voice.tts import ElevenLabsTTS

logger = logging.getLogger(__name__)

# Maximum conversation turns to keep in memory (sliding window)
MAX_HISTORY_TURNS = 10


@dataclass
class VoiceResponse:
    """Complete output of one voice pipeline turn."""
    transcript: str           # what the user said
    llm_response: str         # what the assistant said (text)
    audio_bytes: bytes        # what the assistant said (mp3 audio)
    stt_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    total_latency_ms: float
    stt_confidence: float     # Deepgram confidence (0-1)


@dataclass
class LatencyBreakdown:
    """Per-stage latency report for debugging and teaching."""
    stt_ms: float
    llm_ms: float
    tts_ms: float
    total_ms: float
    bottleneck: str            # which stage is slowest

    @classmethod
    def from_response(cls, r: VoiceResponse) -> "LatencyBreakdown":
        stages = {"stt": r.stt_latency_ms, "llm": r.llm_latency_ms, "tts": r.tts_latency_ms}
        bottleneck = max(stages, key=stages.get)
        return cls(
            stt_ms=r.stt_latency_ms,
            llm_ms=r.llm_latency_ms,
            tts_ms=r.tts_latency_ms,
            total_ms=r.total_latency_ms,
            bottleneck=bottleneck,
        )


class VoicePipeline:
    """
    Batch voice pipeline: audio in → transcribe → LLM → synthesise → audio out.

    This is the BATCH version — all three stages run sequentially. Total latency
    is the sum of all three stages (typically 1.5-3s).

    See STUDENT TODO above for how to implement streaming for < 500ms latency.

    Usage:
        pipeline = VoicePipeline()

        with open("question.wav", "rb") as f:
            result = pipeline.process(f.read(), mimetype="audio/wav")

        # Play result.audio_bytes
        # Log result.stt_latency_ms, result.llm_latency_ms, result.tts_latency_ms
    """

    DEFAULT_SYSTEM = (
        "You are a helpful voice assistant. Keep responses concise and natural "
        "for spoken delivery — no markdown, no bullet points, just clear sentences. "
        "Aim for 2-3 sentences unless the question requires more detail."
    )

    def __init__(self, system_prompt:Optional[ str] = None):
        self._stt    = DeepgramSTT()
        self._tts    = ElevenLabsTTS()
        self._claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._system = system_prompt or self.DEFAULT_SYSTEM
        self._history: list[dict] = []
        logger.info("VoicePipeline initialised")

    # ── Main Pipeline ──────────────────────────────────────────────────────────

    def process(self, audio_bytes: bytes, mimetype: str = "audio/wav") -> VoiceResponse:
        """
        Run the full voice pipeline on a single audio clip.

        Handles three cases cleanly:
          - Empty audio (no speech detected)
          - Low-confidence transcript (unclear speech)
          - Normal path: transcribe → LLM → synthesise

        Args:
            audio_bytes: Raw audio from a microphone or uploaded file.
            mimetype:    MIME type of the audio format.

        Returns:
            VoiceResponse with transcript, LLM text, and output audio.
        """
        t_total = time.perf_counter()

        # ── Stage 1: Speech → Text ─────────────────────────────────────────────
        t_stt = time.perf_counter()
        try:
            stt_result = self._stt.transcribe(audio_bytes, mimetype, raise_on_low_confidence=True)
        except LowConfidenceError as exc:
            logger.warning("VoicePipeline: low confidence (%.2f) — requesting repeat", exc.confidence)
            return self._repeat_response(
                stt_latency_ms=round((time.perf_counter() - t_stt) * 1000, 1),
                t_total=t_total,
                confidence=exc.confidence,
            )
        stt_latency = round((time.perf_counter() - t_stt) * 1000, 1)

        if not stt_result.transcript.strip():
            logger.warning("VoicePipeline: empty transcript — no speech detected")
            return self._repeat_response(stt_latency, t_total, confidence=0.0)

        # ── Stage 2: LLM Reasoning ─────────────────────────────────────────────
        self._history.append({"role": "user", "content": stt_result.transcript})

        t_llm = time.perf_counter()
        message = self._claude.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            system=self._system,
            messages=self._history,
        )
        llm_latency = round((time.perf_counter() - t_llm) * 1000, 1)

        llm_text = message.content[0].text
        self._history.append({"role": "assistant", "content": llm_text})
        self._trim_history()

        # ── Stage 3: Text → Speech ─────────────────────────────────────────────
        t_tts = time.perf_counter()
        tts_result = self._tts.synthesise(llm_text)
        tts_latency = round((time.perf_counter() - t_tts) * 1000, 1)

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        logger.info(
            "VoicePipeline | stt=%sms | llm=%sms | tts=%sms | total=%sms | conf=%.2f",
            stt_latency, llm_latency, tts_latency, total_ms, stt_result.confidence,
        )

        return VoiceResponse(
            transcript=stt_result.transcript,
            llm_response=llm_text,
            audio_bytes=tts_result.audio_bytes,
            stt_latency_ms=stt_latency,
            llm_latency_ms=llm_latency,
            tts_latency_ms=tts_latency,
            total_latency_ms=total_ms,
            stt_confidence=stt_result.confidence,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _repeat_response(
        self, stt_latency_ms: float, t_total: float, confidence: float
    ) -> VoiceResponse:
        """Return a polite 'please repeat' audio response."""
        fallback = "I didn't quite catch that — could you please repeat more clearly?"
        tts = self._tts.synthesise(fallback)
        total = round((time.perf_counter() - t_total) * 1000, 1)
        return VoiceResponse(
            transcript="",
            llm_response=fallback,
            audio_bytes=tts.audio_bytes,
            stt_latency_ms=stt_latency_ms,
            llm_latency_ms=0,
            tts_latency_ms=tts.latency_ms,
            total_latency_ms=total,
            stt_confidence=confidence,
        )

    def _trim_history(self) -> None:
        """Keep only the last MAX_HISTORY_TURNS * 2 messages (user + assistant pairs)."""
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
            logger.debug("VoicePipeline: trimmed history to %d messages", len(self._history))

    def reset_conversation(self) -> None:
        """Clear conversation history to start a fresh session."""
        self._history.clear()
        logger.info("VoicePipeline: conversation history cleared")

    def latency_stats(self, response: VoiceResponse) -> dict:
        """Return a latency breakdown with bottleneck identification — useful for teaching."""
        bd = LatencyBreakdown.from_response(response)
        return {
            "stt_ms": bd.stt_ms,
            "llm_ms": bd.llm_ms,
            "tts_ms": bd.tts_ms,
            "total_ms": bd.total_ms,
            "bottleneck": bd.bottleneck,
            "tip": {
                "stt": "Switch to streaming WebSocket STT to eliminate wait.",
                "llm": "Reduce max_tokens, use caching, or stream the LLM output.",
                "tts": "Use synthesise_stream() and stream audio to the client.",
            }.get(bd.bottleneck, ""),
        }
