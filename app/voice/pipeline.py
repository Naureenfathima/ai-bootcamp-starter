"""
pipeline.py — End-to-end Voice AI pipeline (Stages 1-7).

Flow:  Audio bytes → STT → LLM → TTS → Audio bytes

STUDENT TODO:
  - Add WebSocket support for real-time streaming (the path to <500ms latency).
  - Add conversation history so the assistant remembers previous turns.
  - Add interruption detection: if the user starts speaking, cancel TTS output.
  - Measure and log per-stage latency; identify your bottleneck.
"""

import logging
import time
from dataclasses import dataclass

import anthropic

from app.config import settings
from app.voice.stt import DeepgramSTT
from app.voice.tts import ElevenLabsTTS

logger = logging.getLogger(__name__)


@dataclass
class VoiceResponse:
    transcript: str        # what the user said
    llm_response: str      # what the assistant said (text)
    audio_bytes: bytes     # what the assistant said (audio)
    stt_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    total_latency_ms: float


class VoicePipeline:
    """
    Batch voice pipeline: audio in → transcribe → LLM → synthesise → audio out.

    This is the BATCH version (not streaming). Latency will be ~2-4 seconds.
    See the STUDENT TODO above for how to make it real-time.

    Usage:
        pipeline = VoicePipeline()
        with open("question.wav", "rb") as f:
            result = pipeline.process(f.read(), mimetype="audio/wav")
        # Play result.audio_bytes
    """

    def __init__(self, system_prompt: str | None = None):
        self._stt    = DeepgramSTT()
        self._tts    = ElevenLabsTTS()
        self._claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._system = system_prompt or (
            "You are a helpful voice assistant. Keep your responses concise and natural "
            "for spoken delivery — no markdown, no bullet points, just clear sentences. "
            "Aim for 2-3 sentences unless the question requires more detail."
        )
        self._history: list[dict] = []   # conversation memory

    def process(self, audio_bytes: bytes, mimetype: str = "audio/wav") -> VoiceResponse:
        """
        Run the full voice pipeline on an audio clip.

        Args:
            audio_bytes: Raw audio data from a microphone or file.
            mimetype:    Audio MIME type.

        Returns:
            VoiceResponse with transcript, LLM text, and output audio bytes.
        """
        t_total = time.perf_counter()

        # ── Stage 1: Speech to Text ────────────────────────────────────────────
        stt_result = self._stt.transcribe(audio_bytes, mimetype)
        transcript  = stt_result.transcript

        if not transcript.strip():
            logger.warning("Empty transcript — no speech detected in audio")
            # Return a polite fallback
            fallback_text  = "I didn't catch that. Could you please repeat?"
            fallback_audio = self._tts.synthesise(fallback_text)
            return VoiceResponse(
                transcript="",
                llm_response=fallback_text,
                audio_bytes=fallback_audio.audio_bytes,
                stt_latency_ms=stt_result.latency_ms,
                llm_latency_ms=0,
                tts_latency_ms=fallback_audio.latency_ms,
                total_latency_ms=round((time.perf_counter() - t_total) * 1000, 1),
            )

        # ── Stage 2: LLM Reasoning ─────────────────────────────────────────────
        self._history.append({"role": "user", "content": transcript})

        t_llm = time.perf_counter()
        message = self._claude.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            system=self._system,
            messages=self._history,
        )
        llm_latency_ms = round((time.perf_counter() - t_llm) * 1000, 1)

        llm_text = message.content[0].text
        self._history.append({"role": "assistant", "content": llm_text})

        # Keep conversation history manageable (last 10 turns)
        if len(self._history) > 20:
            self._history = self._history[-20:]

        # ── Stage 3: Text to Speech ────────────────────────────────────────────
        tts_result = self._tts.synthesise(llm_text)

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        logger.info(
            "Voice pipeline | stt=%sms | llm=%sms | tts=%sms | total=%sms",
            stt_result.latency_ms, llm_latency_ms, tts_result.latency_ms, total_ms,
        )

        return VoiceResponse(
            transcript=transcript,
            llm_response=llm_text,
            audio_bytes=tts_result.audio_bytes,
            stt_latency_ms=stt_result.latency_ms,
            llm_latency_ms=llm_latency_ms,
            tts_latency_ms=tts_result.latency_ms,
            total_latency_ms=total_ms,
        )

    def reset_conversation(self) -> None:
        """Clear conversation history to start a fresh session."""
        self._history.clear()
        logger.info("Voice conversation history cleared")
