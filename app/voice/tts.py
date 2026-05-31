from __future__ import annotations
"""
tts.py — Text-to-Speech (Voice AI Stage 3).

Converts text into natural-sounding audio using ElevenLabs.
ElevenLabs' turbo model is optimised for low latency, making it suitable
for near-real-time voice assistant applications.

Systems lesson:
  TTS is typically the slowest stage in the voice pipeline (200-800ms).
  Two patterns reduce perceived latency:
    1. Streaming TTS: start playing audio before the full response is generated.
       Generate in chunks, pipe each chunk to the audio player immediately.
    2. Response caching: cache common short responses (greetings, error messages)
       at startup — common phrases have zero TTS latency on subsequent calls.

Architecture:
  Text (LLM response)
    │
    ├── synthesise()         → batch mode: wait for all audio, return bytes
    └── synthesise_stream()  → streaming mode: yield audio chunks as generated

STUDENT TODO:
  - Add a voice_cache: pre-synthesise fixed phrases at startup.
  - Let users choose a voice from a list (GET /voice/voices endpoint).
  - Add SSML support for control over emphasis, pauses, and speaking rate.
  - Implement streaming TTS → WebSocket: stream audio chunks to the browser
    so playback starts <200ms after the LLM finishes generating.
"""

import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Output of a text-to-speech synthesis call (batch mode)."""
    audio_bytes: bytes    # complete mp3 audio
    latency_ms: float
    char_count: int = 0


class ElevenLabsTTS:
    """
    Synthesises speech using ElevenLabs.

    Usage:
        tts = ElevenLabsTTS()

        # Batch mode (returns complete audio)
        result = tts.synthesise("Hello! How can I help you today?")
        audio_bytes = result.audio_bytes

        # Streaming mode (yields audio chunks as they are generated)
        for chunk in tts.synthesise_stream("This is a long response..."):
            audio_player.write(chunk)

    Model choice:
        eleven_turbo_v2        — fastest, good quality, low latency (~200ms)
        eleven_multilingual_v2 — highest quality, 28 languages, slower (~500ms)
    """

    DEFAULT_MODEL = "eleven_turbo_v2"

    def __init__(self, voice_id:Optional[ str] = None):
        if not settings.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY not set — TTS will fail at runtime.")
        try:
            from elevenlabs.client import ElevenLabs
            self._client   = ElevenLabs(api_key=settings.elevenlabs_api_key)
            self._voice_id = voice_id or settings.elevenlabs_voice_id
        except ImportError:
            raise ImportError(
                "ElevenLabs SDK not installed. Run: pip install elevenlabs"
            )

    def synthesise(self, text: str, model: str = DEFAULT_MODEL) -> TTSResult:
        """
        Convert text to audio bytes (batch mode — waits for the full audio).

        Args:
            text:  The text to speak (keep under ~500 chars for lowest latency).
            model: ElevenLabs model ID.

        Returns:
            TTSResult with mp3 bytes and latency metadata.
        """
        t0 = time.perf_counter()

        generator = self._client.generate(
            text=text,
            voice=self._voice_id,
            model=model,
        )
        audio_bytes = b"".join(generator)
        latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "TTS: %d chars → %d bytes | model=%s | latency=%sms",
            len(text), len(audio_bytes), model, latency_ms,
        )
        return TTSResult(
            audio_bytes=audio_bytes,
            latency_ms=latency_ms,
            char_count=len(text),
        )

    def synthesise_stream(
        self, text: str, model: str = DEFAULT_MODEL
    ) -> Generator[bytes, None, None]:
        """
        Convert text to audio and yield chunks as they are generated.

        Use this in a WebSocket handler to start streaming audio to the client
        before the full synthesis is complete. This can reduce perceived latency
        by 200-400ms for longer responses.

        Args:
            text:  The text to speak.
            model: ElevenLabs model ID.

        Yields:
            bytes: Audio chunks in mp3 format.

        Example:
            async for chunk in tts.synthesise_stream(llm_text):
                await websocket.send_bytes(chunk)

        STUDENT TODO:
            - Wire this to a WebSocket endpoint in api/voice.py
            - Pair it with streaming LLM output so text → audio latency chains
        """
        t0 = time.perf_counter()
        total_bytes = 0

        for chunk in self._client.generate(
            text=text,
            voice=self._voice_id,
            model=model,
            stream=True,
        ):
            if chunk:
                total_bytes += len(chunk)
                yield chunk

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "TTS stream: %d chars → %d bytes | latency=%sms",
            len(text), total_bytes, latency_ms,
        )
