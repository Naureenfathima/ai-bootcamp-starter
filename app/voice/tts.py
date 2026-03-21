"""
tts.py — Text-to-Speech (Voice AI Stage 6).

Converts text into natural-sounding audio using ElevenLabs.

STUDENT TODO:
  - Add streaming TTS: start playing audio before the full response is generated.
  - Let users choose from different voices via the API.
  - Cache common short responses (greetings, errors) to reduce latency.
"""

import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """
    Synthesises speech using the ElevenLabs API.

    Usage:
        tts = ElevenLabsTTS()
        audio_bytes = tts.synthesise("Hello! How can I help you today?")
        # Save or stream audio_bytes as needed.
    """

    def __init__(self):
        if not settings.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY is not set. TTS will not work.")
        try:
            from elevenlabs.client import ElevenLabs
            self._client   = ElevenLabs(api_key=settings.elevenlabs_api_key)
            self._voice_id = settings.elevenlabs_voice_id
        except ImportError:
            raise ImportError("Run: pip install elevenlabs")

    def synthesise(self, text: str) -> "TTSResult":
        """
        Convert text to audio bytes.

        Args:
            text: The text to speak.

        Returns:
            TTSResult containing mp3 bytes and latency metadata.
        """
        t0 = time.perf_counter()

        audio_generator = self._client.generate(
            text=text,
            voice=self._voice_id,
            model="eleven_turbo_v2",   # fastest model; swap to eleven_multilingual_v2 for quality
        )

        audio_bytes = b"".join(audio_generator)
        latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

        logger.info("TTS: %d chars → %d bytes (%s ms)",
                    len(text), len(audio_bytes), latency_ms)

        return TTSResult(audio_bytes=audio_bytes, latency_ms=latency_ms)


from dataclasses import dataclass


@dataclass
class TTSResult:
    audio_bytes: bytes
    latency_ms: float
