"""
stt.py — Speech-to-Text (Voice AI Stage 1).

Converts audio bytes into a text transcript using the Deepgram API.

STUDENT TODO:
  - Add streaming STT: process audio in real-time as the user speaks.
  - Add language detection to support multilingual users.
  - Handle low-confidence transcripts gracefully (ask user to repeat).
"""

import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)


class DeepgramSTT:
    """
    Transcribes audio using the Deepgram API.

    Supported formats: mp3, mp4, wav, flac, ogg, webm, m4a.

    Usage:
        stt = DeepgramSTT()
        result = stt.transcribe(audio_bytes, mimetype="audio/wav")
        print(result.transcript)
    """

    def __init__(self):
        if not settings.deepgram_api_key:
            logger.warning("DEEPGRAM_API_KEY is not set. STT will not work.")
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            self._client  = DeepgramClient(settings.deepgram_api_key)
            self._Options = PrerecordedOptions
        except ImportError:
            raise ImportError("Run: pip install deepgram-sdk")

    def transcribe(self, audio_bytes: bytes, mimetype: str = "audio/wav") -> "STTResult":
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data.
            mimetype:    MIME type of the audio (e.g. "audio/wav", "audio/mp3").

        Returns:
            STTResult with transcript and metadata.
        """
        t0 = time.perf_counter()

        options = self._Options(
            model="nova-2",
            smart_format=True,
            punctuate=True,
            language="en-US",
        )

        response = self._client.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes, "mimetype": mimetype},
            options,
        )

        transcript = (
            response.results.channels[0].alternatives[0].transcript
        )
        confidence = (
            response.results.channels[0].alternatives[0].confidence
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("STT: transcribed %d bytes → %d chars (confidence=%.2f, latency=%sms)",
                    len(audio_bytes), len(transcript), confidence, latency_ms)

        return STTResult(
            transcript=transcript,
            confidence=confidence,
            latency_ms=latency_ms,
        )


from dataclasses import dataclass


@dataclass
class STTResult:
    transcript: str
    confidence: float
    latency_ms: float
