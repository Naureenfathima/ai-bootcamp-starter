from __future__ import annotations
from typing import Optional
"""
stt.py — Speech-to-Text (Voice AI Stage 1).

Converts raw audio bytes into a text transcript using the Deepgram API.
Deepgram's nova-2 model is accurate, fast, and supports 30+ languages.

Systems lesson:
  STT is the first failure point in the voice pipeline. Low-confidence transcripts
  propagate errors to every downstream stage. Handle them explicitly at the boundary:
  check confidence, log it, and decide whether to proceed or ask for a repeat.
  "Fail fast and loudly" is better than silently passing garbage downstream.

Architecture:
  Audio bytes
    │
    ▼ DeepgramSTT.transcribe()
  STTResult: transcript + confidence + latency_ms
    │
    ├── confidence < threshold → raise LowConfidenceError
    └── confidence ≥ threshold → pass to LLM stage

STUDENT TODO:
  - Add streaming STT: process audio in real-time using Deepgram's live websocket.
  - Add diarisation: identify multiple speakers (is_diarize=True option).
  - Add language detection: auto-detect language instead of hard-coding "en-US".
  - Support multiple audio input formats without the caller specifying mimetype
    by detecting the format from the first 4 bytes (magic bytes).
"""

import logging
import time
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Output of a speech-to-text transcription."""
    transcript: str
    confidence: float   # 0.0 (certain garbage) → 1.0 (certain correct)
    latency_ms: float
    words:Optional[ list[dict]] = None   # word-level timestamps (if enabled)
    language: str = "en-US"


class LowConfidenceError(ValueError):
    """Raised when Deepgram returns a transcript below the configured threshold."""
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            f"STT confidence {confidence:.2f} is below threshold {threshold:.2f}. "
            "Ask the user to repeat clearly."
        )
        self.confidence = confidence
        self.threshold = threshold


class DeepgramSTT:
    """
    Transcribes audio using the Deepgram nova-2 model.

    Supported audio formats: mp3, mp4, wav, flac, ogg, webm, m4a.

    Usage:
        stt = DeepgramSTT()
        result = stt.transcribe(audio_bytes, mimetype="audio/wav")
        print(result.transcript, result.confidence)

    Confidence threshold:
        If confidence falls below settings.voice_stt_confidence_threshold,
        LowConfidenceError is raised. The VoicePipeline catches this and
        sends a "please repeat" response.
    """

    def __init__(self):
        if not settings.deepgram_api_key:
            logger.warning("DEEPGRAM_API_KEY not set — STT will fail at runtime.")
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            self._client  = DeepgramClient(settings.deepgram_api_key)
            self._Options = PrerecordedOptions
        except ImportError:
            raise ImportError(
                "Deepgram SDK not installed. Run: pip install deepgram-sdk"
            )

    def transcribe(
        self,
        audio_bytes: bytes,
        mimetype: str = "audio/wav",
        raise_on_low_confidence: bool = True,
    ) -> STTResult:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes:              Raw audio data from microphone or file.
            mimetype:                 MIME type (e.g. "audio/wav", "audio/mp3").
            raise_on_low_confidence:  If True, raise LowConfidenceError when
                                      confidence < settings.voice_stt_confidence_threshold.

        Returns:
            STTResult with transcript, confidence score, and latency.

        Raises:
            LowConfidenceError: if confidence is below threshold and raise_on_low_confidence=True.
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

        channel    = response.results.channels[0]
        best       = channel.alternatives[0]
        transcript = best.transcript
        confidence = best.confidence

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "STT: %d bytes → %d chars | confidence=%.2f | latency=%sms",
            len(audio_bytes), len(transcript), confidence, latency_ms,
        )

        threshold = getattr(settings, "voice_stt_confidence_threshold", 0.5)
        if raise_on_low_confidence and confidence < threshold and transcript.strip():
            logger.warning(
                "STT: low confidence %.2f (threshold=%.2f) for transcript: '%s'",
                confidence, threshold, transcript[:60],
            )
            raise LowConfidenceError(confidence, threshold)

        return STTResult(
            transcript=transcript,
            confidence=confidence,
            latency_ms=latency_ms,
        )
