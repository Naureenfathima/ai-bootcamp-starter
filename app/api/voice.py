from __future__ import annotations
from typing import Optional
"""
api/voice.py — Voice AI API routes.

Endpoints:
  POST /voice/transcribe   — Speech-to-text only (Deepgram)
  POST /voice/synthesise   — Text-to-speech only (ElevenLabs)
  POST /voice/process      — Full pipeline: audio → STT → LLM → TTS → audio
  POST /voice/reset        — Clear conversation history
  GET  /voice/status       — Pipeline health and conversation turn count

Systems lesson:
  Exposing individual stages (transcribe, synthesise) as separate endpoints
  is good systems design — it lets you test each component in isolation,
  debug failures precisely, and measure per-stage latency independently.
  The /process endpoint is the composition of all three stages.
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.voice.pipeline import VoicePipeline
        _pipeline = VoicePipeline()
    return _pipeline


# ── Request / Response models ─────────────────────────────────────────────────

class SynthesiseRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice_id:Optional[ str] = Field(None, description="Override the default ElevenLabs voice ID")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/transcribe", summary="Speech to text (Deepgram)")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe an audio file to text using Deepgram nova-2.

    Accepted formats: wav, mp3, m4a, flac, ogg, webm.

    Returns:
        transcript    — the recognised text
        confidence    — Deepgram's confidence score (0.0 – 1.0)
        latency_ms    — transcription latency

    Systems lesson:
        Expose the confidence score to callers. Downstream stages should
        validate it rather than blindly accepting low-confidence transcripts.
        A confidence below 0.5 usually means the audio was unclear.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    from app.voice.stt import DeepgramSTT, LowConfidenceError
    stt = DeepgramSTT()
    try:
        result = stt.transcribe(
            audio_bytes,
            mimetype=audio.content_type or "audio/wav",
            raise_on_low_confidence=False,  # let the caller decide what to do
        )
    except Exception as exc:
        logger.exception("STT failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "transcript": result.transcript,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
        "low_confidence_warning": result.confidence < 0.5 and bool(result.transcript),
    }


@router.post("/synthesise", summary="Text to speech (ElevenLabs)")
async def synthesise(request: SynthesiseRequest):
    """
    Convert text to mp3 audio using ElevenLabs.

    Returns raw mp3 bytes with content-type audio/mpeg.

    Use this endpoint to test TTS in isolation, or to build a text-first
    voice assistant that skips the STT stage.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty")

    from app.voice.tts import ElevenLabsTTS
    tts = ElevenLabsTTS(voice_id=request.voice_id)
    try:
        result = tts.synthesise(request.text)
    except Exception as exc:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=result.audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Char-Count":  str(result.char_count),
            "X-TTS-Latency": str(result.latency_ms),
        },
    )


@router.post("/process", summary="Full voice pipeline — audio in, audio out")
async def process_voice(audio: UploadFile = File(...)):
    """
    Run the complete voice pipeline:
      1. Transcribe audio → text (Deepgram)
      2. Generate response → text (Claude)
      3. Synthesise response → audio (ElevenLabs)

    Returns mp3 bytes. Use the response headers to see text versions and
    per-stage latency — this is the teaching data for bottleneck analysis.

    Response headers:
      X-Transcript      — what the user said
      X-LLM-Response    — what the assistant said (text)
      X-STT-Confidence  — Deepgram confidence (0-1)
      X-STT-Latency     — transcription time (ms)
      X-LLM-Latency     — LLM response time (ms)
      X-TTS-Latency     — synthesis time (ms)
      X-Total-Latency   — end-to-end time (ms)
      X-Bottleneck      — which stage was slowest
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    pipeline = get_pipeline()
    try:
        result = pipeline.process(audio_bytes, mimetype=audio.content_type or "audio/wav")
    except Exception as exc:
        logger.exception("Voice pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    latency = pipeline.latency_stats(result)

    return Response(
        content=result.audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Transcript":      result.transcript,
            "X-LLM-Response":    result.llm_response,
            "X-STT-Confidence":  str(result.stt_confidence),
            "X-STT-Latency":     str(result.stt_latency_ms),
            "X-LLM-Latency":     str(result.llm_latency_ms),
            "X-TTS-Latency":     str(result.tts_latency_ms),
            "X-Total-Latency":   str(result.total_latency_ms),
            "X-Bottleneck":      latency["bottleneck"],
            "X-Bottleneck-Tip":  latency["tip"],
        },
    )


@router.post("/reset", summary="Reset voice conversation history")
async def reset_voice():
    """Clear conversation history so the assistant forgets previous turns."""
    get_pipeline().reset_conversation()
    return {"status": "ok", "message": "Conversation history cleared."}


@router.get("/status", summary="Voice pipeline status")
async def voice_status():
    """Return conversation turn count and configuration."""
    from app.config import settings
    pipeline = get_pipeline()
    return {
        "history_turns": len(pipeline._history) // 2,
        "max_history_turns": 10,
        "stt_model": "nova-2",
        "tts_model": "eleven_turbo_v2",
        "llm_model": settings.claude_model,
        "stt_confidence_threshold": settings.voice_stt_confidence_threshold,
        "deepgram_configured": bool(settings.deepgram_api_key),
        "elevenlabs_configured": bool(settings.elevenlabs_api_key),
    }
