"""
api/voice.py — Voice AI API routes.

Endpoints:
  POST /voice/transcribe  — Speech-to-text only
  POST /voice/process     — Full pipeline: audio → STT → LLM → TTS → audio
  POST /voice/reset       — Clear conversation history
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter()

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.voice.pipeline import VoicePipeline
        _pipeline = VoicePipeline()
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/transcribe", summary="Speech to text")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe an audio file to text using Deepgram.

    Accepts: wav, mp3, m4a, flac, ogg, webm.
    Returns: { transcript, confidence, latency_ms }
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    from app.voice.stt import DeepgramSTT
    stt = DeepgramSTT()
    try:
        result = stt.transcribe(audio_bytes, mimetype=audio.content_type or "audio/wav")
    except Exception as exc:
        logger.exception("STT failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "transcript": result.transcript,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
    }


@router.post("/process", summary="Full voice pipeline — audio in, audio out")
async def process_voice(audio: UploadFile = File(...)):
    """
    Run the complete voice pipeline:
      1. Transcribe audio with Deepgram
      2. Generate a response with Claude
      3. Synthesise the response with ElevenLabs
      4. Return the audio as an mp3 response

    The response body is raw mp3 bytes.
    Use the X-Transcript and X-LLM-Response headers to see the text versions.
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

    return Response(
        content=result.audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Transcript":    result.transcript,
            "X-LLM-Response":  result.llm_response,
            "X-STT-Latency":   str(result.stt_latency_ms),
            "X-LLM-Latency":   str(result.llm_latency_ms),
            "X-TTS-Latency":   str(result.tts_latency_ms),
            "X-Total-Latency": str(result.total_latency_ms),
        },
    )


@router.post("/reset", summary="Reset voice conversation history")
async def reset_voice():
    """Clear the conversation history so the assistant forgets previous turns."""
    get_pipeline().reset_conversation()
    return {"status": "ok", "message": "Conversation history cleared."}
