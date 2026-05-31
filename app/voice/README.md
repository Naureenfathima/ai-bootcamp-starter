# Voice Module — `app/voice/`

A three-stage voice pipeline: audio in → text → LLM → audio out. The pipeline chains Deepgram (speech-to-text), Claude (reasoning), and ElevenLabs (text-to-speech) into a single request/response flow.

**Module:** 4  
**API prefix:** `/voice`

---

## The Pipeline

```
Audio bytes
    │
    ▼ Stage 1 — STT
DeepgramSTT.transcribe()          ~150-300ms
    │ transcript + confidence score
    ▼ Stage 2 — LLM
Claude messages.create()          ~400-800ms
    │ response text
    ▼ Stage 3 — TTS
ElevenLabsTTS.synthesise()        ~200-400ms
    │
    ▼
Audio bytes (mp3)
```

Target total latency: **< 1.5 seconds**. The current implementation is batch (sequential). Streaming collapses the three waits into a near-continuous flow — see Student TODOs.

---

## File Map

| File | What it does |
|------|-------------|
| `stt.py` | `DeepgramSTT` — uploads audio, returns transcript + confidence |
| `tts.py` | `ElevenLabsTTS` — converts text to mp3 audio bytes |
| `pipeline.py` | `VoicePipeline` — chains STT → LLM → TTS, manages conversation history |

---

## Setup

```
DEEPGRAM_API_KEY=...          # https://console.deepgram.com/
ELEVENLABS_API_KEY=...        # https://elevenlabs.io/
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM   # Rachel (default)
VOICE_STT_CONFIDENCE_THRESHOLD=0.5          # below this, ask user to repeat
```

Test with a WAV or MP3 file:
```
POST /voice/process      (multipart audio upload)
POST /voice/transcribe   (STT only, no LLM or TTS)
```

---

## Latency Debugging

Every `VoiceResponse` includes per-stage latency. The pipeline also identifies the bottleneck:

```python
stats = pipeline.latency_stats(response)
# {"stt_ms": 250, "llm_ms": 620, "tts_ms": 310, "total_ms": 1180, "bottleneck": "llm"}
```

| Bottleneck | Fix |
|------------|-----|
| `llm` | Reduce `max_tokens`, use a smaller model, cache common answers |
| `tts` | Use `synthesise_stream()` and stream audio to the client |
| `stt` | Switch to streaming WebSocket STT instead of prerecorded upload |

---

## Low-Confidence Handling

If Deepgram returns a confidence score below `VOICE_STT_CONFIDENCE_THRESHOLD`, the pipeline skips the LLM and responds with a polite "please repeat" audio message. This prevents confident wrong answers from garbled speech.

---

## Conversation History

The pipeline keeps the last `MAX_HISTORY_TURNS` (10) conversation pairs as a sliding window so Claude has context across turns within a session.

```
POST /voice/reset   # clear history, start a fresh session
```

---

## Student TODOs

1. **Streaming pipeline** — connect STT WebSocket → streaming Claude → streaming TTS → WebSocket out. This collapses three sequential waits into a near-continuous pipeline and cuts total latency to ~300-500ms.
2. **RAG integration** — pass the transcript to `RAGPipeline.query()` before the LLM step so the voice assistant can answer knowledge-base questions accurately.
3. **Interruption detection** — if new audio arrives while TTS is still playing, cancel the ongoing TTS and process the new audio immediately.
4. **Session persistence** — store `_history` in `PersistentStore` (see `app/agent/memory.py`) so conversation context survives server restarts.
