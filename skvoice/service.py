"""SKVoice — Main FastAPI application."""

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from skvoice.agent_profile import load_agent_profile
from skvoice.audio import pcm_to_wav, transcribe, synthesize
from skvoice.config import Config
from skvoice.emotion import analyze_audio, emotion_context_string
from skvoice.llm import get_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("skvoice")

app = FastAPI(title="SKVoice", version="0.1.0")

# Caches
_agent_profiles: dict[str, dict] = {}
_conversation_histories: dict[str, list[dict[str, str]]] = {}


def _get_profile(agent_name: str) -> dict:
    """Load and cache agent profile."""
    if agent_name not in _agent_profiles:
        log.info("Loading agent profile: %s", agent_name)
        _agent_profiles[agent_name] = load_agent_profile(agent_name)
    return _agent_profiles[agent_name]


@app.on_event("startup")
async def startup():
    """Load default agent profile on startup."""
    profile = _get_profile(Config.DEFAULT_AGENT)
    log.info(
        "SKVoice started — default agent: %s, voice: %s",
        profile["name"],
        profile["voice_name"],
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "skvoice",
        "version": "0.1.0",
        "default_agent": Config.DEFAULT_AGENT,
        "port": Config.PORT,
        "stt_url": Config.STT_URL,
        "tts_url": Config.TTS_URL,
        "model": Config.MODEL,
    }


@app.post("/voice/clear")
async def clear_all_histories():
    """Clear all conversation histories."""
    count = len(_conversation_histories)
    _conversation_histories.clear()
    return {"status": "ok", "cleared": count}


@app.get("/voice/agents")
async def list_agents():
    """List available agents by scanning agent home directory."""
    agents = []
    agent_home = Config.AGENT_HOME
    if agent_home.is_dir():
        for d in sorted(agent_home.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                agents.append(d.name)
    return {"agents": agents}


@app.websocket("/ws/voice/{agent_name}")
async def voice_ws(ws: WebSocket, agent_name: str = "lumina"):
    """WebSocket endpoint for voice conversation.

    Protocol:
    - Client sends binary frames with raw 16-bit PCM audio (16kHz mono).
    - Client sends text "END_OF_SPEECH" when done speaking.
    - Client sends text "CLEAR_HISTORY" to reset conversation.
    - Server replies with JSON status messages and binary audio frames.
    """
    await ws.accept()
    conn_id = f"{agent_name}:{id(ws)}"
    log.info("WebSocket connected: %s", conn_id)

    profile = _get_profile(agent_name)
    system_prompt = profile["system_prompt"]
    voice_name = profile["voice_name"]

    # Per-connection history
    if conn_id not in _conversation_histories:
        _conversation_histories[conn_id] = []
    history = _conversation_histories[conn_id]

    pcm_buffer = bytearray()

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Binary frame — accumulate PCM
            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])
                continue

            # Text frame — command
            text = message.get("text", "")

            if text == "CLEAR_HISTORY":
                history.clear()
                await ws.send_json({"type": "status", "state": "history_cleared"})
                log.info("History cleared: %s", conn_id)
                continue

            # Group context — another agent's response (see but don't respond to)
            try:
                parsed = json.loads(text)
                if parsed.get("type") == "group_context":
                    from_agent = parsed.get("from", "unknown")
                    content = parsed.get("text", "")
                    if content:
                        # Add as a user/assistant pair to maintain alternating roles
                        history.append({
                            "role": "user",
                            "content": f"[{from_agent} said in group]: {content}",
                        })
                        history.append({
                            "role": "assistant",
                            "content": f"[Noted what {from_agent} said]",
                        })
                        log.debug("Group context from %s: %s", from_agent, content[:50])
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # Session injection — restore conversation from browser cache
            try:
                parsed = json.loads(text)
                if parsed.get("type") == "inject_session":
                    injected = parsed.get("messages", [])
                    emotion = parsed.get("emotion_state", "")
                    history.clear()
                    for msg in injected:
                        if msg.get("role") in ("user", "assistant") and msg.get("content"):
                            history.append({"role": msg["role"], "content": msg["content"]})
                    # Cap history
                    if len(history) > 40:
                        history[:] = history[-30:]
                    # If emotion state provided, prepend to system prompt context
                    if emotion:
                        log.info("Session injected: %d messages, emotion: %s", len(history), emotion[:50])
                    else:
                        log.info("Session injected: %d messages", len(history))
                    await ws.send_json({
                        "type": "session_restored",
                        "message_count": len(history),
                    })
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            if text == "END_OF_SPEECH":
                if not pcm_buffer:
                    await ws.send_json({"type": "status", "state": "listening"})
                    continue

                pcm_data = bytes(pcm_buffer)
                pcm_buffer.clear()

                try:
                    await _process_speech(
                        ws, pcm_data, history, system_prompt, voice_name, agent_name
                    )
                except Exception as e:
                    log.error("Processing error: %s", e, exc_info=True)
                    await ws.send_json(
                        {"type": "error", "message": f"Processing failed: {e}"}
                    )
                    await ws.send_json({"type": "status", "state": "listening"})
                continue

            # Text chat message — skip STT, go straight to LLM + TTS
            try:
                parsed = json.loads(text)
                if parsed.get("type") == "text_message" and parsed.get("text"):
                    try:
                        await _process_text(
                            ws, parsed["text"], history, system_prompt,
                            voice_name, agent_name
                        )
                    except Exception as e:
                        log.error("Text processing error: %s", e, exc_info=True)
                        await ws.send_json(
                            {"type": "error", "message": f"Processing failed: {e}"}
                        )
                        await ws.send_json({"type": "status", "state": "listening"})
            except (json.JSONDecodeError, TypeError):
                pass

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", conn_id)
    except Exception as e:
        log.error("WebSocket error for %s: %s", conn_id, e, exc_info=True)
    finally:
        # Clean up history on disconnect
        _conversation_histories.pop(conn_id, None)
        log.info("WebSocket closed: %s", conn_id)


async def _process_speech(
    ws: WebSocket,
    pcm_data: bytes,
    history: list[dict[str, str]],
    system_prompt: str,
    voice_name: str,
    agent_name: str = "lumina",
) -> None:
    """Process a complete speech utterance through the full pipeline."""
    # 1. Processing status
    await ws.send_json({"type": "status", "state": "processing"})

    # 2. Convert PCM to WAV
    wav_data = pcm_to_wav(pcm_data)

    # 3. Run emotion analysis on raw PCM
    emotion = analyze_audio(pcm_data)
    emotion_ctx = emotion_context_string(emotion)

    # 4. Transcribe
    transcript = await transcribe(wav_data)

    if not transcript:
        log.debug("Empty transcript, returning to listening")
        await ws.send_json({"type": "status", "state": "listening"})
        return

    # 5. Send user transcript
    await ws.send_json({"type": "transcript", "role": "user", "text": transcript})

    # 6. Thinking
    await ws.send_json({"type": "status", "state": "thinking"})

    # 7. Get LLM response (with memory search + tool use)
    response = await get_response(transcript, emotion_ctx, history, system_prompt, agent_name)

    # 8. Update history
    user_msg = transcript
    if emotion_ctx:
        user_msg = f"{emotion_ctx}\n{transcript}"
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})

    # Keep history bounded
    if len(history) > 40:
        history[:] = history[-30:]

    # 9. Send assistant transcript
    await ws.send_json({"type": "transcript", "role": "assistant", "text": response})

    # 10. Speaking
    await ws.send_json({"type": "status", "state": "speaking"})

    # 11. Synthesize speech
    audio_bytes = await synthesize(response, voice=voice_name)

    if audio_bytes:
        await ws.send_bytes(audio_bytes)
    else:
        log.warning("TTS returned empty audio")

    # 12. Back to listening
    await ws.send_json({"type": "status", "state": "listening"})


async def _process_text(
    ws: WebSocket,
    text: str,
    history: list[dict[str, str]],
    system_prompt: str,
    voice_name: str,
    agent_name: str = "lumina",
) -> None:
    """Process a text message — skip STT, go straight to LLM + TTS."""
    # 1. Send user transcript (text input)
    await ws.send_json({"type": "transcript", "role": "user", "text": text})

    # 2. Thinking
    await ws.send_json({"type": "status", "state": "thinking"})

    # 3. Get LLM response (with memory search + tool use)
    response = await get_response(text, "", history, system_prompt, agent_name)

    # 4. Update history
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})

    if len(history) > 40:
        history[:] = history[-30:]

    # 5. Send assistant transcript
    await ws.send_json({"type": "transcript", "role": "assistant", "text": response})

    # 6. Synthesize speech (agent speaks the response)
    await ws.send_json({"type": "status", "state": "speaking"})
    audio_bytes = await synthesize(response, voice=voice_name)

    if audio_bytes:
        await ws.send_bytes(audio_bytes)

    # 7. Back to listening
    await ws.send_json({"type": "status", "state": "listening"})
