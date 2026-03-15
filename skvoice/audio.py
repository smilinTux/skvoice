"""Audio utilities — PCM/WAV conversion, STT, TTS."""

import io
import logging
import struct
import wave

import httpx

from skvoice.config import Config

log = logging.getLogger("skvoice.audio")


def pcm_to_wav(
    pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1
) -> bytes:
    """Convert raw 16-bit signed PCM to WAV format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def transcribe(wav_bytes: bytes, stt_url: str | None = None) -> str:
    """Send WAV audio to faster-whisper STT and return transcript.

    Args:
        wav_bytes: WAV format audio bytes.
        stt_url: Override STT endpoint URL.

    Returns:
        Transcribed text, or empty string on failure.
    """
    url = stt_url or Config.STT_URL

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            data = {"model": "whisper-1"}
            resp = await client.post(url, files=files, data=data)
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()
    except Exception as e:
        log.error("STT failed: %s", e)
        return ""


async def synthesize(
    text: str, voice: str = "lumina", tts_url: str | None = None
) -> bytes:
    """Send text to Chatterbox TTS and return WAV audio bytes.

    Args:
        text: Text to synthesize.
        voice: Voice name/id.
        tts_url: Override TTS endpoint URL.

    Returns:
        WAV audio bytes, or empty bytes on failure.
    """
    url = tts_url or Config.TTS_URL

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "wav",
            }
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        log.error("TTS failed: %s", e)
        return b""
