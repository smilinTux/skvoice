"""FaceTime framing: the binary wire format skchat's fallback client expects.

skchat has had a complete FaceTime feature for a while (page, WebRTC, and a
WebSocket fallback) that proxies to ``/ws/facetime/{agent}`` and
``/ws/video/{agent}`` on this service. Those routes were never implemented here,
so both endpoints answered **403** and the feature looked dead against a healthy
server. This module supplies the missing half.

Wire format, read off ``skchat/static/facetime.html``. Every binary frame is a
12-byte little-endian header followed by its payload::

    uint32  frame_type     0x01 = JPEG video, 0x02 = audio
    uint32  timestamp_ms   monotonic-ish, for client-side ordering
    uint32  payload_len    length of the payload that follows
    bytes   payload

Text frames stay plain JSON control messages (``transcript``, ``emotion``,
``status``, ``pong``), which the client routes to the same handler its WebRTC
data channel uses. So JSON is unchanged and only binary needs wrapping.

The wrapper below is deliberately a *shim* around the live WebSocket rather than
a second pipeline: it presents the same ``send_json`` / ``send_bytes`` surface
that :func:`skvoice.service._process_speech` already calls, so the whole
STT -> LLM -> TTS turn is reused verbatim and cannot drift from the voice path.
"""

from __future__ import annotations

import struct
import time
from typing import Any

FRAME_VIDEO = 0x01
FRAME_AUDIO = 0x02

#: Canvas size the client allocates. Frames are drawn scaled to fit, so this is
#: a hint rather than a hard requirement, but matching it avoids a resample.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

_EPOCH = time.monotonic()


def _timestamp_ms() -> int:
    """Milliseconds since process start, wrapped into the header's uint32."""
    return int((time.monotonic() - _EPOCH) * 1000) & 0xFFFFFFFF


def frame(frame_type: int, payload: bytes) -> bytes:
    """Wrap *payload* in the 12-byte FaceTime header."""
    return struct.pack("<III", frame_type, _timestamp_ms(), len(payload)) + payload


def audio_frame(wav_bytes: bytes) -> bytes:
    """Frame TTS audio for the client's ``playAudioChunk``."""
    return frame(FRAME_AUDIO, wav_bytes)


def video_frame(jpeg_bytes: bytes) -> bytes:
    """Frame a JPEG still for the client's canvas."""
    return frame(FRAME_VIDEO, jpeg_bytes)


class FaceTimeSocket:
    """Adapts a raw WebSocket to the FaceTime binary protocol.

    Presents the exact surface :func:`skvoice.service._process_speech` uses
    (``send_json``, ``send_bytes``, ``receive``), so the existing voice turn runs
    unmodified and any future change to it applies to FaceTime for free. Only
    ``send_bytes`` differs: audio is framed instead of sent raw.
    """

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send_json(self, data: dict) -> None:
        # Control messages are already JSON text on both paths; no wrapping.
        await self._ws.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        # Everything _process_speech sends as bytes is TTS audio.
        await self._ws.send_bytes(audio_frame(data))

    async def send_video(self, jpeg_bytes: bytes) -> None:
        await self._ws.send_bytes(video_frame(jpeg_bytes))

    async def receive(self):
        return await self._ws.receive()

    def __getattr__(self, name: str):
        # Anything else (accept, close, client_state, ...) passes through.
        return getattr(self._ws, name)


def placeholder_jpeg(
    agent_name: str,
    state: str = "listening",
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> bytes:
    """A minimal JPEG so the client canvas shows *something* while we talk.

    This is honestly a placeholder, not an avatar. Rendering a real talking head
    (MuseTalk or similar) is a separate piece of work with its own GPU budget;
    what matters here is that the transport, framing and client rendering path
    are real and exercised. Swap this for a renderer and nothing else changes.

    Returns empty bytes if Pillow is unavailable, and the caller simply sends no
    video frame rather than failing the call.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover - Pillow is optional
        return b""

    bg = {
        "listening": (18, 22, 33),
        "processing": (26, 30, 48),
        "speaking": (30, 26, 52),
    }.get(state, (18, 22, 33))

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # A simple pulse ring keyed to state, enough to show the stream is live.
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 5
    ring = {
        "listening": (70, 90, 130),
        "processing": (120, 110, 190),
        "speaking": (150, 120, 220),
    }.get(state, (70, 90, 130))
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=ring,
        width=max(2, radius // 20),
    )
    draw.text((cx - 40, cy - 8), agent_name, fill=(220, 225, 235))
    draw.text((20, height - 30), f"skvoice · {state}", fill=(120, 130, 150))

    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()
