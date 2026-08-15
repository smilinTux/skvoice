"""SKVoice configuration with environment variable overrides.

Networking model
----------------
SKVoice runs as an *orchestrator*. It does not host TTS or STT models — it
calls them over HTTP. The orchestrator can run on any host (laptop, agent
home box, server) and reach the GPU-backed STT/TTS services over LAN or
Tailscale.

Two layers of env vars are recognised, in priority order:

1. ``SKVOICE_TTS_URL`` / ``SKVOICE_STT_URL`` — full endpoint URLs
   (e.g. ``http://skworld-100:18793/audio/speech``). Use these in
   distributed deployments where TTS/STT live on a remote tailnet host.

2. ``SKVOICE_TTS_BASE`` / ``SKVOICE_STT_BASE`` — base URL only
   (e.g. ``http://skworld-100:18793``). The standard endpoint paths are
   appended automatically. Convenient for "just point me at the GPU box"
   deployments. ``SKVOICE_WHISPER_URL`` is accepted as a legacy alias
   for ``SKVOICE_STT_BASE`` (matches the README from v0.1.0).

The orchestrator's own listener is controlled by ``SKVOICE_HOST`` (bind
address, default ``127.0.0.1``) and ``SKVOICE_PORT`` (default ``18800``).
The loopback default is deliberate: no route in ``service.py`` is
authenticated, so widening the bind is an explicit operator decision.

Defaults assume STT/TTS run on the same host as the orchestrator
(``localhost``). For the noroc2027/skworld-100 deployment, set
``SKVOICE_TTS_BASE`` and ``SKVOICE_STT_BASE`` to the tailnet name of the
GPU host (or LAN IP as a fallback) — see ``.env.example``.
"""

import os
from pathlib import Path

_TTS_PATH = "/audio/speech"
_STT_PATH = "/v1/audio/transcriptions"


def _resolve_url(full_var: str, base_var: str, default_path: str,
                 default_base: str, *legacy_base_vars: str) -> str:
    """Resolve a service URL from full-URL env, base-URL env, or default.

    Priority: full URL env > base URL env > legacy base env > built default.
    """
    full = os.getenv(full_var)
    if full:
        return full
    base = os.getenv(base_var)
    if not base:
        for legacy in legacy_base_vars:
            base = os.getenv(legacy)
            if base:
                break
    if not base:
        base = default_base
    return base.rstrip("/") + default_path


class Config:
    # Bind address for the uvicorn listener. Defaults to loopback, per
    # UNIFIED_INGRESS_STANDARD: no route in skvoice/service.py carries auth, so
    # a wildcard bind hands every LAN and tailnet host an unauthenticated
    # WebSocket to any agent. Set SKVOICE_HOST explicitly (a tailnet address, or
    # 0.0.0.0 if you accept that) when an off-box client needs to reach this.
    # Until v0.2.8 this was the string literal "0.0.0.0" in __main__.py with no
    # override at all; see SOP.md section 5, Front-end / Exposure.
    HOST: str = os.getenv("SKVOICE_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("SKVOICE_PORT", "18800"))
    DEFAULT_AGENT: str = os.getenv("SKVOICE_AGENT", "lumina")
    # STT (faster-whisper / OpenAI-compat transcriptions endpoint).
    STT_URL: str = _resolve_url(
        "SKVOICE_STT_URL",
        "SKVOICE_STT_BASE",
        _STT_PATH,
        "http://localhost:18794",
        "SKVOICE_WHISPER_URL",  # legacy alias from v0.1.0 README
    )
    # TTS (VoxCPM / Chatterbox / OpenAI-compat speech endpoint).
    TTS_URL: str = _resolve_url(
        "SKVOICE_TTS_URL",
        "SKVOICE_TTS_BASE",
        _TTS_PATH,
        "http://localhost:18793",
    )
    MODEL: str = os.getenv("SKVOICE_MODEL", "claude-haiku-4-5")
    # Primary LLM: OpenAI-compatible chat endpoint (local proxy — no cloud rate limits).
    LLM_URL: str = os.getenv("SKVOICE_LLM_URL", "http://localhost:18783/v1/chat/completions")
    MAX_TOKENS: int = int(os.getenv("SKVOICE_MAX_TOKENS", "200"))
    CREDENTIALS_PATH: Path = Path(
        os.getenv("SKVOICE_CREDENTIALS_PATH", "~/.claude/.credentials.json")
    ).expanduser()
    AGENT_HOME: Path = Path(
        os.getenv("SKVOICE_AGENT_HOME", "~/.skcapstone/agents")
    ).expanduser()
    # Fallback LLM: OpenAI-compatible endpoint (sovereign qwen3.6 abliterated on .100:8082).
    FALLBACK_URL: str = os.getenv(
        "SKVOICE_FALLBACK_URL", "http://192.168.0.100:8082/v1/chat/completions"
    )
    FALLBACK_MODEL: str = os.getenv("SKVOICE_FALLBACK_MODEL", "qwen3.6-27b-abliterated")
