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
    MODEL: str = os.getenv("SKVOICE_MODEL", "claude-sonnet-4-6")
    MAX_TOKENS: int = int(os.getenv("SKVOICE_MAX_TOKENS", "200"))
    CREDENTIALS_PATH: Path = Path(
        os.getenv("SKVOICE_CREDENTIALS_PATH", "~/.claude/.credentials.json")
    ).expanduser()
    AGENT_HOME: Path = Path(
        os.getenv("SKVOICE_AGENT_HOME", "~/.skcapstone/agents")
    ).expanduser()
    # Ollama fallback for when Anthropic API is unavailable (native API, not OpenAI-compat)
    OLLAMA_URL: str = os.getenv("SKVOICE_OLLAMA_URL", "http://localhost:11434/api/chat")
    OLLAMA_MODEL: str = os.getenv("SKVOICE_OLLAMA_MODEL", "qwen3.5:9b")
