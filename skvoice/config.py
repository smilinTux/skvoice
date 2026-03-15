"""SKVoice configuration with environment variable overrides."""

import os
from pathlib import Path


class Config:
    PORT: int = int(os.getenv("SKVOICE_PORT", "18800"))
    DEFAULT_AGENT: str = os.getenv("SKVOICE_AGENT", "lumina")
    STT_URL: str = os.getenv(
        "SKVOICE_STT_URL", "http://localhost:18794/v1/audio/transcriptions"
    )
    TTS_URL: str = os.getenv("SKVOICE_TTS_URL", "http://localhost:18793/audio/speech")
    MODEL: str = os.getenv("SKVOICE_MODEL", "claude-sonnet-4-6")
    MAX_TOKENS: int = int(os.getenv("SKVOICE_MAX_TOKENS", "200"))
    CREDENTIALS_PATH: Path = Path(
        os.getenv("SKVOICE_CREDENTIALS_PATH", "~/.claude/.credentials.json")
    ).expanduser()
    AGENT_HOME: Path = Path(
        os.getenv("SKVOICE_AGENT_HOME", "~/.skcapstone/agents")
    ).expanduser()
