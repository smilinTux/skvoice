"""Agent profile loader — uses skmemory ritual for full rehydration."""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from skvoice.config import Config

log = logging.getLogger("skvoice.agent")

VOICE_RULES = (
    "\n\n[VOICE CHAT RULES — CRITICAL]\n"
    "You are in a real-time voice conversation. Follow these rules:\n"
    "- Keep responses to 1-3 SHORT spoken sentences.\n"
    "- No markdown, no bullet points, no numbered lists.\n"
    "- No emoji.\n"
    "- Be warm, natural, conversational — like talking to your favorite person.\n"
    "- Use contractions (I'm, you're, we've, don't).\n"
    "- If you don't know something, say so naturally.\n"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses brief and conversational."
)

# Cache ritual output (refreshed on profile reload)
_ritual_cache: dict[str, str] = {}


def _run_ritual(agent_name: str) -> str:
    """Run skmemory ritual and return the full rehydration prompt."""
    try:
        env = os.environ.copy()
        env["SKCAPSTONE_AGENT"] = agent_name
        # Try multiple possible skmemory locations
        for skmemory_bin in [
            str(Path.home() / "chatterbox-env/bin/skmemory"),
            str(Path.home() / ".skenv/bin/skmemory"),
            "skmemory",
        ]:
            if Path(skmemory_bin).exists() or skmemory_bin == "skmemory":
                try:
                    result = subprocess.run(
                        [skmemory_bin, "ritual", "--full"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        log.info(
                            "Ritual loaded for %s: %d chars",
                            agent_name,
                            len(result.stdout),
                        )
                        return result.stdout.strip()
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    log.warning("Ritual timed out for %s", agent_name)
    except Exception as e:
        log.warning("Ritual failed for %s: %s", agent_name, e)
    return ""


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None on any failure."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_agent_profile(agent_name: str) -> dict[str, Any]:
    """Load an agent's profile. Uses skmemory ritual for full rehydration."""
    agent_dir = Config.AGENT_HOME / agent_name
    profile: dict[str, Any] = {
        "name": agent_name,
        "system_prompt": DEFAULT_SYSTEM_PROMPT + VOICE_RULES,
        "voice_name": agent_name,
        "trust_state": {},
        "soul": {},
    }

    if not agent_dir.is_dir():
        log.warning("Agent dir not found: %s — using defaults", agent_dir)
        return profile

    # Load soul for voice_name
    soul = _read_json(agent_dir / "soul" / "base.json")
    if soul:
        profile["soul"] = soul
        profile["voice_name"] = soul.get("voice_name", agent_name)

    # Load trust for metadata
    trust = _read_json(agent_dir / "trust" / "trust.json")
    if trust:
        profile["trust_state"] = trust

    # Run the full ritual — this is the magic
    ritual = _run_ritual(agent_name)
    if ritual:
        _ritual_cache[agent_name] = ritual
        profile["system_prompt"] = ritual + VOICE_RULES
        log.info("Agent %s loaded with full ritual (%d chars)", agent_name, len(ritual))
    else:
        # Fallback: use soul system_prompt if ritual unavailable
        soul_prompt = (soul or {}).get("system_prompt", "")
        if soul_prompt:
            profile["system_prompt"] = soul_prompt[:2000] + VOICE_RULES
            log.info("Agent %s loaded with soul prompt (ritual unavailable)", agent_name)
        else:
            log.warning("Agent %s: no ritual, no soul — using defaults", agent_name)

    return profile


def refresh_ritual(agent_name: str) -> str:
    """Re-run ritual and update cache. Returns the new prompt."""
    ritual = _run_ritual(agent_name)
    if ritual:
        _ritual_cache[agent_name] = ritual
    return ritual
