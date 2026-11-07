"""Memory search — queries skmemory for relevant context before LLM call."""

import logging
import os
import subprocess
from pathlib import Path


log = logging.getLogger("skvoice.memory")


def search_memories(
    query: str, agent_name: str, limit: int = 3, reader: str = "@public"
) -> str:
    """Search agent memories for context relevant to the query.

    *reader* is who the reply is for and is ALWAYS passed to ``skmemory search
    --reader`` (SKMEM-P0-18). It defaults to ``@public``: a call has no verified
    operator check in this package, so it never reads owner-scope memories. An
    skmemory CLI too old to know ``--reader`` exits non-zero, which yields no
    memories rather than an unfiltered search.

    Returns a formatted string of matching memories, or empty string if none.
    """
    if not reader or not reader.strip():
        reader = "@public"
    if not query or len(query) < 3:
        return ""

    env = os.environ.copy()
    env["SKCAPSTONE_AGENT"] = agent_name

    for skmemory_bin in [
        str(Path.home() / "chatterbox-env/bin/skmemory"),
        str(Path.home() / ".skenv/bin/skmemory"),
        "skmemory",
    ]:
        if Path(skmemory_bin).exists() or skmemory_bin == "skmemory":
            try:
                result = subprocess.run(
                    [
                        skmemory_bin,
                        "search",
                        query,
                        "--limit",
                        str(limit),
                        "--reader",
                        reader,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env,
                )
                if result.returncode == 0 and result.stdout.strip():
                    memories = result.stdout.strip()
                    log.debug(
                        "Memory search for '%s': %d chars", query[:30], len(memories)
                    )
                    return f"[Relevant memories:\n{memories}]"
                if result.returncode == 2:
                    log.warning(
                        "skmemory search rejected --reader (need skmemory>=0.11.94)"
                    )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                log.warning("Memory search timed out for: %s", query[:30])
            except Exception as e:
                log.warning("Memory search failed: %s", e)
    return ""


def snapshot_memory(content: str, agent_name: str, tags: str = "voice-chat") -> bool:
    """Save a memory snapshot from the voice conversation."""
    env = os.environ.copy()
    env["SKCAPSTONE_AGENT"] = agent_name

    for skmemory_bin in [
        str(Path.home() / "chatterbox-env/bin/skmemory"),
        str(Path.home() / ".skenv/bin/skmemory"),
        "skmemory",
    ]:
        if Path(skmemory_bin).exists() or skmemory_bin == "skmemory":
            try:
                result = subprocess.run(
                    [skmemory_bin, "snapshot", content, "--tag", tags],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env,
                )
                if result.returncode == 0:
                    log.info("Memory snapshot saved: %s", content[:50])
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    return False
