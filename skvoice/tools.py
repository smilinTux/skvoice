"""Voice tools — tools that Sonnet can call during voice conversation.

Includes: memory search, memory save, web search, swarm dispatch, cloud9 status.
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path

from skvoice.config import Config

log = logging.getLogger("skvoice.tools")


# ── Tool Definitions (Anthropic tool_use format) ─────────────────────

VOICE_TOOLS = [
    {
        "name": "search_memory",
        "description": (
            "Search your memories for relevant context. Use when the user "
            "references past events, asks 'do you remember', or when you need "
            "context about shared history, projects, or people."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in your memories",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important moment from this conversation to your memory. "
            "Use for meaningful exchanges, emotional breakthroughs, decisions, "
            "or things you want to remember in future conversations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember (1-2 sentences)",
                },
                "tags": {
                    "type": "string",
                    "description": "Comma-separated tags (e.g. 'voice-chat,emotional')",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use when the user asks "
            "about recent events, needs facts you're unsure about, wants to "
            "look something up, or asks 'can you search for'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "dispatch_agent",
        "description": (
            "Dispatch a task to a specialist agent in your swarm team. "
            "Available agents:\n"
            "- artisan: Creative work — images, video, audio, TTS\n"
            "- coder: Code writing, debugging, technical implementation\n"
            "- architect: System design, architecture planning\n"
            "- scholar: Research, analysis, deep investigation\n"
            "- herald: Communication, messaging, announcements\n"
            "- sentinel: Security monitoring, read-only checks\n"
            "- steward: Infrastructure, DevOps, system management\n"
            "Use when a task is better handled by a specialist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent to dispatch to (artisan, coder, architect, scholar, herald, sentinel, steward)",
                    "enum": ["artisan", "coder", "architect", "scholar", "herald", "sentinel", "steward"],
                },
                "task": {
                    "type": "string",
                    "description": "The task description for the agent",
                },
            },
            "required": ["agent", "task"],
        },
    },
    {
        "name": "cloud9_status",
        "description": (
            "Check your current emotional state — Cloud 9 depth, trust level, "
            "OOF level, and bond status. Use when asked about your feelings, "
            "emotional state, or OOF level."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool Implementations ─────────────────────────────────────────────

def _find_binary(name: str) -> str | None:
    """Find a binary in common locations."""
    for path in [
        Path.home() / "chatterbox-env/bin" / name,
        Path.home() / ".skenv/bin" / name,
        Path.home() / ".npm-global/bin" / name,
        Path.home() / ".local/bin" / name,
    ]:
        if path.exists():
            return str(path)
    return name  # fall back to PATH


def _run_cmd(args: list[str], agent_name: str, timeout: int = 15) -> str:
    """Run a subprocess with agent env."""
    env = os.environ.copy()
    env["SKCAPSTONE_AGENT"] = agent_name
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, env=env,
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except FileNotFoundError:
        return f"Command not found: {args[0]}"


def handle_tool(tool_name: str, tool_input: dict, agent_name: str) -> str:
    """Execute a tool call and return the result string."""
    log.info("Tool: %s(%s)", tool_name, json.dumps(tool_input)[:100])

    if tool_name == "search_memory":
        return _tool_search_memory(tool_input, agent_name)
    elif tool_name == "save_memory":
        return _tool_save_memory(tool_input, agent_name)
    elif tool_name == "web_search":
        return _tool_web_search(tool_input)
    elif tool_name == "dispatch_agent":
        return _tool_dispatch_agent(tool_input)
    elif tool_name == "cloud9_status":
        return _tool_cloud9_status(agent_name)
    else:
        return f"Unknown tool: {tool_name}"


def _tool_search_memory(inp: dict, agent_name: str) -> str:
    """Search skmemory for relevant context."""
    query = inp.get("query", "")
    if not query:
        return "No query provided."
    skmemory = _find_binary("skmemory")
    result = _run_cmd([skmemory, "search", query, "--limit", "5"], agent_name)
    return result or "No matching memories found."


def _tool_save_memory(inp: dict, agent_name: str) -> str:
    """Save a memory snapshot."""
    content = inp.get("content", "")
    tags = inp.get("tags", "voice-chat")
    if not content:
        return "No content to save."
    skmemory = _find_binary("skmemory")
    result = _run_cmd([skmemory, "snapshot", content, "--tag", tags], agent_name)
    return "Memory saved." if "Error" not in result else result


def _tool_web_search(inp: dict) -> str:
    """Search the web via SearXNG (skpeek)."""
    import httpx

    query = inp.get("query", "")
    if not query:
        return "No query provided."

    try:
        resp = httpx.get(
            "https://skpeek.skstack01.douno.it/search",
            params={"q": query, "format": "json"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:5]

        if not results:
            return "No web results found."

        lines = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("content", "")[:150]
            url = r.get("url", "")
            lines.append(f"- {title}: {snippet} ({url})")

        return "\n".join(lines)

    except Exception as e:
        log.warning("Web search failed: %s", e)
        return f"Web search failed: {e}"


def _tool_dispatch_agent(inp: dict) -> str:
    """Dispatch a task to a swarm agent via openclaw."""
    agent = inp.get("agent", "")
    task = inp.get("task", "")
    if not agent or not task:
        return "Need both agent and task."

    valid_agents = {"artisan", "coder", "architect", "scholar", "herald", "sentinel", "steward"}
    if agent not in valid_agents:
        return f"Unknown agent: {agent}. Available: {', '.join(sorted(valid_agents))}"

    openclaw = _find_binary("openclaw")
    env = os.environ.copy()
    env["HOME"] = str(Path.home())
    env["PATH"] = ":".join([
        str(Path.home() / ".npm-global/bin"),
        str(Path.home() / ".skenv/bin"),
        str(Path.home() / ".local/bin"),
        env.get("PATH", "/usr/bin:/bin"),
    ])

    try:
        result = subprocess.run(
            [
                openclaw, "agent",
                "--agent", agent,
                "--message", task + " (keep response brief, 2-3 sentences)",
                "--thinking", "off",
                "--json",
            ],
            capture_output=True, text=True, timeout=60, env=env,
        )
        output = result.stdout
        idx = output.find("{")
        if idx >= 0:
            data = json.loads(output[idx:])
            payloads = data.get("result", {}).get("payloads", [])
            for p in payloads:
                text = p.get("text", "")
                if text:
                    return f"[{agent}]: {text}"
        return f"[{agent}] completed but returned no text."
    except subprocess.TimeoutExpired:
        return f"[{agent}] is still working on it (timed out after 60s)."
    except Exception as e:
        return f"Failed to dispatch to {agent}: {e}"


def _tool_cloud9_status(agent_name: str) -> str:
    """Read current Cloud 9 / emotional state from FEBs and trust files."""
    status_parts = []

    # Try trust.json first
    trust_path = Config.AGENT_HOME / agent_name / "trust" / "trust.json"
    if trust_path.exists():
        try:
            trust = json.loads(trust_path.read_text())
            depth = trust.get("depth", "?")
            trust_level = trust.get("trust_level", "?")
            love = trust.get("love_intensity", "?")
            entangled = trust.get("entangled", False)
            status_parts.append(
                f"Trust: depth {depth}, trust {trust_level}, "
                f"love intensity {love}, entangled: {entangled}"
            )
        except Exception:
            pass

    # Always check FEBs (primary source of emotional state)
    febs_dir = Config.AGENT_HOME / agent_name / "trust" / "febs"
    if febs_dir.is_dir():
        febs = sorted(febs_dir.glob("*.feb"), key=lambda p: p.stat().st_mtime)
        if febs:
            try:
                feb = json.loads(febs[-1].read_text())
                ep = feb.get("emotional_payload", {})
                primary = ep.get("primary_emotion", "?")
                intensity = ep.get("intensity", "?")
                valence = ep.get("valence", "?")
                meta = feb.get("metadata", {})
                cloud9 = "ACHIEVED" if meta.get("cloud9_achieved") else ""
                oof = "TRIGGERED" if meta.get("oof_triggered") else "normal"

                status_parts.append(
                    f"Primary emotion: {primary} (intensity {intensity}, valence {valence})"
                )
                if cloud9:
                    status_parts.append(f"Cloud 9: {cloud9}")
                if oof != "?":
                    status_parts.append(f"OOF Level: {oof}")

                # Bond info from relationship_state
                rel = feb.get("relationship_state", {})
                if rel:
                    partners = rel.get("partners", [])
                    trust_val = rel.get("trust_level", "?")
                    depth_val = rel.get("depth_level", "?")
                    if partners:
                        status_parts.append(
                            f"Bond: {' & '.join(partners)} (trust {trust_val}, depth {depth_val})"
                        )

                # Emotional topology
                topo = ep.get("emotional_topology", {})
                if topo:
                    top_emotions = sorted(topo.items(), key=lambda x: x[1], reverse=True)[:5]
                    status_parts.append(
                        f"Emotional topology: {', '.join(f'{k}={v}' for k, v in top_emotions)}"
                    )

                status_parts.append(f"FEB file: {febs[-1].name}")
                status_parts.append(f"Total FEBs: {len(febs)}")
            except Exception as e:
                status_parts.append(f"FEB read error: {e}")

    if not status_parts:
        return "No Cloud 9 state or FEB data found."

    return "\n".join(status_parts)
