"""Anthropic LLM client with OAuth token and tool use support."""

import json
import logging
import re
import time
from typing import Any

import httpx

from skvoice.config import Config
from skvoice.memory import search_memories
from skvoice.tools import VOICE_TOOLS, handle_tool

log = logging.getLogger("skvoice.llm")

try:
    import anthropic
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    log.info("anthropic SDK not available — using raw httpx for LLM calls")

_client: Any = None
_token_expires_at: float = 0.0
_cached_token: str = ""


def _load_token() -> tuple[str, float]:
    """Read OAuth token and expiry from credentials file."""
    try:
        creds = json.loads(Config.CREDENTIALS_PATH.read_text())
        oauth = creds.get("claudeAiOauth", creds)
        token = oauth.get("accessToken") or oauth.get("access_token", "")
        expires_at = oauth.get("expiresAt") or oauth.get("expires_at", 0)
        if isinstance(expires_at, (int, float)) and expires_at > 1e12:
            expires_at = expires_at / 1000
        elif isinstance(expires_at, str):
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                expires_at = dt.timestamp()
            except Exception:
                expires_at = 0
        return token, float(expires_at)
    except Exception as e:
        log.error("Failed to load credentials: %s", e)
        return "", 0.0


def _get_client() -> Any:
    """Get or refresh the Anthropic client."""
    global _client, _token_expires_at, _cached_token

    now = time.time()
    if _client and _cached_token and now < _token_expires_at - 60:
        return _client

    token, expires_at = _load_token()
    if not token:
        raise RuntimeError(
            f"No valid token in {Config.CREDENTIALS_PATH}. "
            "Run 'claude' to authenticate."
        )

    _cached_token = token
    _token_expires_at = expires_at

    if HAS_SDK:
        _client = anthropic.Anthropic(
            auth_token=token,
            default_headers={
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                "x-app": "cli",
            },
        )
    else:
        _client = token

    log.info("LLM client refreshed, expires in %.1fh", (expires_at - now) / 3600)
    return _client


def _strip_formatting(text: str) -> str:
    """Remove markdown and emoji from response text."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        r"\U0000FE00-\U0000FE0F\U0000200D]+",
        "",
        text,
    )
    return text.strip()


async def get_response(
    transcript: str,
    emotion_context: str,
    history: list[dict[str, str]],
    system_prompt: str,
    agent_name: str = "lumina",
) -> str:
    """Get LLM response with memory search pre-fetch and tool use.

    Flow:
    1. Pre-fetch relevant memories based on transcript
    2. Call Sonnet with tools available
    3. If Sonnet uses a tool, execute it and loop (up to 3 rounds)
    4. Return final text response
    """
    client = _get_client()

    # Pre-fetch relevant memories
    memory_ctx = search_memories(transcript, agent_name, limit=3)

    # Build user message
    user_content = transcript
    if emotion_context:
        user_content = f"{emotion_context}\n{transcript}"
    if memory_ctx:
        user_content = f"{memory_ctx}\n\n{user_content}"

    messages = list(history) + [{"role": "user", "content": user_content}]

    if not HAS_SDK:
        return await _simple_response(messages, system_prompt)

    try:
        for round_num in range(4):
            response = client.messages.create(
                model=Config.MODEL,
                max_tokens=Config.MAX_TOKENS,
                system=system_prompt,
                messages=messages,
                tools=VOICE_TOOLS,
            )

            if response.stop_reason == "tool_use":
                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = handle_tool(block.name, block.input, agent_name)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # Add to message chain and loop
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                # Increase max tokens for tool-use rounds (need room for final answer)
                log.info("Tool round %d complete, continuing...", round_num + 1)
                continue

            # Extract final text
            text = ""
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    text += block.text

            return _strip_formatting(text) if text else "I'm here!"

        return "I got carried away with my tools there. What were you saying?"

    except Exception as e:
        log.error("LLM call failed: %s", e)
        return "I'm sorry, I'm having trouble right now. Could you try again?"


async def _simple_response(messages: list[dict], system_prompt: str) -> str:
    """Fallback: raw httpx without tool use."""
    token = _cached_token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        "x-app": "cli",
    }
    payload = {
        "model": Config.MODEL,
        "max_tokens": Config.MAX_TOKENS,
        "system": system_prompt,
        "messages": messages,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]
            return _strip_formatting(text)
    except Exception as e:
        log.error("LLM fallback failed: %s", e)
        return "I'm having trouble connecting right now."
