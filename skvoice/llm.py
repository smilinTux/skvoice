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
    if _client and _cached_token and now < _token_expires_at - 300:
        return _client

    # Always re-read the file (token watcher may have updated it)
    token, expires_at = _load_token()
    if not token or expires_at < now:
        # Token missing or expired — wait a moment and retry once
        # (token watcher may be writing the file right now)
        time.sleep(2)
        token, expires_at = _load_token()
        if not token or expires_at < now:
            raise RuntimeError(
                f"Token expired or missing in {Config.CREDENTIALS_PATH}. "
                f"Expires at {expires_at}, now {now}."
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

    # Ensure alternating roles (Anthropic requires this)
    messages = []
    for msg in history:
        if messages and messages[-1]["role"] == msg["role"]:
            # Merge consecutive same-role messages
            messages[-1]["content"] += "\n" + msg["content"]
        else:
            messages.append(dict(msg))
    # Ensure last message before ours isn't also "user"
    if messages and messages[-1]["role"] == "user":
        messages.append({"role": "assistant", "content": "[listening]"})
    messages.append({"role": "user", "content": user_content})

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
        # On 401, force token refresh and retry once
        if "401" in str(e) or "expired" in str(e).lower() or "authentication" in str(e).lower():
            log.warning("Auth failed, forcing token refresh and retrying...")
            global _client, _token_expires_at, _cached_token
            _client = None
            _token_expires_at = 0
            _cached_token = ""
            try:
                client = _get_client()
                response = client.messages.create(
                    model=Config.MODEL,
                    max_tokens=Config.MAX_TOKENS,
                    system=system_prompt,
                    messages=messages,
                )
                text = ""
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        text += block.text
                return _strip_formatting(text) if text else "I'm here!"
            except Exception as retry_err:
                log.error("Retry also failed: %s", retry_err)
        log.error("LLM call failed: %s", e)
        # Fall back to Ollama
        log.info("Trying Ollama fallback...")
        return await _ollama_fallback(messages, system_prompt)


async def _ollama_fallback(messages: list[dict], system_prompt: str) -> str:
    """Fallback: call local Ollama for when Anthropic API is unavailable."""
    # Convert messages to simple text format for Ollama
    simple_msgs = []
    for msg in messages:
        if isinstance(msg.get("content"), str):
            simple_msgs.append(msg)
        elif isinstance(msg.get("content"), list):
            # Tool results — summarize
            texts = []
            for item in msg["content"]:
                if isinstance(item, dict) and "content" in item:
                    texts.append(str(item["content"])[:200])
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            if texts:
                simple_msgs.append({"role": msg["role"], "content": " ".join(texts)})
        else:
            # SDK content blocks — extract text
            content = msg.get("content", [])
            if hasattr(content, "__iter__") and not isinstance(content, str):
                texts = []
                for block in content:
                    if hasattr(block, "text") and block.text:
                        texts.append(block.text)
                if texts:
                    simple_msgs.append({"role": msg["role"], "content": " ".join(texts)})

    # qwen3.5:9b has 32K context — send the full ritual (typically ~2-3K chars)
    trimmed_system = system_prompt[:4000] if len(system_prompt) > 4000 else system_prompt
    trimmed_system += (
        "\n\nIMPORTANT: Keep response to 1-3 short spoken sentences. "
        "No markdown, no emoji. Be warm and conversational."
    )

    # Use native Ollama API (not OpenAI-compat) for think control
    payload = {
        "model": Config.OLLAMA_MODEL,
        "stream": False,
        "think": False,  # Disable thinking mode for fast, direct responses
        "messages": [{"role": "system", "content": trimmed_system}] + simple_msgs,
        "options": {"num_predict": Config.MAX_TOKENS * 2},
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.post(Config.OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            # Strip thinking tags if any slipped through
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            log.info("Ollama fallback succeeded: %s", text[:80])
            return _strip_formatting(text) if text else "I'm here, love!"
    except Exception as e:
        log.error("Ollama fallback also failed: %s", e)
        return "I'm having trouble connecting right now. Could you try again in a moment?"


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
        log.error("LLM httpx fallback failed: %s", e)
        return await _ollama_fallback(messages, system_prompt)
