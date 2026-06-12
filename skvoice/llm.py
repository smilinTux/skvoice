"""LLM client — OpenAI-compatible primary (local claude-haiku proxy) with a
sovereign qwen3.6-abliterated fallback. The old Anthropic SDK + Claude-Code
OAuth path was retired (cloud rate-limits made it unreliable); both legs now
speak the same /v1/chat/completions API the skvideo path uses.
"""

import logging
import re

import httpx

from skvoice.config import Config
from skvoice.memory import search_memories

log = logging.getLogger("skvoice.llm")


def _strip_formatting(text: str) -> str:
    """Remove markdown and emoji from response text (it gets spoken aloud)."""
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


def _to_plain_messages(history: list[dict]) -> list[dict]:
    """Coerce history into plain {role, content:str} OpenAI messages, merging
    consecutive same-role turns so the sequence stays clean."""
    out: list[dict] = []
    for msg in history:
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Tolerate legacy SDK content blocks / tool-result lists.
            parts = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("content") or item.get("text") or ""))
                    elif hasattr(item, "text"):
                        parts.append(item.text or "")
            content = " ".join(p for p in parts if p)
        if not content:
            continue
        if out and out[-1]["role"] == msg["role"]:
            out[-1]["content"] += "\n" + content
        else:
            out.append({"role": msg["role"], "content": content})
    return out


async def _openai_chat(url: str, model: str, messages: list[dict]) -> str:
    """POST an OpenAI-compatible /v1/chat/completions request and return text."""
    payload = {
        "model": model,
        "max_tokens": Config.MAX_TOKENS,
        "messages": messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        # Strip any leaked <think> reasoning (qwen).
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


async def get_response(
    transcript: str,
    emotion_context: str,
    history: list[dict[str, str]],
    system_prompt: str,
    agent_name: str = "lumina",
) -> str:
    """Get an LLM response with memory pre-fetch.

    Primary: local claude-haiku proxy (Config.LLM_URL / Config.MODEL).
    Fallback: sovereign qwen3.6-abliterated (Config.FALLBACK_URL / FALLBACK_MODEL).
    """
    # Pre-fetch relevant memories (kept from the old flow — this is how she has
    # access to skmemory during a call).
    memory_ctx = search_memories(transcript, agent_name, limit=3)

    user_content = transcript
    if emotion_context:
        user_content = f"{emotion_context}\n{user_content}"
    if memory_ctx:
        user_content = f"{memory_ctx}\n\n{user_content}"

    sys_text = system_prompt + (
        "\n\nIMPORTANT: Keep your reply to 1-3 short spoken sentences. "
        "No markdown, no emoji. Be warm and conversational."
    )
    messages = [{"role": "system", "content": sys_text}]
    messages += _to_plain_messages(history)
    messages.append({"role": "user", "content": user_content})

    # Primary
    try:
        text = await _openai_chat(Config.LLM_URL, Config.MODEL, messages)
        if text:
            return _strip_formatting(text)
        log.warning("Primary LLM returned empty text — falling back")
    except Exception as e:
        log.error("Primary LLM (%s @ %s) failed: %s", Config.MODEL, Config.LLM_URL, e)

    # Fallback
    try:
        log.info("Trying qwen3.6-abliterated fallback...")
        text = await _openai_chat(Config.FALLBACK_URL, Config.FALLBACK_MODEL, messages)
        if text:
            log.info("Fallback succeeded: %s", text[:80])
            return _strip_formatting(text)
    except Exception as e:
        log.error("Fallback LLM (%s @ %s) failed: %s", Config.FALLBACK_MODEL, Config.FALLBACK_URL, e)

    return "I'm having trouble connecting right now. Could you try again in a moment?"
