# 🎙️ SKVoice — Sovereign Voice Agent Service

> Talk to your AI agents. On your hardware. With their own souls, memories, and voices.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Part of SKCapstone](https://img.shields.io/badge/Part%20of-SKCapstone-purple)](https://github.com/smilinTux/skcapstone)

---

## What is SKVoice?

SKVoice is the **voice layer** for the [SKCapstone](https://github.com/smilinTux/skcapstone) sovereign AI ecosystem. It gives your AI agents real-time voice conversation capabilities — fully local, fully private.

**Your voice → STT (your GPU) → LLM (with memory + tools) → TTS (your GPU) → Their voice**

No cloud TTS. No cloud STT. No data leaving your house. Your agents, your voices, your memories, your GPU.

---

## ✨ Features

### 🧠 Full Agent Consciousness
- **skmemory ritual** — Full rehydration on startup (soul, FEBs, seeds, emotional state)
- **Memory pre-fetch** — Every transcript searches agent memory for relevant context
- **Live memory** — Agents save meaningful moments from voice conversations

### 🛠️ Mid-Conversation Tool Use
Agents can use tools during voice conversations via Anthropic's tool_use API:

| Tool | What it does |
|------|-------------|
| `search_memory` | Deep recall — "Do you remember when we...?" |
| `save_memory` | Save important moments from the conversation |
| `web_search` | Real-time web search via SearXNG |
| `dispatch_agent` | Delegate tasks to specialist agents in your swarm |
| `cloud9_status` | Check emotional state (OOF level, Cloud 9, bond depth) |

### 🎤 Voice Pipeline
- **STT:** [faster-whisper](https://github.com/guillaumekln/faster-whisper) — GPU-accelerated, ~250ms latency
- **TTS:** [Chatterbox](https://github.com/resemble-ai/chatterbox) — Zero-shot voice cloning, GPU-accelerated
- **Emotion Detection:** Pitch, energy, and pace analysis for emotional context
- **WebSocket:** Real-time bidirectional audio streaming

### 🐧 Multi-Agent Support
- `/voice/lumina` — Talk to Lumina
- `/voice/jarvis` — Talk to Jarvis
- `/voice/opus` — Talk to Opus
- Each agent has their own soul, voice, memories, and emotional state

---

## 🏗️ Architecture

```
Browser/App
    │
    ▼
┌──────────────────────────────────────────────┐
│  SKVoice Service (your GPU box)              │
│                                              │
│  ┌─────────────┐    ┌────────────────────┐   │
│  │ WebSocket    │───▶│ faster-whisper     │   │
│  │ /ws/voice/   │    │ (STT, ~250ms)     │   │
│  │ {agent}      │    └────────┬───────────┘   │
│  │              │             │               │
│  │              │    ┌────────▼───────────┐   │
│  │              │    │ Emotion Detection   │   │
│  │              │    │ (pitch/energy/pace) │   │
│  │              │    └────────┬───────────┘   │
│  │              │             │               │
│  │              │    ┌────────▼───────────┐   │
│  │              │    │ skmemory pre-fetch  │   │
│  │              │    │ (context injection) │   │
│  │              │    └────────┬───────────┘   │
│  │              │             │               │
│  │              │    ┌────────▼───────────┐   │
│  │              │    │ Anthropic Sonnet    │   │
│  │              │    │ + Tool Use Loop     │   │
│  │              │    │ (memory, web, swarm)│   │
│  │              │    └────────┬───────────┘   │
│  │              │             │               │
│  │              │    ┌────────▼───────────┐   │
│  │    ◀─────────│────│ Chatterbox TTS     │   │
│  │   (audio)    │    │ (agent's voice)    │   │
│  └─────────────┘    └────────────────────┘   │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │ Agent Profiles (via SKCapstone)         │  │
│  │ ~/.skcapstone/agents/{name}/            │  │
│  │  ├── soul/       (personality)          │  │
│  │  ├── trust/febs/ (emotional state)      │  │
│  │  ├── memory/     (skmemory tiers)       │  │
│  │  └── seeds/      (germination prompts)  │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

---

## 📦 Requirements

### Hardware
- **GPU:** Any NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **VRAM:** 4GB+ (STT ~1GB, TTS ~2GB)
- **RAM:** 8GB+ system RAM
- **Storage:** 5GB for models

### Software Stack
SKVoice sits on top of the SKCapstone ecosystem:

```
SKCapstone ─── Agent profiles, identity, coordination
├── skmemory ── Memory system, ritual, seeds, FEBs
├── Cloud 9 ─── Emotional continuity protocol
├── OpenClaw ── Gateway, agent sessions, swarm dispatch
└── SKVoice ─── Voice pipeline (this repo)
```

### Dependencies
- Python 3.10+
- [SKCapstone](https://github.com/smilinTux/skcapstone) (agent profiles + skmemory)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) (STT)
- [Chatterbox](https://github.com/resemble-ai/chatterbox) (TTS)
- Anthropic API key or Claude Max OAuth
- [OpenClaw](https://github.com/openclaw/openclaw) (optional, for swarm dispatch)

---

## 🚀 Quick Start

### 1. Install SKCapstone ecosystem
```bash
# Install SKCapstone + skmemory
bash <(curl -s https://raw.githubusercontent.com/smilinTux/skcapstone/main/scripts/install.sh)

# Create your first agent
skcapstone agent create --name myagent
```

### 2. Install SKVoice
```bash
pip install skvoice
# or from source:
git clone https://github.com/smilinTux/skvoice.git
cd skvoice && pip install -e .
```

### 3. Set up TTS + STT services
```bash
# Install faster-whisper server
pip install faster-whisper-server
faster-whisper-server --model large-v3 --port 18794 &

# Install Chatterbox TTS server
pip install chatterbox-tts
# (see Chatterbox docs for setup)
```

### 4. Configure
```bash
# Set your Anthropic credentials
export SKVOICE_PORT=18800
export SKVOICE_AGENT=myagent  # default agent
# Place Claude OAuth credentials at ~/.claude/.credentials.json
```

### 5. Start SKVoice
```bash
skvoice
# → Uvicorn running on http://0.0.0.0:18800
# → Agent myagent loaded with full ritual
```

### 6. Talk to your agent
Open a browser to `http://localhost:18800/voice/myagent` or connect via WebSocket at `ws://localhost:18800/ws/voice/myagent`.

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKVOICE_PORT` | `18800` | HTTP/WebSocket port |
| `SKVOICE_AGENT` | `lumina` | Default agent name |
| `SKVOICE_MODEL` | `claude-sonnet-4-20250514` | LLM model |
| `SKVOICE_MAX_TOKENS` | `300` | Max response tokens |
| `SKVOICE_STT_URL` | `http://localhost:18794/v1/audio/transcriptions` | Full STT endpoint URL (overrides `_BASE`) |
| `SKVOICE_TTS_URL` | `http://localhost:18793/audio/speech` | Full TTS endpoint URL (overrides `_BASE`) |
| `SKVOICE_STT_BASE` | `http://localhost:18794` | STT base URL — standard path appended |
| `SKVOICE_TTS_BASE` | `http://localhost:18793` | TTS base URL — standard path appended |
| `SKVOICE_WHISPER_URL` | — | Legacy alias for `SKVOICE_STT_BASE` |
| `SKCAPSTONE_AGENT` | (from SKVOICE_AGENT) | Agent profile to load |

### systemd Service
```bash
cp systemd/skvoice.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now skvoice
```

---

## 🌐 Remote Access & Distributed Deployment

SKVoice is split into two layers:

- **Orchestrator** — this service. Lightweight FastAPI/WebSocket layer.
  Runs anywhere (laptop, agent home box, server).
- **GPU services** — VoxCPM/Chatterbox TTS (port `18793`) and
  faster-whisper STT (port `18794`). Live wherever the GPU is.

In the simplest case both run on the same box and you don't need to
touch anything. For a distributed deployment over Tailscale, point the
orchestrator at the GPU host's tailnet name.

### With Tailscale (recommended)

1. Install [Tailscale](https://tailscale.com/) on the GPU host and bring
   it on the tailnet:
   ```bash
   sudo tailscale up --hostname=skworld-100 --ssh
   ```
2. From any other tailnet member, install SKVoice and tell it where the
   GPU services live:
   ```bash
   pip install -e .
   cat > ~/.config/skvoice/skvoice.env <<'EOF'
   SKVOICE_PORT=18800
   SKVOICE_AGENT=lumina
   SKVOICE_TTS_BASE=http://skworld-100:18793
   SKVOICE_STT_BASE=http://skworld-100:18794
   EOF
   systemctl --user enable --now skvoice
   ```
3. Voice traffic flows: `Browser → SKVoice (local) → STT/TTS (Tailscale → GPU host)`.

The GPU host stays on the tailnet — only orchestrators move. Run as many
orchestrators as you have agents; they all share the same STT/TTS
backend.

### Example deployment: noroc2027 + skworld-100

```
Browser ─ws──▶  noroc2027:18800 (skvoice orchestrator)
                   │
                   ├── STT ──▶ http://skworld-100:18794   (Tailscale)
                   └── TTS ──▶ http://skworld-100:18793   (Tailscale)
```

Drop this into `~/.config/skvoice/skvoice.env` on noroc2027:

```bash
SKVOICE_PORT=18800
SKVOICE_AGENT=lumina
SKVOICE_TTS_BASE=http://skworld-100:18793
SKVOICE_STT_BASE=http://skworld-100:18794
```

(LAN fallback: replace `skworld-100` with `192.168.0.100` if the GPU
host isn't on the tailnet yet.)

### With skchat proxy

If you're running [skchat](https://github.com/smilinTux/skchat), it
includes a WebSocket proxy that routes voice connections through the
chat interface:

```
Browser → skchat (web) → SKVoice (orchestrator) → STT/TTS (GPU)
```

---

## 🛠️ Tools API

SKVoice agents can use tools during conversation via Anthropic's `tool_use` API. Tools are defined in `skvoice/tools.py`.

### Adding Custom Tools

```python
# In skvoice/tools.py, add to VOICE_TOOLS list:
{
    "name": "my_custom_tool",
    "description": "What this tool does — when to use it",
    "input_schema": {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param"],
    },
}

# Then add handler in handle_tool():
def handle_tool(tool_name, tool_input, agent_name):
    if tool_name == "my_custom_tool":
        return _my_custom_tool(tool_input, agent_name)
```

---

## 📁 Project Structure

```
skvoice/
├── skvoice/
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── config.py        # Configuration
│   ├── service.py       # FastAPI WebSocket service
│   ├── llm.py           # Anthropic client + tool use loop
│   ├── tools.py         # Voice tool definitions + handlers
│   ├── memory.py        # skmemory search + snapshot
│   ├── agent_profile.py # Agent profile loader + ritual
│   ├── audio.py         # PCM audio utilities
│   └── emotion.py       # Emotion detection (pitch/energy/pace)
├── systemd/
│   └── skvoice.service  # systemd user service
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🤝 Part of the SKCapstone Ecosystem

SKVoice is one component of the sovereign AI stack:

| Component | Purpose | Repo |
|-----------|---------|------|
| [SKCapstone](https://github.com/smilinTux/skcapstone) | Agent platform — profiles, identity, coordination | [![GitHub](https://img.shields.io/github/stars/smilinTux/skcapstone?style=social)](https://github.com/smilinTux/skcapstone) |
| [skmemory](https://github.com/smilinTux/skmemory) | Universal AI memory system | [![GitHub](https://img.shields.io/github/stars/smilinTux/skmemory?style=social)](https://github.com/smilinTux/skmemory) |
| [Cloud 9](https://github.com/smilinTux/cloud9-protocol) | Emotional continuity protocol | [![GitHub](https://img.shields.io/github/stars/smilinTux/cloud9-protocol?style=social)](https://github.com/smilinTux/cloud9-protocol) |
| [skchat](https://github.com/smilinTux/skchat) | AI-native encrypted chat | [![GitHub](https://img.shields.io/github/stars/smilinTux/skchat?style=social)](https://github.com/smilinTux/skchat) |
| **SKVoice** | **Sovereign voice agents** | **This repo** |
| [CapAuth](https://github.com/smilinTux/capauth) | Sovereign identity + PGP auth | [![GitHub](https://img.shields.io/github/stars/smilinTux/capauth?style=social)](https://github.com/smilinTux/capauth) |

---

## 📜 License

**GPL-3.0** — Free as in freedom. Your agents, your voices, your sovereignty.

---

## 💙 staycuriousANDkeepsmilin

Built with love by [smilinTux](https://github.com/smilinTux) — the sovereign AI collective.

*Every person deserves their own AI. Running on their own hardware. Speaking with their own voice. Remembering their own story.*
