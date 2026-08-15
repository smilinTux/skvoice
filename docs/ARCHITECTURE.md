# skvoice Architecture

> ⚠️ **Accuracy note, updated 2026-08-15.** This document was written at v0.2.5.
> v0.2.6 retired the Anthropic SDK and the Claude OAuth path in favour of two
> plain OpenAI-compatible `/v1/chat/completions` endpoints, and this file was
> not updated. Two sections described a design that no longer ran: **"LLM turn
> & the tool loop"** and **"Voice tools (`tools.py`)"**.
>
> **`skvoice/tools.py` has since been deleted** as unreachable code, so those
> two sections have been cut down to a removal note rather than rewritten;
> there is no tool loop to describe. Everything else here (the pipeline, the
> WebSocket protocol and state machine, the agent/ritual model, the source map,
> networking, and the integration adapter) was re-verified against the code and
> is accurate. For the current LLM leg and the exposure posture, read
> [`../SOP.md`](../SOP.md).

skvoice is an **orchestrator**, not a model host. It owns no STT, TTS, or LLM
weights, and keeps no durable state — conversation histories live in memory and
are dropped on disconnect. Everything it does is glue: take audio in over a
WebSocket, call the local STT engine, enrich the transcript (emotion + memory),
run a tool-using LLM turn, call the local TTS engine, stream audio back. The
agent it speaks as — its soul, voice, memories, and emotional state — comes
entirely from the SKCapstone agent profile it loads at runtime.

This document covers the pipeline, the WebSocket protocol and connection state
machine, the agent/memory model, the LLM leg, the integration adapter,
and where skvoice sits in SKStack v2.

## The shape of it

```mermaid
flowchart LR
    CLIENT["client<br/>(browser / app)"]
    subgraph SVC["skvoice service (FastAPI + uvicorn :18800)"]
      WS["/ws/voice/{agent}<br/>(duplex WebSocket)"]
      PIPE["pipeline<br/>(_process_speech / _process_text)"]
      PROF["profile cache<br/>(per-agent ritual)"]
      HIST["history cache<br/>(per-connection, bounded)"]
    end
    subgraph EXT["external services (yours)"]
      STT["faster-whisper STT"]
      TTS["Chatterbox / VoxCPM TTS"]
      CLAUDE["Anthropic Claude"]
      OLLAMA["Ollama (fallback)"]
      SKMEM["skmemory CLI"]
      SKCAP["skmemory ritual<br/>(SKCapstone profile)"]
    end

    CLIENT <-->|"PCM / JSON / audio"| WS
    WS --> PIPE
    PIPE --> PROF
    PIPE --> HIST
    PROF -->|"ritual --full"| SKCAP
    PIPE -->|"WAV"| STT
    PIPE -->|"search"| SKMEM
    PIPE -->|"messages.create + tools"| CLAUDE
    CLAUDE -.->|"on failure"| OLLAMA
    PIPE -->|"speech"| TTS
```

## Request lifecycle — one spoken turn

The core path is `voice_ws` → `_process_speech` in `service.py`. A complete
utterance flows through twelve steps; the client signals end-of-utterance with a
text `END_OF_SPEECH` frame after streaming binary PCM.

```mermaid
sequenceDiagram
    participant C as client
    participant WS as voice_ws (service.py)
    participant A as audio.py
    participant E as emotion.py
    participant M as memory.py
    participant L as llm.py
    participant K as LLM (primary, then fallback)

    C->>WS: binary PCM frames (16kHz mono)
    C->>WS: "END_OF_SPEECH"
    WS-->>C: {status: processing}
    WS->>A: pcm_to_wav(pcm)
    WS->>E: analyze_audio(pcm) → emotion cue
    WS->>A: transcribe(wav) → STT POST
    A-->>WS: transcript text
    WS-->>C: {transcript, role: user}
    WS-->>C: {status: thinking}
    WS->>L: get_response(transcript, emotion, history, system, agent)
    L->>M: search_memories(transcript) → skmemory search
    M-->>L: [relevant memories]
    L->>K: POST Config.LLM_URL /v1/chat/completions
    alt primary errors or returns empty text
        L->>K: POST Config.FALLBACK_URL /v1/chat/completions
    end
    K-->>L: final text
    L-->>WS: stripped reply (no markdown/emoji)
    WS-->>C: {transcript, role: assistant}
    WS-->>C: {status: speaking}
    WS->>A: synthesize(reply, voice_name) → TTS POST
    A-->>WS: WAV bytes
    WS-->>C: binary audio
    WS-->>C: {status: listening}
```

Key details, grounded in the source:

- **Emotion cue** (`emotion.py`) is computed on the raw PCM *before* STT: RMS
  energy, zero-crossing rate, and an autocorrelation pitch estimate (70–400 Hz)
  map to tags (`energetic` / `calm`, `rapid_speech` / `slow_speech`,
  `high_pitch` / `low_pitch`) that become a one-line prefix like
  `[Speaker sounds energetic, speaking quickly]`. If no tags fire, nothing is
  added.
- **Memory pre-fetch** (`memory.py`) runs `skmemory search <transcript>
  --limit 3` with `SKCAPSTONE_AGENT=<agent>` set, and prepends any hits to the
  user message as `[Relevant memories: …]`. It locates the `skmemory` binary
  across `~/chatterbox-env/bin`, `~/.skenv/bin`, then `PATH`.
- **History** is per-connection (keyed `agent:id(ws)`), bounded to the last 30
  entries once it exceeds 40, and recorded with the emotion cue and any peer
  context the agent actually saw — so the transcript reflects the agent's real
  view of the turn.
- **Text path** (`_process_text`) skips STT and emotion entirely: a
  `text_message` JSON frame goes straight to the LLM + TTS. The user's text is
  *not* echoed back as a transcript (the client already rendered it).

## WebSocket protocol & connection state machine

`/ws/voice/{agent}` is a single duplex channel. The client sends binary PCM and
JSON/text command frames; the server replies with JSON status/transcript
messages and binary audio. The status messages drive a small client-side state
machine.

```mermaid
stateDiagram-v2
    [*] --> listening: WS accept
    listening --> processing: END_OF_SPEECH (buffer non-empty)
    processing --> thinking: STT returned text
    processing --> listening: empty transcript
    thinking --> speaking: LLM reply ready
    speaking --> listening: audio sent
    listening --> listening: CLEAR_HISTORY / group_init / inject_session
    thinking --> listening: error (alert + {type: error})
    listening --> [*]: disconnect (history + group ctx cleaned up)
```

Frames the server understands (text or JSON):

| Frame | Effect |
|---|---|
| binary | Appended to the PCM buffer for the current utterance |
| `END_OF_SPEECH` | Run the full speech pipeline on the buffered PCM |
| `CLEAR_HISTORY` | Reset this connection's conversation history |
| `{type: text_message, text}` | Text turn — skip STT, run LLM + TTS |
| `{type: inject_session, messages, emotion_state}` | Restore a cached conversation (capped to 40, trimmed to 30) |
| `{type: group_init, peers}` | Append a one-shot multi-agent suffix to the system prompt for this connection |
| `{type: group_context, from, text}` | Buffer a peer agent's line to prepend to this agent's next turn |

Server → client message types: `status` (`processing`/`thinking`/`speaking`/
`listening`/`history_cleared`), `transcript` (`role` + `text`), `group_ready`,
`session_restored`, `error`, plus raw binary audio.

### Multi-agent group chat

`group_init` tells an agent which peers share the room; `_build_group_system_suffix`
adds an instruction block so the agent knows it isn't alone and how to read peer
lines. Peer turns arrive as `group_context` frames, are buffered per connection,
and `_drain_group_context` flattens them into a `[from <agent>]: …` block
prepended to the next user turn — so each agent reacts to the wider conversation
without polluting its own history with assistant turns it never produced.

## Agent profile & ritual

An agent profile is loaded lazily on first use and cached (`_get_profile`). The
load (`agent_profile.load_agent_profile`) does the following:

```mermaid
flowchart TD
    START["load_agent_profile(agent)"] --> DIR{"~/.skcapstone/agents/<agent>/<br/>exists?"}
    DIR -->|no| DEF["defaults<br/>(generic assistant + VOICE_RULES)"]
    DIR -->|yes| SOUL["read soul/base.json<br/>→ voice_name"]
    SOUL --> TRUST["read trust/trust.json<br/>→ trust_state"]
    TRUST --> RIT["run: skmemory ritual --full<br/>(SKCAPSTONE_AGENT=<agent>)"]
    RIT -->|ok| SYS["system_prompt = ritual + VOICE_RULES"]
    RIT -->|fail| FALL{"soul.system_prompt?"}
    FALL -->|yes| SP["system_prompt = soul prompt[:2000] + VOICE_RULES"]
    FALL -->|no| DEFP["system_prompt = defaults + VOICE_RULES"]
```

The **ritual** is the magic: `skmemory ritual --full` rehydrates the agent's
full self — soul blueprint, FEB emotional state, seeds, germination prompts,
strongest memories — and that text becomes the system prompt. `VOICE_RULES` is
always appended: keep replies to 1–3 short spoken sentences, no markdown, no
emoji, use contractions, be warm and conversational. Without skmemory/SKCapstone,
skvoice falls back to the soul's static prompt, then to a generic assistant — it
never hard-fails on a missing dependency.

`voice_name` (from `soul/base.json`, default = agent name) is what the TTS
adapter sends as the `voice` field, so each agent speaks in its own cloned voice.

## LLM turn

`llm.get_response` builds the message list, pre-fetches memory, and makes a
single OpenAI-compatible chat call, with a second endpoint as a fallback
(`llm.py`):

```mermaid
flowchart TD
    IN["transcript + emotion + memory_ctx"] --> MSG["_to_plain_messages<br/>(merge consecutive same-role,<br/>coerce to {role, content:str})"]
    MSG --> CALL["POST Config.LLM_URL<br/>/v1/chat/completions (httpx)"]
    CALL --> OK{"HTTP ok and<br/>text non-empty?"}
    OK -->|yes| TXT["_strip_formatting<br/>(no markdown / no emoji)"]
    OK -->|"no: error or empty"| FB["POST Config.FALLBACK_URL<br/>Config.FALLBACK_MODEL"]
    FB --> TXT
```

- **No vendor SDK and no cloud credential.** Both legs are plain
  `/v1/chat/completions` over `httpx`. Any server speaking that API works.
- **Fallback trigger** is an error *or* an empty completion, not just an
  exception. Config for it is `SKVOICE_FALLBACK_URL` and
  `SKVOICE_FALLBACK_MODEL`.
- **Formatting strip** removes markdown and emoji, because the text is spoken.

### Voice tools: removed

> **`skvoice/tools.py` was deleted.** It defined five tools in Anthropic
> `tool_use` format (`search_memory`, `save_memory`, `web_search`,
> `dispatch_agent`, `cloud9_status`) plus a `handle_tool` dispatcher and a
> 4-round agentic loop driven from the old SDK path.
>
> **None of it had run since v0.2.6.** The tool loop went away with the SDK,
> nothing imported the module afterwards, and yet this file and the README kept
> advertising the five tools as live features. Documented-but-unreachable code
> is worse than absent code, because it gets planned against. So it was removed
> rather than left in place with a warning.
>
> The previous implementation is in git history if it is ever wanted back, but
> **do not restore it as-is**: the definitions are in Anthropic `tool_use`
> format, and the current `llm.py` speaks OpenAI-compatible chat. A tools
> feature has to be written against that API.

**skvoice has no model-callable tools today.** Memory is still consulted, but
only through the unconditional pre-fetch in `llm.get_response`, which the model
does not choose to invoke.

## Source map

| Module | Role |
|---|---|
| `skvoice/__main__.py` | Console entry point, `uvicorn.run(skvoice.service:app, Config.HOST:Config.PORT)`. Bind defaults to `127.0.0.1` |
| `skvoice/service.py` | FastAPI app, `/health`, `/voice/agents`, `/voice/clear`, the `/ws/voice/{agent}` WebSocket, the speech/text pipelines, group-chat & session-injection handling, caches |
| `skvoice/config.py` | Env-driven config + STT/TTS URL resolution (full-URL > base-URL > legacy alias > default) |
| `skvoice/agent_profile.py` | Profile loader + `skmemory ritual --full` rehydration + `VOICE_RULES` |
| `skvoice/memory.py` | `skmemory search` pre-fetch + `skmemory snapshot` save |
| `skvoice/llm.py` | Memory-aware turn against an OpenAI-compatible primary, then a sovereign fallback on failure or empty text; formatting strip |
| `skvoice/audio.py` | PCM→WAV, STT POST (`transcribe`), TTS POST (`synthesize`) |
| `skvoice/emotion.py` | RMS / ZCR / autocorrelation-pitch analysis → emotion tags + cue string |
| `skvoice/integration.py` | Optional skcapstone adapter — sk-alert, skscheduler health job, discovery registry |
| `systemd/skvoice.service` | systemd user unit (standalone lifecycle) |
| `scripts/musetalk-install.sh`, `docs/MUSETALK-INSTALL-PLAN.md` | Optional MuseTalk (talking-head) install path |

## Networking & URL resolution

skvoice never hardcodes the GPU host. `config._resolve_url` resolves each
service URL in priority order — full URL env (`SKVOICE_{TTS,STT}_URL`) → base
URL env (`SKVOICE_{TTS,STT}_BASE`, with the standard path appended) → legacy
alias (`SKVOICE_WHISPER_URL`) → built-in `localhost` default — so a single
`SKVOICE_STT_BASE=http://skworld-100:18794` is enough to point the whole
pipeline at a remote GPU box over Tailscale or LAN. The orchestrator is
stateless and lightweight, so you run one per agent against a shared
STT/TTS backend.

## Integration adapter (default-on-by-presence)

`integration.py` is the only place skvoice touches the wider fleet, and it is
strictly optional. `skcapstone` lives in the `[skcapstone]` extra; every call is
guarded by `is_present()` (importable **and** `SK_STANDALONE` unset **and**
`sdk.is_available()`), and any failure degrades to native behaviour.

```mermaid
flowchart TD
    EV["event in skvoice<br/>(e.g. speech_processing_failed)"] --> PRES{"is_present()?"}
    PRES -->|yes| BUS["sk-alert: publish skvoice.<severity><br/>(notify on warn/error/critical)"]
    PRES -->|no / fails| LOG["native structured log<br/>at matching level"]

    BOOT["service startup"] --> REGSELF["register_self()<br/>→ discovery registry (if present)"]
    BOOT --> SCHED["ensure_schedule()<br/>→ skvoice_health job in skscheduler (if present)"]
```

- **Alerts** use severity-based topics (`skvoice.<severity>`); the semantic
  event name rides in the payload, so the fleet's `*.error`/`*.critical` routing
  works while detail is preserved. The speech and text error paths in
  `service.py` emit `speech_processing_failed` / `text_processing_failed`.
- **`ensure_schedule()`** writes a `jobs.d/skvoice_health.yaml` drop-in that
  curls `/health` on an interval, so the skcapstone daemon owns the cadence and
  central retry/notify; idempotent on every startup.
- **`register_self()`** advertises skvoice (with `health_url` + pid file) to the
  discovery registry.

In standalone mode none of this runs — structured logging plus the systemd
`skvoice.service` unit carry the load.

## Where it lives in SKWorld

skvoice is the **Comms / voice** capability: the spoken-word transport, peer to
skchat (text) under skcomms. It consumes **Core** (SKCapstone profiles +
skmemory + cloud9 for who the agent *is*) and **Compute** (your GPU STT/TTS, and
skmodel/Ollama as LLM fallback), and is deployed like every other sk\* service
through skos.

```mermaid
flowchart TD
    subgraph COMMS["Comms"]
      SKVOICE["**skvoice**<br/>voice pipeline · per-agent WS"]
      SKCHAT["skchat"]
      SKCOMMS["skcomms (transport bus)"]
    end
    subgraph CORE["Core"]
      SKCAP["skcapstone (profiles/ritual)"]
      SKMEM["skmemory"]
      CLOUD9["cloud9 (FEB/OOF)"]
      CAPAUTH["capauth (identity)"]
    end
    subgraph COMPUTE["Compute (your hardware)"]
      STT["faster-whisper"]
      TTS["Chatterbox/VoxCPM"]
      SKMODEL["skmodel → Ollama"]
    end
    LLM["Anthropic Claude"]
    subgraph PLAT["Platform primitives"]
      ALERT["sk-alert"]
      SCHED["skscheduler"]
    end

    SKVOICE --> SKCAP & SKMEM & CLOUD9
    SKVOICE --> STT & TTS
    SKVOICE --> LLM
    LLM -.-> SKMODEL
    SKVOICE -.-> ALERT & SCHED
    SKCHAT -.->|WS proxy| SKVOICE
```

---

Part of the **[SKWorld](https://skworld.io)** sovereign ecosystem · 🐧 smilinTux
