# Changelog

All notable changes to `skvoice` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html). **The git tag IS the
version** (setuptools-scm); a release is cut by pushing a `v*` tag. There is no
version string in the tree, so do not add one.

This file was started on 2026-08-14. Entries for releases before that date were
reconstructed from the git history and the tag list, and are deliberately terse:
they record what shipped, not a retro-fitted narrative.

## [Unreleased]

### Removed

- 🔴 **`skvoice/tools.py` is DELETED.** It defined five voice tools
  (`search_memory`, `save_memory`, `web_search`, `dispatch_agent`,
  `cloud9_status`), a `handle_tool` dispatcher, and a 4-round agentic loop, and
  `README.md` plus `docs/ARCHITECTURE.md` documented all five as **live
  features**. **None of them had run since v0.2.6**, when the Anthropic SDK path
  (and with it the tool loop) was retired. Nothing imported the module: a grep
  for `from skvoice.tools`, `from skvoice import tools`, `import skvoice.tools`,
  `VOICE_TOOLS`, and `handle_tool` across every `.py` under `~/clawd` found no
  importer, and `tests/` never referenced it.

  **If you were counting on any of those five tools, they were already gone.**
  This change removes the code and the false documentation, it does not remove
  a working feature. Memory is still consulted every turn through the
  unconditional pre-fetch in `llm.get_response`, which is not model-callable.

  The old implementation is in git history, but **do not restore it as-is**: its
  definitions are in Anthropic `tool_use` format and the current `llm.py` speaks
  OpenAI-compatible chat. A tools feature has to be written against that API.
  A `docs-evidence` check now fails if `skvoice/tools.py`, `VOICE_TOOLS`, or
  `def handle_tool` reappears under `skvoice/`.
- **`anthropic>=0.40` dropped from `pyproject.toml` dependencies.** Declared
  since the SDK path was retired in v0.2.6, imported by nothing:
  `grep -rni anthropic skvoice/` matches only a prose mention in `llm.py`'s
  docstring. Every install has been pulling the SDK and its transitive
  dependencies for no reason.
- **The `skvoice.skworld.io` link is out of the `README.md` footer.** The name
  does not resolve from this node (`getent hosts` returns nothing), so it is
  now noted as planned rather than advertised as live. No request was made to
  the public internet, and whether it resolves elsewhere was not tested.

### Security

- 🔐 **`SKVOICE_HOST` added; the bind now defaults to `127.0.0.1`.**
  `skvoice/__main__.py` passed the string literal `"0.0.0.0"` to `uvicorn.run`
  with **no environment override**, unlike every other configurable in the
  service. Since **no route in `skvoice/service.py` is authenticated** (`/health`,
  `/voice/agents`, `POST /voice/clear`, and the `/ws/voice/{agent}`,
  `/ws/video/{agent}`, `/ws/facetime/{agent}` WebSockets), the bind address is
  the only access control skvoice has, and it could not be narrowed without
  editing code. Anyone who could reach the port could converse as any agent
  (with that agent's full rehydrated identity as the system prompt), wipe every
  live conversation, or enumerate `~/.skcapstone/agents`.

  **Measured blast radius, not inflated:** LAN `192.168.0.0/16` plus the
  tailnet, **not the internet**. The host has no public interface and
  `tailscale funnel status` does not publish `:18800`.

  ⚠️ **This is a DEFAULT CHANGE and it can break a client.** A caller that
  reached skvoice over the LAN or the tailnet stops connecting after upgrade
  unless `SKVOICE_HOST` is set. It fails closed, which is the intent. Of the
  callers a fleet grep found, four use loopback and are unaffected; skchat's
  `deploy/skstack01-stack.yml` pins `ws://192.168.0.158:18800/ws/voice` and
  **would break**, though that stack is not currently deployed. A grep is not
  an audit. Read `SOP.md` section 5 before reinstalling, and set `SKVOICE_HOST`
  explicitly (preferring a tailnet address to `0.0.0.0`) if you need the old
  reach. **The live systemd unit and env file were deliberately not touched.**

  Authentication is still absent. The bind default contains the exposure; it
  does not authenticate anything.

### Added

- **`SOP.md`**, the standard operating procedures: architecture with mermaid
  diagrams, build, test, release and rollback, an explicit
  `### Front-end / Exposure` section, configuration reference, API reference,
  a symptom-to-check troubleshooting table, and maturity/version reference.
  It ends with a `docs-evidence` block: hermetic shell checks that each exit
  non-zero the moment a documented fact drifts from the code. Now 15 of them,
  after the bind and dead-code checks above were reworked.
- **`SECURITY.md`**: GitHub private vulnerability reporting as the primary
  channel, a 72 hour acknowledgement SLA, in and out of scope, a supported
  versions table, a safe-harbour statement, and a plainly stated known posture.
  skvoice is **not** a crypto component: key-material tier T0, N/A.
- **`CONTRIBUTING.md`** and **`CODE_OF_CONDUCT.md`** (Contributor Covenant 2.1).
- **`CHANGELOG.md`**, this file.
- **`.github/workflows/ci.yml`**: the first CI gate that actually runs the test
  suite on an ordinary push or pull request, across Python 3.10, 3.11, and 3.12,
  with no `continue-on-error`.
- **`.github/workflows/docs-check.yml`**: the shared `sk-standards` docs gate,
  now at `tiers: "1,2,3"` (required docs present; a code change also updates
  this file; **and every check in the `docs-evidence` block is executed**).
  Tier 3 being live is why a bind change has to update `SOP.md` in the same PR.
- A **`dev` extra** in `pyproject.toml` (`pytest`). `publish.yml:41` has always
  run `pip install -e ".[dev]" || pip install -e .`, but no `dev` extra existed,
  so the first half always failed, the fallback installed no pytest, and the
  test step died on "No module named pytest" into a `continue-on-error: true`
  that swallowed it. **The tests in this repo had never actually run in CI.**

### Fixed

- **`README.md` documented a design that was retired in v0.2.6.** The docs were
  written at v0.2.5 and the LLM leg was rewritten one release later; the docs
  never caught up. Corrected against `skvoice/config.py` and `skvoice/llm.py`:
  - `SKVOICE_MODEL` default is `claude-haiku-4-5`, not `claude-sonnet-4-6`.
  - `SKVOICE_OLLAMA_URL` and `SKVOICE_OLLAMA_MODEL` **do not exist**. The real
    variables are `SKVOICE_LLM_URL` (primary) and `SKVOICE_FALLBACK_URL` plus
    `SKVOICE_FALLBACK_MODEL` (fallback), both OpenAI-compatible
    `/v1/chat/completions` endpoints.
  - The Anthropic SDK and Claude OAuth path is gone (`skvoice/llm.py:1-5`), so
    "Claude via OAuth token or API key, falls back to local Ollama" was wrong,
    as was the Anthropic credentials requirement.
  - The Claude `tool_use` loop is **not wired in**: nothing imports
    `skvoice/tools.py`. The five voice tools were documented as live and are
    not. (Superseded: the module has since been deleted outright, see
    **Removed** above.)
  - `/ws/video/{agent}` and `/ws/facetime/{agent}`, shipped in v0.2.8, were
    undocumented in the README.
  - The listen address is `0.0.0.0`, and the README did not say that this is
    hardcoded with no environment override. (Superseded: `SKVOICE_HOST` now
    exists and defaults to loopback, see **Security** above.)
- **`docs/ARCHITECTURE.md`** carried the same retired LLM design. The "LLM turn
  & the tool loop" section has been **rewritten** against the current `llm.py`
  (single OpenAI-compatible call, fallback on error or empty text, no tool loop,
  no OAuth), and "Voice tools" is now a removal note rather than a table of
  five tools that do not exist. The sequence diagram no longer shows a
  `tools.py` participant, and the source map no longer lists the module. The
  pipeline, WebSocket protocol, agent/ritual, and integration sections remain
  accurate and are unchanged.

### Known, documented, still not changed here

The first three items below were raised by the documentation pass and have since
been **fixed** by the `SKVOICE_HOST` and dead-code entries above. They are left
listed, struck through, so the sequence stays legible. The rest remain open and
are recorded in `SOP.md` section 9 as follow-ups needing an operator decision:

- ~~The uvicorn bind host is hardcoded `0.0.0.0` with no override.~~ Fixed.
  **No route is authenticated, and that is still true.**
- ~~`skvoice/tools.py` is dead code.~~ Deleted.
- ~~`anthropic>=0.40` is declared but imported by nothing.~~ Dropped.
- `Config.CREDENTIALS_PATH` is read by nothing. Still present.
- 🔴 **The deployed build is six releases behind.** On both 2026-08-14 and
  2026-08-15, `/health` on noroc2027 reported `"version":"0.2.2"` while the
  newest remote tag was `v0.2.8`. **This is a deploy issue, not a code one**,
  and nothing in this changelog fixes it: the operator action is a reinstall
  plus restart. Note that the same reinstall is what activates the loopback
  bind default, so read the `SKVOICE_HOST` entry above first.
- `publish.yml` keeps `continue-on-error: true`, so a release can still ship
  on a red suite.
- The autouse fixture in `tests/test_integration_adapter.py:34-41` false-positives
  on any host where skvoice is running, because the live service writes
  `~/.skcapstone/config/jobs.d/skvoice_health.yaml` at startup.

## [0.2.8] - 2026-08-12

### Added

- `/ws/video/{agent}` and `/ws/facetime/{agent}`. skchat had a complete FaceTime
  feature (page, WebRTC, and a WebSocket fallback) proxying to these routes, but
  they were never implemented, so both answered **403** and the feature looked
  dead against a healthy server. `/ws/video` shares `_conversation_loop` with
  `/ws/voice` verbatim so the two cannot drift; `/ws/facetime` wraps the same
  turn in the 12-byte little-endian binary framing the fallback client parses
  (`skvoice/facetime.py`).

## [0.2.7] - 2026-08-12

### Changed

- Release plumbing: PyPI publish moved to a tag-triggered workflow with Trusted
  Publishing (OIDC), bound to `owner=smilinTux workflow=publish.yml
  environment=pypi`.

## [0.2.6] - 2026-06-12

### Changed

- **Retired the Anthropic SDK and the Claude Code OAuth path.** Cloud rate
  limits made it unreliable. Both LLM legs now speak the same OpenAI-compatible
  `/v1/chat/completions` API: a local `claude-haiku-4-5` proxy as primary, and
  a sovereign `qwen3.6-27b-abliterated` endpoint as fallback.
- Side effects of that change, not noticed at the time: the `tools.py` tool loop
  stopped being called, `Config.CREDENTIALS_PATH` became unused, the `anthropic`
  dependency became unused, and `README.md` plus `docs/ARCHITECTURE.md` went
  stale. All four are corrected or recorded in the Unreleased section above.

## [0.2.5] - 2026-06-11

### Added

- `docs/ARCHITECTURE.md` and a rewritten `README.md` on the SKStack v2 template,
  with placement and workflow mermaid diagrams.

## [0.2.4] - 2026-06-09

### Added

- The **skcapstone integration adapter** (`skvoice/integration.py`), following
  the default-on-by-presence pattern: alerts routed to `skvoice.<severity>` on
  the sk-alert bus, a `skvoice_health` job registered with skscheduler, and the
  service advertised in the discovery registry, all only when `skcapstone` is
  importable and `SK_STANDALONE` is unset. Covered by
  `tests/test_integration_adapter.py`.

## [0.2.3] - 2026-06-09

### Added

- README first-principles and vertical-placement section (Comms / Voice layer).

## [0.2.2] - 2026-04-27

Earliest release still reachable in the tag history. Releases at and before this
point were cut by an auto-bump on push and carry no per-release notes; read
`git log` for detail.

[Unreleased]: https://github.com/smilinTux/skvoice/compare/v0.2.8...HEAD
[0.2.8]: https://github.com/smilinTux/skvoice/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/smilinTux/skvoice/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/smilinTux/skvoice/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/smilinTux/skvoice/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/smilinTux/skvoice/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/smilinTux/skvoice/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/smilinTux/skvoice/releases/tag/v0.2.2
