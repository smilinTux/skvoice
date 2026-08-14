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

### Added

- **`SOP.md`**, the standard operating procedures: architecture with mermaid
  diagrams, build, test, release and rollback, an explicit
  `### Front-end / Exposure` section, configuration reference, API reference,
  a symptom-to-check troubleshooting table, and maturity/version reference.
  It ends with a `docs-evidence` block: 12 hermetic shell checks that each exit
  non-zero the moment a documented fact drifts from the code.
- **`SECURITY.md`**: GitHub private vulnerability reporting as the primary
  channel, a 72 hour acknowledgement SLA, in and out of scope, a supported
  versions table, a safe-harbour statement, and a plainly stated known posture.
  skvoice is **not** a crypto component: key-material tier T0, N/A.
- **`CONTRIBUTING.md`** and **`CODE_OF_CONDUCT.md`** (Contributor Covenant 2.1).
- **`CHANGELOG.md`**, this file.
- **`.github/workflows/ci.yml`**: the first CI gate that actually runs the test
  suite on an ordinary push or pull request, across Python 3.10, 3.11, and 3.12,
  with no `continue-on-error`.
- **`.github/workflows/docs-check.yml`**: the shared `sk-standards` docs gate at
  tiers 1 and 2 (required docs present; a code change also updates this file).
  Tier 3, which executes the `docs-evidence` block, is a follow-up once the gate
  has run clean.
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
    not.
  - `/ws/video/{agent}` and `/ws/facetime/{agent}`, shipped in v0.2.8, were
    undocumented in the README.
  - The listen address is `0.0.0.0`, and the README did not say that this is
    hardcoded with no environment override.
- **`docs/ARCHITECTURE.md`** carried the same retired LLM design. An accuracy
  note now marks the two affected sections and points at `SOP.md`. The pipeline,
  WebSocket protocol, agent/ritual, and integration sections remain accurate and
  are unchanged.

### Known, documented, deliberately not changed here

This is a documentation change; no runtime behaviour was touched. Recorded in
`SOP.md` section 9 as follow-ups needing an operator decision:

- The uvicorn bind host is hardcoded `0.0.0.0` with no override
  (`skvoice/__main__.py:9-12`), which deviates from UNIFIED_INGRESS_STANDARD,
  and no route is authenticated. On the current host the reachable surface is
  LAN plus tailnet, not the internet: there is no public interface and Tailscale
  Funnel does not publish `:18800`.
- `skvoice/tools.py` is dead code, `anthropic>=0.40` is declared but imported by
  nothing, and `Config.CREDENTIALS_PATH` is read by nothing.
- `publish.yml:43` keeps `continue-on-error: true`, so a release can still ship
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
