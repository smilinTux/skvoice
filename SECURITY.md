# Security Policy

`skvoice` is a voice **orchestrator**. It hosts no models, stores no durable
state, and performs no cryptography. What it does hold, for the lifetime of a
WebSocket connection, is an agent's rehydrated identity (the `skmemory ritual
--full` output: soul, emotional state, seeds, strongest memories) and the
conversation in progress. That is the asset worth protecting here.

**Maturity-tier:** operational · **Crypto-component:** no · **Key-material
tier:** T0, N/A (skvoice generates, stores, signs with, and wraps **no** keys) ·
**Canonical-home:** this file.

## Supported versions

The version is derived from the git tag by setuptools-scm; there is no version
string in the tree. Only the latest released tag receives fixes.

| Version | Supported |
|---|---|
| Latest `v*` tag on `main` | ✅ |
| Any earlier tag | ❌ upgrade first |
| Unreleased `main` | best effort, no SLA |

Check what you are actually running with `curl -s localhost:18800/health` and
read the `version` field. The installed package and the newest tag routinely
disagree.

## Reporting a vulnerability

Report privately. Do **not** open a public issue for a security bug.

- **Primary channel:** GitHub private vulnerability reporting on this repo
  (`Security` ▸ `Report a vulnerability`).
- **Alternate:** a PGP-encrypted report to the SKWorld security contact via
  CapAuth identity, or the smilinTux / SKWorld maintainers through the SKCapstone
  coordination channel.

Include the affected version (from `/health`), a reproduction, and the impact you
observed.

**Acknowledgement SLA: 72 hours.** You will get a human reply confirming receipt
and an initial severity read within 72 hours of the report landing. Fix
timelines depend on severity and are agreed with you after triage.

### Safe harbour

Good-faith security research on this software is welcome and will not be met
with legal action. Good faith means: you test only against systems you own or
are authorized to test, you do not access, modify, or exfiltrate data belonging
to anyone else, you do not degrade service for other users, and you give us a
reasonable window to fix before disclosing publicly. If you follow that and
something still goes wrong, tell us; we will work it out.

### In scope

- The `skvoice` code in this repository.
- The HTTP routes (`/health`, `/voice/clear`, `/voice/agents`) and the WebSocket
  routes (`/ws/voice`, `/ws/video`, `/ws/facetime`) in `skvoice/service.py`.
- The FaceTime binary framing in `skvoice/facetime.py` (a parser reachable by
  untrusted input).
- Config handling in `skvoice/config.py`, including URL resolution.
- Subprocess invocation of the `skmemory` CLI in `skvoice/memory.py` and
  `skvoice/agent_profile.py`. (`skvoice/tools.py` also shelled out to
  `skmemory` and `openclaw`, but it was dead code and has been deleted.)
- The packaging and release path (`pyproject.toml`, `.github/workflows/`).

### Out of scope

- **The STT, TTS, and LLM services skvoice calls.** They are separate systems
  reached over HTTP; report those to their own projects.
- **`skmemory`, `skcapstone`, `skchat`, `cloud9`.** Separate repos, separate
  policies. A finding in the ritual output belongs to `skmemory`.
- **Operator deployment choices**: firewall rules, which interface the host
  exposes, whether the tailnet is trusted. Note the one exception below, which
  *is* in scope because the code forces it.
- Missing hardening headers on a service that serves no HTML.
- Denial of service by flooding an unauthenticated endpoint that is documented
  as unauthenticated. Report a *cheap* amplification or a crash, not raw volume.
- Reports produced solely by an automated scanner with no demonstrated impact.

## Known posture, stated plainly

These are **current, deliberate-or-known properties**, not vulnerabilities to
report. They are written down so a deployment decision is made with open eyes.

### 1. No authentication or authorization, anywhere

There is no auth middleware in `skvoice/service.py`. Any client that can reach
the port can:

- open `/ws/voice/{agent}` as **any** agent and converse with it, with that
  agent's full rehydrated identity as the system prompt;
- `POST /voice/clear` and drop **every** live conversation on the service;
- `GET /voice/agents` and enumerate every directory under
  `~/.skcapstone/agents`.

Access control is entirely the network's job today.

### 2. The bind address is the only access control, so it defaults to loopback

Because item 1 means nothing authenticates, **the bind address is the entire
access-control story.** `skvoice/__main__.py` passes `Config.HOST` to
`uvicorn.run`, and `Config.HOST` reads `SKVOICE_HOST` with a default of
`127.0.0.1`.

**Through v0.2.8 the host was the string literal `"0.0.0.0"` with no override,**
so those releases listened on every interface of the host and could not be
narrowed without editing the code. That deviated from
UNIFIED_INGRESS_STANDARD, which requires loopback or a tailnet bind and never a
wildcard. Setting `SKVOICE_HOST=0.0.0.0` restores the old behaviour if you need
it, and that is now an explicit choice rather than the only option.

**Upgrading from <= v0.2.8 changes behaviour.** A client that reached skvoice
over the LAN or the tailnet will stop connecting after the upgrade unless
`SKVOICE_HOST` is set. That is the intended direction: it fails closed. Audit
your callers first. `SOP.md` section 5 lists the ones a fleet grep found.

**Scope of the exposure on the current deployment**, so the risk is neither
overstated nor waved away: the host running skvoice has no public interface. Its
addresses are loopback, a LAN address, a Tailscale address, and docker bridges.
Internet ingress is Tailscale Funnel only, and Funnel publishes a specific list
of routes that **does not include `:18800`**. So the reachable surface is **LAN
plus tailnet**, not the internet, and that measurement was taken while the
service was still running a wildcard-bind build. On a differently configured
host, a wildcard bind plus no auth would be internet-exposed; do not deploy this
behind a public interface without a proxy that authenticates, whatever
`SKVOICE_HOST` is set to.

### 3. Agent identity is loaded into every session

On first use of an agent, `skvoice/agent_profile.py` shells out to `skmemory
ritual --full` and uses the result as the system prompt. That text contains the
agent's soul blueprint, FEB emotional state, seeds, and strongest memories. A
model can be induced to repeat its system prompt. Combined with item 1, anyone
who can reach the socket can attempt to read that material. Treat reachability
of `:18800` as equivalent to read access to the agent's identity.

### 4. Subprocess use

`skmemory` is invoked with `subprocess` and an argument **list** (never a shell
string), with the transcript passed as an argv element, so there is no shell
metacharacter path. The binary is located across `~/chatterbox-env/bin`,
`~/.skenv/bin`, then `PATH`. A writable directory earlier on that search order is
a code-execution path; keep those directories owned by the service user.

## Secret handling

**This repo stores no secrets and must never store one.** There is no key
generation, no signing, no encryption, no wrapping. `Config.CREDENTIALS_PATH`
points at `~/.claude/.credentials.json` but is **read by nothing** since the
Anthropic OAuth path was retired (`skvoice/llm.py:1-5`); it is a leftover, and
removing it is a tracked follow-up.

Credentials for anything skvoice calls belong in the operator's environment or
the KeePass vault (`skvault`), never in the repo, never in a card, never in a
`.env` file that is tracked.

`.github/workflows/secret-scan.yml` runs the **gitleaks binary** on every push
and pull request over the **full history**, and fails the build on a finding. The
binary rather than `gitleaks-action`, because that wrapper requires a paid
licence for organization-owned repos and exits before scanning a single byte,
producing a permanently red check that scans nothing and is therefore ignored.
The history scanned clean on 2026-08-14. If it ever goes red, a secret was
**added**: rotate it and purge it, do not weaken the gate to an incremental scan.

## Dependency posture

- Runtime dependencies are declared in `pyproject.toml`. `skcapstone` is
  **optional**, in the `[skcapstone]` extra, and every use is guarded by
  `integration.is_present()`.
- `anthropic>=0.40` is still declared but **imported by nothing** in `skvoice/`.
  Removing it is a tracked follow-up; until then it is unnecessary attack
  surface installed by default.
- The version is derived from the git tag (setuptools-scm). A hardcoded version
  drifts and rebuilds an already-published release; do not add one.
- Releases publish to PyPI via Trusted Publishing (OIDC), bound to
  `owner=smilinTux workflow=publish.yml environment=pypi`. No PyPI token exists
  in the CI path. Because the binding is to the workflow **filename**, the
  publish step cannot be relocated to another file without re-registering the
  publisher.

## What this repo does NOT claim

- It makes **no cryptographic claims of any kind**. It is not a crypto
  component. Nothing here is post-quantum, because nothing here is cryptography.
- Transport confidentiality is **not** provided by skvoice. It speaks plain
  `ws://` and plain HTTP on the LAN and tailnet. Tailscale provides encryption
  for tailnet traffic; LAN traffic is in the clear. If you need TLS, terminate
  it in a proxy in front of skvoice.
- It does **not** guarantee that agent identity material stays inside the
  process. See item 3.
- The test suite is small (2 executed cases plus 4 skipped integration cases)
  and covers the integration adapter only. **A green CI run on this repo is not
  evidence that the voice pipeline works.** Check `/health` and a real call.
