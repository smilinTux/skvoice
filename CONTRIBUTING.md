# Contributing to skvoice

skvoice is a voice orchestrator that owns no models. Nearly every change here is
either a change to the WebSocket protocol, a change to how a turn is assembled,
or a change to which HTTP service gets called. Keep that framing in mind: if a
change makes skvoice do model work locally, it is the wrong change for this
repo.

Read `SOP.md` first. It has the architecture, the exposure posture, the known
drift, and the troubleshooting table. `SECURITY.md` has the threat posture.

## Ground rules

1. **Do not invent operational detail in a doc.** Every claim carries its
   verifier: a command, a test name, a `file:line`. A confident wrong doc is
   worse than an honest partial one, because it gets trusted.
2. **The service must degrade, never crash.** Every external call in this repo
   (STT, TTS, both LLM legs, the `skmemory` subprocess) is wrapped so a failure
   returns an empty result and the WebSocket survives. Preserve that. A backend
   being down must not close a client's socket.
3. **No secrets, ever.** Not in code, not in a test fixture, not in a committed
   `.env`. `secret-scan` runs the gitleaks binary over the full history on every
   push and pull request.
4. **No em dashes or en dashes** in any text you write: docs, comments, commit
   messages, PR bodies. Use commas, parentheses, a colon, or a new sentence.
   Regular hyphens are fine.

## Setup

```bash
git clone https://github.com/smilinTux/skvoice
cd skvoice
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

`fetch-depth: 0` matters: setuptools-scm derives the version from the git tag,
and a shallow clone with no tags produces `0.0.0+unknown`.

## Tests

```bash
python -m pytest tests/ -q
```

Expected: **2 passed, 4 skipped**. The skips are the integrated-mode cases,
which need the optional `skcapstone` package.

If you get six errors saying `Integration test leaked files in
~/.skcapstone/config/jobs.d`, that is a **false positive on a host where skvoice
is running**: the live service already wrote `skvoice_health.yaml` there at
startup, and the autouse fixture in `tests/test_integration_adapter.py:34-41`
cannot tell that apart from a leak. Rerun with an isolated home:

```bash
env -i PATH=/usr/bin:/bin HOME=$(mktemp -d) .venv/bin/python -m pytest tests/ -q
```

Fixing that fixture is a welcome contribution.

## What to test

The existing suite covers the integration adapter only. **A green run is not
evidence the voice pipeline works.** If you touch the pipeline, say in the PR
how you exercised it against a real STT/TTS pair, and paste the `/health` output
you tested against.

New tests should stay hermetic: no network, no live service, no `systemctl`, no
`ssh`. If a test needs a running skvoice, it does not belong in `tests/`.

## CI gates

| Workflow | Trigger | Blocks a PR? |
|---|---|---|
| `ci.yml` | push, pull request | **yes**, this is the test gate |
| `docs-check.yml` | push, pull request | **yes**, tiers 1 and 2 |
| `secret-scan.yml` | push, pull request | **yes** |
| `publish.yml` | `v*` tag, manual dispatch | no, and its `test` job carries `continue-on-error: true`, so it can never fail. Do not cite it as evidence |

`docs-check` tier 2 means: **if your PR touches `pyproject.toml`, it must also
touch `CHANGELOG.md`.** Add an entry under `## [Unreleased]`.

Never land a gate that starts red. If you add a check, prove it can fail before
you propose it.

## Documentation you must keep in step

`SOP.md` ends with a `docs-evidence` block: a set of hermetic shell checks that
each exit 0 while a documented fact holds and non-zero when it drifts. If you
change one of the facts they pin (the entry point, the default port, the bind
host, the `ExecStart` line, the WebSocket route names, the default model, the
license, the setuptools-scm setup), **the matching check will fail and that is
the gate working**. Update the prose and the check together, in the same PR.

Run it locally before pushing:

```bash
python3 path/to/sk-standards/scripts/docs_check.py --repo . --tier 1 --tier 3
```

## Branches, commits, and PRs

- Branch from `main`: `fix/...`, `feat/...`, `docs/...`, `ci/...`.
- **Never commit to `main` directly, and never push a tag.** A `v*` tag triggers
  a PyPI publish, and `publish.yml` also contains a job that auto-cuts the next
  patch tag. Record the baseline with `git ls-remote --tags origin` before
  pushing anything from an unfamiliar clone, and check `.git/hooks/pre-push`.
- Conventional-commit subjects: `feat(service): ...`, `fix(llm): ...`,
  `docs(sop): ...`, `ci: ...`.
- Explain **why**, not just what. This repo's git history is unusually good at
  this and several comments in `publish.yml` exist only because someone wrote
  down the failure that caused them. Keep that up.
- One logical change per PR.

Commit trailer for AI-assisted work:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## Releasing

Maintainers only. A release is a `v*` tag pushed to `main`; `publish.yml` builds
and publishes via PyPI Trusted Publishing. See `SOP.md` section 5 for the guards
and the rollback procedure. Do not hardcode a version anywhere.

## Changing the exposure posture

**Done, as of the `SKVOICE_HOST` change.** The bind is now
`os.getenv("SKVOICE_HOST", "127.0.0.1")` via `Config.HOST`, the `docs-evidence`
checks pin the new default instead of the old literal, and operators who need
the wildcard set the variable. **The still-open half is authentication**: no
route in `skvoice/service.py` has any, so the bind address remains the only
access control. A PR that adds auth is wanted.

If you touch the exposure posture again, the same rules apply:

- prefer a new variable with a safe default over changing behaviour silently,
- **update the `docs-evidence` block in `SOP.md` in the SAME PR.** It runs at
  `tiers: "1,2,3"`, so it is executed in CI, and it pins the bind. Changing the
  bind without updating it turns the gate red. That is the gate working.
- call out any default change in the PR body, with the callers you checked and
  the ones you could not.

## Reporting a security issue

Do not open a public issue. See `SECURITY.md`: GitHub private vulnerability
reporting, 72 hour acknowledgement.

## Code of Conduct

Participation is governed by `CODE_OF_CONDUCT.md`.
