# Agent instructions

ZEOS is a transformer operating system: a deterministic kernel that schedules,
protects, and pages LLM jobs. The design corpus is `docs/ProjectDescription/`;
the module map and what-is-real-vs-simulated live in
`docs/ProjectDescription/Implementation.md`. This file is the canonical agent
instructions; `CLAUDE.md` is a symlink to it -- edit only this file.

## Commands

```bash
uv sync                            # env from the committed lockfile
uv run pytest                      # the whole suite (~2s)
uv run pytest -m determinism       # the flagship gate alone
uv run pyright                     # strict, must stay at 0 errors
uv run pre-commit run --all-files  # ruff check/format, codespell, hygiene
```

All four gates must be green before any commit. CI runs exactly these.

## Hard rules

- **Determinism is the acceptance gate.** Same descriptor tree + same event
  schedule + same seed must produce a byte-identical journal. Nothing in
  `src/zeos/core/` may read a clock, use unseeded randomness, or iterate an
  unordered collection into a kernel decision. If the determinism gate goes
  red, you introduced one of those.
- **`src/zeos/core/` is stdlib-only.** Enforced by
  `tests/contract/test_core_is_stdlib_only.py`; the allowed import seams are
  listed there. Never add a third-party import anywhere the kernel reaches.
- **Golden traces are refreshed one way only:**
  `uv run python -m tests.replay.regenerate`, after deciding the shape change
  is intended, and the commit message says why. Never hand-edit
  `tests/replay/golden/`.
- **There are no flaky tests.** An intermittent failure is a bug in the kernel
  or the test, never something to retry or quarantine.
- **Integration assertions test journal properties**, never transcript text:
  "the alarm preempted supervision within one boundary" is a structural fact.
- New markers must be declared in `pyproject.toml` (`--strict-markers`).
- Journal events must not declare fields named `seq` or `kind` (codec guard).
- **Guard nothing "just in case."** The kernel fails loudly on purpose: a
  check belongs at a declared contract (a capability, a schema, a load-time
  lint rule) or not at all. A fallback that quietly absorbs a bad state is a
  bug wearing a seatbelt.
- **Document what exists, never what is planned.** Committed code, docs, and
  comments describe the repository as it is; roadmaps, "to be written" stubs,
  and future-tense promises stay out of them -- unless explicitly asked to
  write down a plan and execute it.
- **Find the cause before writing the fix.** If a patch makes a symptom go
  away and you cannot say which invariant was broken, you have hidden the bug,
  not removed it -- keep digging until the journal tells you what happened.

## Style

- Ruff is the only formatter and linter; the pre-commit hooks mirror CI.
- The prose register is deliberate: British English in docs and comments
  ("behaviour", "fulfilment" -- codespell knows), and comments state
  constraints the code cannot show, not narration of the next line.
- Reference design claims against the corpus before asserting them: the specs
  and the code deliberately mark what is real, simulated, or future work.

## PRs and merges

Main takes PRs only, squash-merged: the PR title becomes the commit subject
(imperative, names the symptom or capability). Follow the PR template: tests
for new code, no refactor-only PRs, one concern per PR, drafts for
work-in-progress, AI assistance disclosed.

## Layout notes

- Demos live in `demo/`, each as a uv workspace member with its own pyproject;
  a demo's package must never ship inside the zeos wheel.
- `runs/` directories and `.env` files are gitignored artifacts and secrets.
