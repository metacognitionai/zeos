# Contributing to ZEOS

ZEOS takes changes through pull requests on GitHub. This guide says how to set up,
what we accept, and what a pull request needs before it is merged. The rules the
code must keep are in [`AGENTS.md`](AGENTS.md); they apply to people as much as to
agents, and this guide does not repeat them.

## Before you start

- For a bug, open an issue with the four things that make a ZEOS run reproduce
  exactly: the descriptor tree or a minimal case, the event schedule, the seed,
  and the journal if you have one.
- For a security problem, do not open an issue. Report it privately as
  [`SECURITY.md`](SECURITY.md) describes.
- For a new capability or a change to the kernel, open an issue first. Name the
  design document in [`docs/ProjectDescription/`](docs/ProjectDescription/) that
  covers it. If the change departs from what that document says, update the
  document in the same pull request.

## Setup

```bash
git clone git@github.com:metacognitionai/zeos.git
cd zeos
uv sync
```

Four gates must be green before you open a pull request. CI runs exactly these:

```bash
uv run pytest                      # the whole suite, about two seconds
uv run pytest -m determinism       # the flagship gate alone
uv run pyright                     # strict, zero errors
uv run pre-commit run --all-files  # ruff check and format, codespell, hygiene
```

## What we accept

Welcome:

- Fixes to documentation and to the design corpus.
- Bug fixes, each with a test that reproduces the bug.
- New features, discussed first as a feature-request issue. Once the issue is
  agreed, open a pull request for it and link the issue from the pull request.

Demonstrations built on ZEOS go to
[zeos-demos](https://github.com/metacognitionai/zeos-demos), not to this
repository. The demos under `demo/` here are the kernel's own examples and test
seats.

Changes under `src/zeos/core/` are held to the kernel's contracts: the package is
standard-library only, nothing in it reads a clock, uses unseeded randomness or
iterates an unordered collection into a decision, and the same descriptor tree,
event schedule and seed must produce a byte-identical journal. Integration tests
assert journal properties, never transcript text. Golden traces are refreshed
only with `uv run python -m tests.replay.regenerate`, after deciding the shape
change is intended, and the commit message says why.

Not accepted:

- Refactor-only pull requests.
- Checks added "just in case". The kernel fails loudly on purpose: a check
  belongs at a declared contract or not at all.
- Documentation of planned work. Committed text describes the repository as it is.
- A flaky test, or a retry around one. An intermittent failure is a bug.

## Pull requests

- One concern per pull request, as small as it can be. Open it as a draft while
  the work is in progress.
- The title is imperative and names the symptom or the capability, not the
  implementation: "fix: replay diverges when a vector fires on the last tick"
  rather than "fix: add null check in vectors.py". Main is squash-merged and the
  title becomes the commit subject.
- Fill in every section of the pull request template. Evidence shows the four
  gates and whatever makes validation easy, such as a journal excerpt or a
  before-and-after behaviour.
- New code comes with tests. A fix comes with the test that failed before it.
- Prose is British English, in documentation and in comments, and a comment
  states a constraint the code cannot show rather than narrating the next line.
- `main` is protected: pull requests only, with the `ci-gate` check green, and a
  maintainer's review.

## AI-assisted contributions

AI assistance is welcome and must be disclosed: one line in the pull request
saying what was AI-assisted is enough. A person who understands the change owns
it. That person has read every line, can explain why it is there, and has checked
every claim in the description against the code and the journal. Replies to a
review are written by that person; model output quoted in a reply is marked as a
quote.

## Contribution terms

ZEOS is free software under the GNU Affero General Public License v3.0 only. To
have a pull request merged you must have signed the
[Metacognition AI Contributor License Agreement](https://drive.google.com/file/d/1qW5waIW2u5z3RIZo7Yg7D1N9KZEtaPG1/view?usp=sharing),
once, for all your contributions.

A pull request from a contributor who has not signed is not merged.

## Releases

Maintainers cut releases as [`docs/release.md`](docs/release.md) describes.
A contribution does not bump the version.
