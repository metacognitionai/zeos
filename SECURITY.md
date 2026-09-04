# Security policy

## Reporting a vulnerability

Report privately through GitHub: **[open a security advisory](https://github.com/metacognitionai/zeos/security/advisories/new)**.
Do not open a public issue for a vulnerability.

Please include what you need to make the case reproducible: the descriptor tree
or a minimal case, the event schedule, the seed, and the journal if you have
one. A ZEOS run is deterministic, so a report carrying those four things
reproduces exactly.

## Supported versions

ZEOS is pre-1.0 and under active development. Fixes go onto the latest release;
older versions are not patched.

## What is in scope

The kernel's protection claims are what this policy is about:

- **Ring and capability enforcement.** Content carries the ring of where it came
  from, and effects are capability-checked at pipe boundaries. A way to make the
  kernel permit an effect a job was not granted is a vulnerability.
- **Integrity watermark.** A way to raise content's integrity above its source's
  is a vulnerability.
- **Determinism.** The same descriptor tree, event schedule and seed must
  produce a byte-identical journal. A way to make a run diverge is a bug, and
  where it is reachable by supplied content, a vulnerability.
- **Descriptor loading.** A descriptor tree that escapes its case directory, or
  executes anything during a load or lint, is a vulnerability.

## What is not

- **A model doing what a prompt asked.** ZEOS does not claim a model can be
  argued out of anything. It claims the kernel decides what an effect is allowed
  to do regardless of what the model was persuaded to emit. A prompt that makes
  a job *say* something is not a finding; one that makes the kernel *permit*
  something is.
- **The debugger's HTTP server bound to a non-loopback address.** `zeos debug`
  serves a single fixed case to `127.0.0.1`. Exposing it to a network is a
  deployment choice, and it is not built to be one.
- **Anything under `demo/`.** The demonstrations are illustrative, carry their
  own dependencies, and are not part of the published `zeos` package.
