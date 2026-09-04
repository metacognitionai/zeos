# ZEOS -- a transformer operating system

[![ci](https://github.com/metacognitionai/zeos/actions/workflows/ci.yml/badge.svg)](https://github.com/metacognitionai/zeos/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/zeos)](https://pypi.org/project/zeos/)
[![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://github.com/metacognitionai/zeos/blob/main/pyproject.toml)
[![license](https://img.shields.io/github/license/metacognitionai/zeos)](https://github.com/metacognitionai/zeos/blob/main/LICENSE)
[![docs](https://img.shields.io/badge/docs-design%20corpus-blue)](https://github.com/metacognitionai/zeos/tree/main/docs/ProjectDescription/)

ZEOS programs complex, real-time, transformer-mediated systems the way real-time
operating systems are programmed: a hierarchy of task descriptors scheduled by a
kernel, with interrupts driving preemptive switching, pipes providing both dataflow
and blocking-based scheduling, and a stack of suspended job states enabling
resumption.

> **Jobs block, never poll; the kernel wakes, never asks.**

> **Initial release.** ZEOS is under active development. This release covers
> jobs, interrupts, and pipes; future releases will expand to other ZEOS
> features. The public API and journal formats may change between releases.

## Key paradigms

- **Scheduling belongs to a kernel, not the model** -- jobs block on pipes at zero
  cost, and a higher-priority event preempts within one token boundary instead of
  waiting to be noticed.
- **One behaviour, one file** -- you write each behaviour on its own, as if it were
  the robot's only job, and the kernel combines them into the whole system.
- **Text can persuade; only the kernel can permit** -- content carries the ring of
  where it came from, effects are capability-checked at pipe boundaries, and prompt
  injection becomes a managed fault class instead of an open-ended vulnerability.
- **Resume is not transparent** -- a suspended job's saved state contains beliefs
  about a world that moved, so the kernel injects a diff of exactly what changed
  before the job acts again.

## Quick start

Run the kernel on two jobs that count to five, spawn a successor and exit.

```bash
# 1. Install (needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh)
git clone git@github.com:metacognitionai/zeos.git
cd zeos
uv sync

# 2. Run it
uv run python demo/counter/run.py --case simple_count
```

Every line is one journal event, prefixed by the tick that produced it, with the text of
each decode beneath it. [`demo/counter/`](https://github.com/metacognitionai/zeos/blob/main/demo/counter/README.md) explains the events and
puts the same case on a live model instead -- the Claude API, or a local llama.cpp Qwen.
The kernel cannot tell the difference.

For ZEOS driving a real model at more interesting scale, see
[`demo/coop-count/`](https://github.com/metacognitionai/zeos/blob/main/demo/coop-count/README.md): two agents that hand a baton to each other
through a pipe, interrupted by a keypress that rewrites the count.
See [`demo/coop-count/TUTORIAL.md`](https://github.com/metacognitionai/zeos/blob/main/demo/coop-count/TUTORIAL.md) for a better idea of how
the mechanics of ZEOS work.

## The ZEOS design

The design is specified in [`docs/ProjectDescription/`](https://github.com/metacognitionai/zeos/tree/main/docs/ProjectDescription/):

| Document | Covers |
| --- | --- |
| `Transformer-OS.md` | **<-- Start Here!** Core: jobs, priorities, pipes, interrupts, the suspension stack, resume |
| `MP-Protected-Mode.md` | Protection: segments, rings, integrity watermark, capability-checked effects |
| `Programming-Model.md` | The paradigm, with a worked household-robot example |
| `Fleet.md` | Teams of robots: the same mechanisms reused |
| `NLI.md` | Why a malicious instruction fails structurally rather than by refusal |
| `Abstract-Machine.md` | The machine ZEOS runs on: the five ops, blocks, masking. Normative for `zeos.machine`, and the contract a real serving stack must meet |
| `Implementation.md` | The system layout: what is real vs. simulated, the module map, and the backend interface |

**Demonstration cases are not included in the kernel.** ZEOS proper contains no
application-specific information, code, data or dependencies. Cases are
application-specific by definition: it describes a particular operation, with particular
pipes, particular world objects and particular thresholds. Cases live with the demo that
needs them, under [`demo/`](https://github.com/metacognitionai/zeos/blob/main/demo/README.md), and nothing in `src/zeos/` imports from
there.


## Citing ZEOS

If you use ZEOS in your research, please cite it (see also [`CITATION.cff`](https://github.com/metacognitionai/zeos/blob/main/CITATION.cff)):

```bibtex
@software{zeos2026,
  author  = {van den Hengel, Anton and Avraham, Gil and Zhang, Jiahao and Gould, Stephen},
  title   = {{ZEOS}: A Transformer Operating System},
  year    = {2026},
  url     = {https://github.com/metacognitionai/zeos},
  version = {0.1.0}
}
```


## Licence

ZEOS is free software under the [GNU Affero General Public License v3.0 only](https://github.com/metacognitionai/zeos/blob/main/LICENSE)
(`AGPL-3.0-only`).
