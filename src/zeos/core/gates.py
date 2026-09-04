# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Action gates: the semantic check on the way *out*.

Capability checks are **syntactic** -- which pipes -- and NLI is honest that this leaves
a residual:

> A composition of individually-permitted actions can be dangerous (the job
> legitimately holds the lift actuator; it lifts the beam over the walkway).

An action gate is the endorser pattern pointed at output instead of input.
An endorser stands between untrusted content and a job's context and may refuse to
let content in; a gate stands between a job's intended actuation and the device pipe
and may refuse to let it out. Same shape, opposite direction, and the same reason for
existing: the judgment lives in

> a small, auditable, separately-owned file -- the safety team's file -- rather than
> diffused through every task.

**Gates are jobs.** Not kernel policy, not a callback, not a config predicate: an
ordinary descriptor, spawned by the kernel, budgeted, schedulable, preemptible or
pinned as its latency demands, and testable in CI by firing synthetic plans at it. That
matters for three reasons. It means the gate's own reasoning is journaled like any
other job's. It means a gate that hangs is a *scheduling* problem the kernel already
knows how to describe, rather than a novel deadlock. And it means the safety team can
write a gate with the same tools as everyone else, which is the difference between a
mechanism people use and a mechanism people work around.

The held write reuses ``pending_write`` -- the same parking spot backpressure uses.
A write waiting for a verdict and a write waiting for buffer space are the same
situation from the job's point of view: it asked to act, it has not acted yet, and it
is descheduled until it can. Nothing new was needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import DescriptorName, JobId, PipeName

__all__ = [
    "GateSpec",
    "GateRequest",
    "GateVerdict",
    "GateTable",
    "parse_verdict",
    "ALLOW",
    "VETO",
]

ALLOW = "allow"
VETO = "veto"


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One guard on one actuator path.

    ``requests`` and ``verdicts`` are ordinary pipes, which is what keeps the gate an
    ordinary job. The kernel writes the intended action to ``requests``; the gate
    reads it, decides, and writes to ``verdicts``; the kernel resolves the held write.
    """

    #: The device pipe being guarded.
    pipe: PipeName
    #: The guard descriptor.
    descriptor: DescriptorName
    requests: PipeName
    verdicts: PipeName
    #: What happens if the gate faults, is cancelled, or never answers. Defaults to
    #: refusing, because a gate that cannot answer is not evidence that the action is
    #: safe -- and "fail open" on the last check before an actuator is how safety
    #: interlocks become decorative.
    on_gate_failure: str = VETO
    #: Ticks the actuating job may wait before the failure policy applies. A gate is
    #: a job and jobs can block; an unbounded wait would turn a stuck gate into a
    #: silently stalled actuator.
    timeout_ticks: int = 32

    def render(self) -> str:
        return f"{self.descriptor} guards {self.pipe}"


@dataclass(frozen=True, slots=True)
class GateRequest:
    """A write held pending a verdict."""

    job: JobId
    pipe: PipeName
    gate: DescriptorName
    payload: str
    gate_job: JobId | None = None
    #: Token clock at which the timeout policy applies.
    deadline: int = 0

    def render(self) -> str:
        return f"job {self.job} → {self.pipe}: {self.payload!r}"


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """What the gate decided."""

    allowed: bool
    reason: str = ""

    def render(self) -> str:
        return ALLOW if self.allowed else f"{VETO}: {self.reason}"


def parse_verdict(text: str) -> GateVerdict | None:
    """Read a verdict out of what the gate wrote.

    The grammar is deliberately two words wide. A gate that answers in prose is a
    gate whose answer has to be interpreted, and interpreting the safety layer's
    output with a language model would put the persuadable component back on the
    critical path -- after taking some trouble to get it off.

    Anything unrecognised returns ``None``, which the kernel treats as the gate
    having failed rather than as consent.
    """
    stripped = " ".join(text.strip().split())
    lowered = stripped.lower()
    if lowered == ALLOW or lowered.startswith(f"{ALLOW}:") or lowered.startswith(f"{ALLOW} "):
        _, _, reason = stripped.partition(":")
        return GateVerdict(allowed=True, reason=reason.strip())
    if lowered == VETO or lowered.startswith(f"{VETO}:") or lowered.startswith(f"{VETO} "):
        _, _, reason = stripped.partition(":")
        return GateVerdict(allowed=False, reason=reason.strip() or "no reason given")
    return None


@dataclass
class GateTable:
    """Which pipes are gated, and by what.

    A pipe has at most one gate. Chaining guards would need a defined composition
    rule -- do two gates both veto, or does the first allow short-circuit? -- and
    picking one silently is worse than requiring the site to compose its checks
    inside a single reviewable file, which is where the design wants the judgment anyway.
    """

    gates: dict[PipeName, GateSpec] = field(default_factory=dict[PipeName, GateSpec])

    def __init__(self, specs: Iterable[GateSpec] = ()) -> None:
        self.gates = {}
        for spec in specs:
            self.declare(spec)

    def declare(self, spec: GateSpec) -> None:
        existing = self.gates.get(spec.pipe)
        if existing is not None and existing.descriptor != spec.descriptor:
            raise ValueError(
                f"pipe {spec.pipe!r} is already gated by {existing.descriptor!r}; "
                f"a pipe has at most one gate"
            )
        self.gates[spec.pipe] = spec

    def for_pipe(self, pipe: PipeName) -> GateSpec | None:
        return self.gates.get(pipe)

    def by_verdict_pipe(self, pipe: PipeName) -> GateSpec | None:
        return next(
            (self.gates[p] for p in sorted(self.gates) if self.gates[p].verdicts == pipe),
            None,
        )

    def guards(self) -> tuple[DescriptorName, ...]:
        return tuple(sorted({self.gates[p].descriptor for p in self.gates}))

    def names(self) -> tuple[PipeName, ...]:
        return tuple(sorted(self.gates))

    def all(self) -> tuple[GateSpec, ...]:
        return tuple(self.gates[p] for p in self.names())

    def __len__(self) -> int:
        return len(self.gates)


def gate_problems(
    table: GateTable,
    *,
    descriptors: Mapping[DescriptorName, object],
    pipes: Sequence[PipeName],
) -> tuple[str, ...]:
    """Load-time checks on the gate wiring.

    A misconfigured gate is the worst kind of safety mechanism: it reads as present
    in the config and is absent in effect. All three of these fail that way -- a gate
    naming a descriptor that does not exist can never be spawned, one that cannot
    write its verdict pipe can never answer, and one guarding a pipe nobody declared
    guards nothing.
    """
    problems: list[str] = []
    known = set(pipes)
    for spec in table.all():
        if spec.descriptor not in descriptors:
            problems.append(f"{spec.pipe}: gate descriptor {str(spec.descriptor)!r} does not exist")
            continue
        guard = descriptors[spec.descriptor]
        held = {capability.pipe for capability in getattr(guard, "capabilities", ())}
        if spec.verdicts not in held:
            problems.append(
                f"{spec.descriptor}: guards {spec.pipe} but holds no capability for "
                f"its verdict pipe {spec.verdicts!r}, so it can never answer"
            )
        bindings = getattr(guard, "pipes", None)
        stdin = getattr(bindings, "stdin", None) if bindings is not None else None
        if stdin != spec.requests:
            problems.append(
                f"{spec.descriptor}: guards {spec.pipe} but does not read its request "
                f"pipe {spec.requests!r} as stdin, so it can never see the action"
            )
        if known and spec.pipe not in known:
            problems.append(f"{spec.descriptor}: guards undeclared pipe {spec.pipe!r}")
    return tuple(problems)
