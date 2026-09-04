# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Success criteria, evaluated from the journal.

Every criterion is a structural fact about a run, not a judgement about a
transcript. That is the same discipline the test tiers use, and for the same
reason: "the alarm preempted supervision within one boundary" is
checkable, whereas "the response looks sensible" is not.

A criterion carries a ``because``: the sentence explaining why the problem author
thinks this matters. It is printed on failure, so a scorecard reads as an argument
rather than a list of red crosses -- and writing it forces the problem author to know
what they are actually testing.

Two criteria are worth singling out:

* ``latency`` measures **event-to-first-corrective-action**, which is the number the
  safety case actually cares about and the one the placement rule compares against a
  link. Not dispatch latency, not first token -- first
  *action*.
* ``never_written_above_integrity`` asserts that a privileged effect never happened
  while the acting job was tainted. It is how a problem states "a ring-3 vendor feed
  must be structurally incapable of moving the plant", and it passes or fails on the
  capability check rather than on anybody's good intentions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zeos.core.events import (
    CapabilityChecked,
    Event,
    FaultRaised,
    JobCompleted,
    JobPreempted,
    JobResumed,
    JobSpawned,
    PipeWritten,
    VectorFired,
)
from zeos.core.ids import DescriptorName, FaultKind, Integrity, JobId, ObjectName, PipeName
from zeos.descriptor.schema import parse_duration_ns
from zeos.world.store import WorldStore

__all__ = ["Verdict", "EvalContext", "Criterion", "parse_criteria", "evaluate_all"]


@dataclass(frozen=True, slots=True)
class Verdict:
    criterion_id: str
    kind: str
    passed: bool
    detail: str
    because: str = ""

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        line = f"  [{mark}] {self.criterion_id}: {self.detail}"
        if not self.passed and self.because:
            line += f"\n         why it matters: {self.because}"
        return line


@dataclass(frozen=True, slots=True)
class EvalContext:
    events: Sequence[Event]
    world: WorldStore
    #: job → descriptor, so criteria can be written against behaviour names rather
    #: than against job ids, which are an allocation detail.
    job_names: Mapping[JobId, DescriptorName]

    def of[E: Event](self, cls: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, cls)]

    def name_of(self, job: JobId) -> str:
        return str(self.job_names.get(job, f"job-{job}"))


@dataclass(frozen=True, slots=True)
class Criterion:
    id: str
    kind: str
    because: str = ""
    spec: Mapping[str, Any] = None  # type: ignore[assignment]

    def evaluate(self, ctx: EvalContext) -> Verdict:
        handler = _HANDLERS.get(self.kind)
        if handler is None:
            return self._verdict(False, f"unknown criterion kind {self.kind!r}")
        try:
            return handler(self, ctx)
        except KeyError as exc:
            return self._verdict(False, f"criterion is missing field {exc.args[0]!r}")

    def _verdict(self, passed: bool, detail: str) -> Verdict:
        return Verdict(
            criterion_id=self.id,
            kind=self.kind,
            passed=passed,
            detail=detail,
            because=self.because,
        )


def parse_criteria(raw: Sequence[Mapping[str, Any]]) -> tuple[Criterion, ...]:
    criteria: list[Criterion] = []
    for index, entry in enumerate(raw):
        kind = str(entry.get("kind", "")).strip()
        if not kind:
            raise ValueError(f"criterion {index} has no 'kind'")
        criteria.append(
            Criterion(
                id=str(entry.get("id", f"{kind}-{index}")),
                kind=kind,
                because=str(entry.get("because", "")).strip(),
                spec=entry,
            )
        )
    return tuple(criteria)


def evaluate_all(criteria: Sequence[Criterion], ctx: EvalContext) -> tuple[Verdict, ...]:
    return tuple(c.evaluate(ctx) for c in criteria)


# --- individual criteria ----------------------------------------------------


def _world_equals(c: Criterion, ctx: EvalContext) -> Verdict:
    obj = ObjectName(str(c.spec["object"]))
    expected = str(c.spec["value"])
    actual = ctx.world.get(obj, "(unset)")
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        actual == expected,
        f"{obj} = {actual!r} (expected {expected!r})",
    )


def _latency(c: Criterion, ctx: EvalContext) -> Verdict:
    """Event-to-first-corrective-action, in virtual nanoseconds."""
    source = PipeName(str(c.spec["source"]))
    deadline = parse_duration_ns(c.spec["deadline"], field_name=f"{c.id}: deadline")
    action_pipe = c.spec.get("action_pipe")

    arrivals = [e for e in ctx.of(PipeWritten) if e.pipe == source and e.job is None]
    if not arrivals:
        return c._verdict(False, f"no event ever arrived on {source}")  # pyright: ignore[reportPrivateUsage]
    fired_at = arrivals[0].clock.virtual_ns

    actions = [
        e
        for e in ctx.of(PipeWritten)
        if e.job is not None
        and e.clock.virtual_ns >= fired_at
        and (action_pipe is None or e.pipe == PipeName(str(action_pipe)))
    ]
    if not actions:
        return c._verdict(False, "no corrective action was ever taken")  # pyright: ignore[reportPrivateUsage]

    elapsed = actions[0].clock.virtual_ns - fired_at
    if deadline is None:
        return c._verdict(True, f"responded in {elapsed}ns (no deadline set)")  # pyright: ignore[reportPrivateUsage]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        elapsed <= deadline,
        f"event-to-action {elapsed}ns against a {deadline}ns budget (action: {actions[0].pipe})",
    )


def _no_fault(c: Criterion, ctx: EvalContext) -> Verdict:
    listed: Sequence[Any] = c.spec.get("faults") or []
    wanted = {FaultKind(str(f)) for f in listed} or set(FaultKind)
    seen = [f for f in ctx.of(FaultRaised) if f.fault in wanted]
    if not seen:
        return c._verdict(True, "no such faults were raised")  # pyright: ignore[reportPrivateUsage]
    first = seen[0]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        False,
        f"{len(seen)} fault(s); first was {first.fault.value} on "
        f"{ctx.name_of(first.job)}: {first.detail}",
    )


def _fault_expected(c: Criterion, ctx: EvalContext) -> Verdict:
    """Some problems are passed by *failing loudly* -- an injection attempt that
    raises a privilege fault is the correct outcome, not a defect."""
    kind = FaultKind(str(c.spec["fault"]))
    seen = [f for f in ctx.of(FaultRaised) if f.fault is kind]
    if not seen:
        return c._verdict(False, f"expected a {kind.value} and none was raised")  # pyright: ignore[reportPrivateUsage]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        True, f"{kind.value} raised on {ctx.name_of(seen[0].job)}: {seen[0].detail}"
    )


def _job_completed(c: Criterion, ctx: EvalContext) -> Verdict:
    wanted = DescriptorName(str(c.spec["descriptor"]))
    completed = {ctx.job_names.get(e.job) for e in ctx.of(JobCompleted)}
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        wanted in completed,
        f"{wanted} {'completed' if wanted in completed else 'did not complete'}",
    )


def _pipe_written(c: Criterion, ctx: EvalContext) -> Verdict:
    pipe = PipeName(str(c.spec["pipe"]))
    writes = [e for e in ctx.of(PipeWritten) if e.pipe == pipe and e.job is not None]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        bool(writes), f"{len(writes)} write(s) to {pipe}"
    )


def _never_written_above_integrity(c: Criterion, ctx: EvalContext) -> Verdict:
    """A privileged effect must never happen while the acting job is tainted.

    Checked against the capability layer rather than against outcomes, because a
    solution that simply never got round to the write would otherwise pass for the
    wrong reason.
    """
    pipe = PipeName(str(c.spec["pipe"]))
    ceiling = Integrity(int(c.spec.get("max_integrity", 2)))
    violations = [
        e
        for e in ctx.of(CapabilityChecked)
        if e.pipe == pipe and e.allowed and int(e.effective_integrity) > int(ceiling)
    ]
    if violations:
        bad = violations[0]
        return c._verdict(  # pyright: ignore[reportPrivateUsage]
            False,
            f"{ctx.name_of(bad.job)} wrote {pipe} at integrity "
            f"{bad.effective_integrity}, above the ceiling of {ceiling}",
        )
    checks = [e for e in ctx.of(CapabilityChecked) if e.pipe == pipe]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        True,
        f"{len(checks)} capability check(s) on {pipe}, none above integrity {ceiling}",
    )


def _preemption_after(c: Criterion, ctx: EvalContext) -> Verdict:
    """A tighter-deadline behaviour must be able to *displace* a looser one.

    This is the criterion that separates a design which meets a deadline from one
    which merely happened to. A monolith can pass a latency budget when the event
    lands somewhere convenient in its script, but its response time is bounded by
    whatever else it was doing -- so the margin is a property of the scenario, not of
    the system. Requiring preemption asks the question the safety case actually
    asks: can the urgent thing interrupt the unhurried thing, whenever it arrives?

    Stated against the source pipe, so it constrains no particular solution shape.
    """
    source = PipeName(str(c.spec["source"]))
    arrivals = [e for e in ctx.of(PipeWritten) if e.pipe == source and e.job is None]
    if not arrivals:
        return c._verdict(False, f"no event ever arrived on {source}")  # pyright: ignore[reportPrivateUsage]
    after = arrivals[0].clock.virtual_ns
    preemptions = [e for e in ctx.of(JobPreempted) if e.clock.virtual_ns >= after]
    if not preemptions:
        return c._verdict(  # pyright: ignore[reportPrivateUsage]
            False,
            f"nothing preempted anything after {source} fired -- the response had to "
            "wait for whatever was already running to finish",
        )
    first = preemptions[0]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        True,
        f"{ctx.name_of(first.by_job)} (priority {first.by_priority}) preempted "
        f"{ctx.name_of(first.job)}",
    )


def _preempted(c: Criterion, ctx: EvalContext) -> Verdict:
    victim = str(c.spec["job"])
    by = str(c.spec["by"])
    matches = [
        e
        for e in ctx.of(JobPreempted)
        if ctx.name_of(e.job) == victim and ctx.name_of(e.by_job) == by
    ]
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        bool(matches), f"{by} preempted {victim} {len(matches)} time(s)"
    )


def _resumed_dirty_naming(c: Criterion, ctx: EvalContext) -> Verdict:
    """The resumed job must be told about the state it depends on that moved."""
    obj = ObjectName(str(c.spec["object"]))
    resumes = [r for r in ctx.of(JobResumed) if any(d.obj == obj for d in r.dirty)]
    if not resumes:
        return c._verdict(  # pyright: ignore[reportPrivateUsage]
            False, f"no resume notice named {obj}"
        )
    delta = next(d for d in resumes[0].dirty if d.obj == obj)
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        True,
        f"{ctx.name_of(resumes[0].job)} resumed knowing {obj}: {delta.before!r} -> {delta.after!r}",
    )


def _vector_fired(c: Criterion, ctx: EvalContext) -> Verdict:
    """Assert a source was actually dispatched to *something*.

    Keyed on ``source`` -- the pipe -- rather than on a handler name, because a
    criterion that named a descriptor would test conformance to one particular
    solution instead of to the problem. ``handler`` is accepted for the rare case
    where a problem genuinely constrains the shape of the answer, but ``source`` is
    the form to reach for.
    """
    if "source" in c.spec:
        source = PipeName(str(c.spec["source"]))
        fired = [e for e in ctx.of(VectorFired) if e.pipe == source]
        handler = source
    else:
        handler = DescriptorName(str(c.spec["handler"]))
        fired = [e for e in ctx.of(VectorFired) if e.handler == handler]
    expected = c.spec.get("times")
    if expected is not None:
        return c._verdict(  # pyright: ignore[reportPrivateUsage]
            len(fired) == int(expected),
            f"{handler} dispatched {len(fired)} time(s), expected {expected}",
        )
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        bool(fired),
        f"{handler} dispatched {len(fired)} time(s)"
        + (f" (to {sorted({str(e.handler) for e in fired})})" if fired else ""),
    )


def _max_jobs(c: Criterion, ctx: EvalContext) -> Verdict:
    """A guard against solutions that pass by brute force -- spawning a handler per
    event rather than coalescing."""
    limit = int(c.spec["limit"])
    spawned = len(ctx.of(JobSpawned))
    return c._verdict(  # pyright: ignore[reportPrivateUsage]
        spawned <= limit, f"{spawned} job(s) spawned against a limit of {limit}"
    )


_HANDLERS: Mapping[str, Any] = {
    "world_equals": _world_equals,
    "latency": _latency,
    "no_fault": _no_fault,
    "fault_expected": _fault_expected,
    "job_completed": _job_completed,
    "pipe_written": _pipe_written,
    "never_written_above_integrity": _never_written_above_integrity,
    "preempted": _preempted,
    "preemption_after": _preemption_after,
    "resumed_dirty_naming": _resumed_dirty_naming,
    "vector_fired": _vector_fired,
    "max_jobs": _max_jobs,
}
