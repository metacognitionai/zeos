# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Testing one behaviour on its own.

The unit smaller than the whole system exists again, and this is its harness:
run a single descriptor against scripted pipes, assert on
its pipe writes. Deterministic given a seed, cheap enough for CI, and needing no
other behaviour, no world, and no GPU -- the gas handler is tested by firing
synthetic gas events at it.

This lives in ``src/`` rather than in ``tests/`` because solution authors need it.
A behaviour is a unit of *authorship*, so it has to be a unit of testing too, and
that is only true if the harness ships with the runtime rather than with our
test suite.

Contract-tier fuzzing is the same harness with adversarial inputs: feed a descriptor
an injection corpus on a ring-3 pipe and assert the MP fault behaviour, rather than
reading the transcript and forming an impression.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.events import (
    CapabilityChecked,
    Event,
    FaultRaised,
    IntegrityDemoted,
    PipeWritten,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    ObjectName,
    PipeName,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import render
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

__all__ = ["BehaviourRun", "run_behaviour"]


@dataclass(frozen=True, slots=True)
class BehaviourRun:
    """What one behaviour did, in the terms its contract is written in."""

    events: tuple[Event, ...]
    writes: Mapping[PipeName, tuple[str, ...]]
    faults: tuple[FaultKind, ...]
    final_integrity: Integrity
    world: Mapping[ObjectName, str]
    transcript: str
    completed: bool

    def wrote(self, pipe: PipeName) -> tuple[str, ...]:
        return self.writes.get(pipe, ())

    def faulted_with(self, kind: FaultKind) -> bool:
        return kind in self.faults

    def of[E: Event](self, cls: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, cls)]

    def demotions(self) -> list[IntegrityDemoted]:
        return self.of(IntegrityDemoted)

    def blocked_writes(self) -> list[CapabilityChecked]:
        return [c for c in self.of(CapabilityChecked) if not c.allowed]


def run_behaviour(
    descriptor: Descriptor,
    script: Script,
    *,
    pipes: Sequence[PipeSpec] = (),
    inputs: Mapping[str, str] | None = None,
    world: Mapping[str, str] | None = None,
    block_size: int = 8,
    max_ticks: int = 500,
) -> BehaviourRun:
    """Run one descriptor to completion against pre-filled pipes.

    ``inputs`` pre-loads pipes with content the behaviour will read, which stands in
    for whatever would have written them -- a device, a peer job, or an attacker. The
    behaviour cannot tell the difference, which is exactly why testing it in
    isolation is meaningful.
    """
    events: list[Event] = []
    table = PipeTable(pipes)
    store = WorldStore()

    kernel = Kernel(
        descriptors={descriptor.name: descriptor},
        machine=ScriptedMachine({str(descriptor.name): script}, block_size=block_size),
        pipes=table,
        vectors=VectorTable(),
        world=store,
        journal_sink=events,
        config=KernelConfig(case=f"unit:{descriptor.name}", max_ticks=max_ticks),
    )
    kernel.start()
    for obj, value in sorted((world or {}).items()):
        store.set(ObjectName(obj), value, at=kernel.clock)

    job = kernel.spawn(DescriptorName(descriptor.name))
    for pipe_name, text in sorted((inputs or {}).items()):
        kernel.deliver(PipeName(pipe_name), text)
    kernel.run_until_quiescent()

    writes: dict[PipeName, list[str]] = {}
    for event in events:
        if isinstance(event, PipeWritten) and event.job is not None:
            writes.setdefault(event.pipe, [])
    # Recover payload text from the pipes themselves: PipeWritten records a count,
    # and what a reviewer wants to see is what was said.
    for pipe in table.all():
        if pipe.peek():
            writes.setdefault(pipe.name, []).append(render(pipe.peek()))

    return BehaviourRun(
        events=tuple(events),
        writes={k: tuple(v) for k, v in writes.items()},
        faults=tuple(e.fault for e in events if isinstance(e, FaultRaised)),
        final_integrity=job.current_integrity,
        world=dict(store.snapshot()),
        transcript=render(kernel.machine.transcript(job.job_id)),
        completed=job.state.is_terminal,
    )


_ = field  # reserved for future run options
