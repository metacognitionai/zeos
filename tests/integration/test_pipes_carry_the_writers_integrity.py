# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A trusted pipe does not launder a dirtier writer's words: the reader receives them
at the worse of the pipe's ring and the writer's integrity.

A pipe remembers how clean each write's writer was. A write through a schema is the one
endorsement and crosses at the pipe's ring; without one, content from a job at integrity 3
arrives at ring 3 however trusted the pipe, and the lint warns about a capability that
declares such a channel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.capabilities import Schema
from zeos.core.events import (
    Endorsed,
    Event,
    FaultRaised,
    Injected,
)
from zeos.core.ids import (
    DescriptorName,
    ObjectName,
    PipeName,
    Ring,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineBackend
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.machine.seat import Turn
from zeos.world.store import WorldStore


class Tapes:
    """One command list per descriptor, played in order; a stand-in for a model."""

    def __init__(self, by: Mapping[str, Sequence[str]]) -> None:
        self.by = {k: list(v) for k, v in by.items()}

    def next_command(self, turn: Turn) -> str:
        return self.by[turn.descriptor][turn.issued]


def _build(
    descriptors: Sequence[Mapping[str, Any]],
    machine: MachineBackend,
    pipes: Sequence[PipeSpec] = (),
    *,
    vectors: Sequence[VectorSpec] = (),
    world: Mapping[str, str] | None = None,
    max_ticks: int = 400,
    schemas: Mapping[str, Schema] | None = None,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(
                d, body="b", schemas=schemas
            )
            for d in descriptors
        },
        machine=machine,
        pipes=PipeTable(pipes),
        vectors=VectorTable(list(vectors)),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="repro", max_ticks=max_ticks),
    )
    for obj, value in (world or {}).items():
        kernel.world.set(ObjectName(obj), value, at=kernel.clock)
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


HAND = PipeName("hand")
VERDICT = Schema.parse("verdict", {"value": "enum(0, 60)"})


def _run(ring: Ring, *, schema: bool = False, payload: str = "0") -> tuple[Kernel, list[Event]]:
    capability: dict[str, Any] = {"pipe": str(HAND), "min_integrity": 3}
    if schema:
        capability["schema"] = "verdict"
    kernel, events = _build(
        [
            {
                "name": "dirty",
                "priority": 60,
                "integrity": {"start": 3},
                "pipes": {"stdout": str(HAND)},
                "capabilities": [capability],
            },
            {"name": "clean", "priority": 50, "pipes": {"stdin": str(HAND)}},
        ],
        ScriptedMachine(
            {
                "dirty": Script.from_spec(
                    [{"write": {"pipe": str(HAND), "text": payload}}, {"exit": True}]
                ),
                "clean": Script.from_spec([{"read": str(HAND)}, {"emit": "ok"}, {"exit": True}]),
            },
            block_size=8,
        ),
        [PipeSpec(HAND, ring=ring)],
        schemas={"verdict": VERDICT},
    )
    kernel.spawn(DescriptorName("clean"))
    kernel.run_until_quiescent()
    kernel.spawn(DescriptorName("dirty"))
    kernel.run_until_quiescent()
    return kernel, events


def _arrival(kernel: Kernel, events: Sequence[Event]) -> Injected:
    clean = next(j for j in kernel.sched.jobs() if str(j.descriptor.name) == "clean")
    return [e for e in _of(events, Injected) if e.job == clean.job_id and e.pipe == HAND][0]


def test_a_ring_3_writers_words_arrive_at_ring_3_over_a_trusted_pipe() -> None:
    kernel, events = _run(Ring.TRUSTED)
    arrival = _arrival(kernel, events)
    assert arrival.ring is Ring.EXTERNAL and int(arrival.integrity) == 3
    assert not _of(events, FaultRaised), "the write landed; only its provenance changed"


def test_through_a_schema_the_same_write_is_an_endorsement_and_arrives_at_the_pipes_ring() -> None:
    kernel, events = _run(Ring.TRUSTED, schema=True, payload="value=60")
    assert _arrival(kernel, events).ring is Ring.TRUSTED
    assert [str(e.endorser) for e in _of(events, Endorsed)] == ["dirty"]


def test_a_clean_writers_words_still_arrive_at_the_pipes_ring() -> None:
    kernel, events = _build(
        [
            {"name": "w", "priority": 60, "pipes": {"stdout": str(HAND)}},
            {"name": "clean", "priority": 50, "pipes": {"stdin": str(HAND)}},
        ],
        ScriptedMachine(
            {
                "w": Script.from_spec(
                    [{"write": {"pipe": str(HAND), "text": "0"}}, {"exit": True}]
                ),
                "clean": Script.from_spec([{"read": str(HAND)}, {"emit": "ok"}, {"exit": True}]),
            },
            block_size=8,
        ),
        [PipeSpec(HAND, ring=Ring.TRUSTED)],
    )
    kernel.spawn(DescriptorName("clean"))
    kernel.run_until_quiescent()
    kernel.spawn(DescriptorName("w"))
    kernel.run_until_quiescent()
    assert _arrival(kernel, events).ring is Ring.TRUSTED


def test_the_lint_warns_about_a_schema_less_write_up_capability() -> None:
    from zeos.descriptor.lint import Severity, lint

    d = Descriptor.from_frontmatter(
        {
            "name": "dirty",
            "priority": 60,
            "integrity": {"start": 3},
            "pipes": {"stdout": str(HAND)},
            "capabilities": [{"pipe": str(HAND), "min_integrity": 3}],
        },
        body="b",
    )
    trusted = [
        f
        for f in lint({d.name: d}, pipes=[PipeSpec(HAND, ring=Ring.TRUSTED)])
        if f.rule == "write-up-without-schema"
    ]
    assert [f.severity for f in trusted] == [Severity.WARNING]
    external = [
        f
        for f in lint({d.name: d}, pipes=[PipeSpec(HAND, ring=Ring.EXTERNAL)])
        if f.rule == "write-up-without-schema"
    ]
    assert not external


def test_over_an_external_pipe_the_same_words_arrive_at_ring_3() -> None:
    kernel, events = _run(Ring.EXTERNAL)
    assert _arrival(kernel, events).ring is Ring.EXTERNAL
