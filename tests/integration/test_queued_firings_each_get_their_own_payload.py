# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Queue policy: every queued firing's handler is handed the write that fired it.

N writes to the source pipe while the handler is busy are N pending firings, and a pipe
remembers where each write ended, so each handler's payload is its own write and the
writes behind it wait for the firings still pending. A job's ordinary read still takes
everything in the pipe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import (
    Event,
    JobCompleted,
    JobSpawned,
    PipeReadEvent,
)
from zeos.core.ids import (
    DescriptorName,
    ObjectName,
    PipeName,
    Principal,
    Priority,
    VectorName,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorPolicy, VectorSpec, VectorTable
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
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d, body="b")
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


KEYS = PipeName("keys")
NUMBERS = PipeName("keys.number")
VECTOR = VectorSpec(
    name=VectorName("v"),
    source=KEYS,
    handler=DescriptorName("h"),
    priority=Priority(5),
    policy=VectorPolicy.QUEUE,
)


def _kernel(handler_steps: int = 1) -> tuple[Kernel, list[Event]]:
    return _build(
        [
            {"name": "g", "priority": 50},
            {"name": "h", "priority": 5, "pipes": {"stdin": str(KEYS)}},
        ],
        ScriptedMachine(
            {
                "g": Script.from_spec([{"emit": "w"}] * 3 + [{"exit": True}]),
                "h": Script.from_spec([{"emit": "set"}] * handler_steps + [{"exit": True}]),
            },
            block_size=8,
        ),
        [PipeSpec(KEYS, device=True, principal=Principal.USER, capacity_tokens=16)],
        vectors=[VECTOR],
    )


def _handlers(events: Sequence[Event]) -> list[Any]:
    return [e.job for e in _of(events, JobSpawned) if str(e.descriptor) == "h"]


def _payloads(events: Sequence[Event]) -> list[tuple[str, ...]]:
    return [e.text for e in _of(events, PipeReadEvent) if e.pipe == KEYS]


def test_five_keypresses_in_one_tick_are_five_handlers_each_with_its_own_payload() -> None:
    kernel, events = _kernel()
    kernel.spawn(DescriptorName("g"))
    for k in range(5):
        kernel.deliver(KEYS, f"x {100 + k}")
    kernel.run_until_quiescent()

    handlers = _handlers(events)
    assert len(handlers) == 5
    completed = {e.job for e in _of(events, JobCompleted)}
    assert all(h in completed for h in handlers), "no handler starved"
    assert _payloads(events) == [("x", str(100 + k)) for k in range(5)], "each its own, in order"


def test_a_keypress_landing_while_a_handler_runs_waits_for_its_own_firing() -> None:
    kernel, events = _kernel(handler_steps=4)
    kernel.spawn(DescriptorName("g"))
    kernel.deliver(KEYS, "x 100")
    kernel.tick()
    kernel.deliver(KEYS, "x 101")
    kernel.tick()
    kernel.deliver(KEYS, "x 102")
    kernel.run_until_quiescent()

    assert len(_handlers(events)) == 3
    assert _payloads(events) == [("x", "100"), ("x", "101"), ("x", "102")]


def test_five_keypresses_spaced_out_are_five_handlers_each_with_its_own_payload() -> None:
    kernel, events = _kernel()
    kernel.spawn(DescriptorName("g"))
    for k in range(5):
        kernel.deliver(KEYS, f"x {100 + k}")
        kernel.run_until_quiescent()

    assert len(_handlers(events)) == 5
    assert _payloads(events) == [("x", str(100 + k)) for k in range(5)]


def test_a_jobs_read_still_takes_everything_in_the_pipe() -> None:
    inbox = PipeName("inbox")
    kernel, events = _build(
        [{"name": "r", "priority": 50, "pipes": {"stdin": str(inbox)}}],
        ScriptedMachine(
            {"r": Script.from_spec([{"read": str(inbox)}, {"emit": "ok"}, {"exit": True}])},
            block_size=8,
        ),
        [PipeSpec(inbox, device=True, capacity_tokens=16)],
    )
    kernel.spawn(DescriptorName("r"))
    kernel.run_until_quiescent()
    kernel.deliver(inbox, "one")
    kernel.deliver(inbox, "two three")
    kernel.run_until_quiescent()
    assert [e.tokens for e in _of(events, PipeReadEvent) if e.pipe == inbox] == [3]
