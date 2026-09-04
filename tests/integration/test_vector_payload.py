# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Firing a vector consumes the payload that caused it.

Firing is edge-triggered off the write call, and ``VectorTable.on_write`` decides from
the vector's own state without ever consulting the pipe. Nothing consumed the payload
afterwards either: a handler is a fresh job reading whatever pipes its own descriptor
binds, and a bare-signal source binds no reader at all. So the buffer filled to capacity
and stayed there, and every write past that was dropped and journalled as
``PipeWritten(tokens=0)`` -- indistinguishable at a glance from one that landed.

``Transformer-OS.md`` already specifies the other half: the lifecycle diagram (:321)
and the latency budget (:206, :219) both price handling an interrupt as the prefill of
its event payload, which presumes the payload reaches the handler's context.
"""

from __future__ import annotations

from zeos.core.events import Event, Injected, PipeWritten, VectorFired
from zeos.core.ids import DescriptorName, PipeName, Priority, VectorName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.segments import TAG_DESCRIPTOR
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

KEYS = PipeName("keys.interrupt")
CAPACITY = 8


def _build() -> tuple[Kernel, ScriptedMachine, list[Event]]:
    """A pure-signal source: bound to a vector, and bound by no descriptor."""
    descriptor = Descriptor.from_frontmatter({"name": "handler", "priority": 5}, body="ack it")
    machine = ScriptedMachine(
        {"handler": Script.from_spec([{"emit": "ack"}, {"exit": True}])}, block_size=16
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("handler"): descriptor},
        machine=machine,
        pipes=PipeTable([PipeSpec(name=KEYS, device=True, capacity_tokens=CAPACITY)]),
        vectors=VectorTable(
            [
                VectorSpec(
                    name=VectorName("irq"),
                    source=KEYS,
                    handler=DescriptorName("handler"),
                    priority=Priority(5),
                )
            ]
        ),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="vector-payload", max_ticks=500),
    )
    kernel.start()
    return kernel, machine, events


def _press(kernel: Kernel, events: list[Event], count: int) -> list[int]:
    """Deliver ``count`` signals, running each firing out, and report tokens accepted."""
    accepted: list[int] = []
    for _ in range(count):
        before = len(events)
        kernel.deliver(KEYS, "x")
        accepted += [
            e.tokens for e in events[before:] if isinstance(e, PipeWritten) and e.pipe == KEYS
        ]
        for _ in range(6):
            if not kernel.tick():
                break
    return accepted


def test_the_source_pipe_does_not_fill_across_repeated_firings() -> None:
    """The leak. Twelve signals through a capacity-eight pipe used to drop the last four."""
    kernel, _, events = _build()

    accepted = _press(kernel, events, CAPACITY + 4)

    assert accepted == [1] * (CAPACITY + 4), "every signal must land; a dropped one is silent"
    assert kernel.pipes.get(KEYS).available == 0, "the buffer is drained, not latched"


def test_the_payload_reaches_the_handler_behind_its_own_body() -> None:
    """Ordering matters as much as delivery: the code comes before the event.

    The payload is taken when the vector fires but injected at ``_start_job``, because
    a handler spawned and injected in one step would read the event above the
    instructions that explain it.
    """
    kernel, machine, _ = _build()

    kernel.deliver(KEYS, "level 42")
    for _ in range(6):
        if not kernel.tick():
            break

    handler = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("handler"))
    body = next(r for r in handler.segments.all() if r.provenance.tag == TAG_DESCRIPTOR)
    payload = next(r for r in handler.segments.all() if r.provenance.tag == str(KEYS))
    assert body.end <= payload.start, "the handler's code precedes the event it handles"
    text = " ".join(t.text for t in machine.transcript(handler.job_id)[payload.start : payload.end])
    assert text == "level 42"


def test_the_payload_keeps_the_source_pipe_s_provenance() -> None:
    """Injected as device input, not as kernel content.

    Laundering a source pipe's content to kernel ring on the way in would hand
    imperative standing to whatever wrote the pipe.
    """
    kernel, _, events = _build()

    kernel.deliver(KEYS, "press")
    for _ in range(6):
        if not kernel.tick():
            break

    injected = [e for e in events if isinstance(e, Injected) and e.pipe == KEYS]
    assert injected, "the payload was never injected"
    assert injected[0].ring is kernel.pipes.get(KEYS).spec.ring
    assert [e for e in events if isinstance(e, VectorFired)], "the vector never fired"
