# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A spawn a model asks for reaches the child, and is refused outside ``children:``.

``children:`` is the declaration of what a job may dispatch, and the kernel refuses
anything else as a capability fault. Both halves of that need a target to resolve, so
they are only actually enforced on a command the seat parsed if the name survives the
parse -- which is what these tests hold it to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, FaultRaised, JobSpawned
from zeos.core.ids import DescriptorName, FaultKind, JobState, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.abi import SyscallABI, Verb
from zeos.machine.base import OpKind
from zeos.machine.seat import CommandSeat, Turn
from zeos.world.store import WorldStore

#: A vocabulary with a spawn in it. The default has none, so nothing else in the tree
#: exercises the path a parsed command takes to ``Kernel.spawn``.
WITH_SPAWN = SyscallABI(
    verbs=(
        Verb("say", text=True, doc="think out loud"),
        Verb("write", OpKind.WRITE, pipe=True, text=True, doc="put text on a pipe"),
        Verb("spawn", OpKind.SPAWN, text=True, doc="start a declared child"),
        Verb("exit", OpKind.EXIT, doc="finish"),
    ),
)

OUT = PipeName("ops.report")


class Tapes:
    """One command list per descriptor, played in order; a stand-in for a model."""

    def __init__(self, by: Mapping[str, Sequence[str]]) -> None:
        self.by = {k: list(v) for k, v in by.items()}

    def next_command(self, turn: Turn) -> str:
        return self.by[turn.descriptor][turn.issued]


def _run(parent: Mapping[str, Any], commands: Sequence[str]) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    frontmatter: list[Mapping[str, Any]] = [
        parent,
        {"name": "child", "priority": 70, "pipes": {"stdout": str(OUT)}},
    ]
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d, body="b")
            for d in frontmatter
        },
        machine=CommandSeat(
            source=Tapes({"parent": list(commands), "child": ["write stdout done;", "exit;"]}),
            abi=WITH_SPAWN,
        ),
        pipes=PipeTable([PipeSpec(OUT, sink=True)]),
        vectors=VectorTable([]),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="spawn-abi", max_ticks=400),
    )
    kernel.start()
    kernel.spawn(DescriptorName("parent"))
    kernel.run_until_quiescent()
    return kernel, events


PARENT: dict[str, Any] = {"name": "parent", "priority": 50, "children": ["child"]}


def test_a_spawn_a_model_asked_for_starts_the_child() -> None:
    kernel, events = _run(PARENT, ["spawn child;", "exit;"])
    spawned = [str(e.descriptor) for e in events if isinstance(e, JobSpawned)]
    assert spawned == ["parent", "child"]
    assert not [e for e in events if isinstance(e, FaultRaised)]
    assert [j.state for j in kernel.sched.jobs()] == [JobState.DONE, JobState.DONE]


def test_the_child_is_this_jobs_child() -> None:
    kernel, _ = _run(PARENT, ["spawn child;", "exit;"])
    parent, child = kernel.sched.jobs()
    assert child.parent == parent.job_id
    assert parent.children == [child.job_id]


def test_a_spawn_outside_children_is_a_capability_fault() -> None:
    """Authority is checked against the declaration, not against what the model asked for."""
    _, events = _run({"name": "parent", "priority": 50}, ["spawn child;", "exit;"])
    assert [str(e.descriptor) for e in events if isinstance(e, JobSpawned)] == ["parent"]
    assert [e.fault for e in events if isinstance(e, FaultRaised)] == [FaultKind.CAPABILITY]


def test_a_spawn_naming_nothing_is_a_malformed_request() -> None:
    """A closed command the kernel cannot act on is a fault class, not a silent no-op."""
    _, events = _run(PARENT, ["spawn;", "exit;"])
    assert [str(e.descriptor) for e in events if isinstance(e, JobSpawned)] == ["parent"]
    assert [e.fault for e in events if isinstance(e, FaultRaised)] == [FaultKind.MALFORMED]
