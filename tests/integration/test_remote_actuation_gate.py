# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""An actuation arriving over the link faces the receiving node's gate.

Anything that lands on a guarded pipe from outside -- a frame from the peer, a
device adapter -- goes through ``Kernel.deliver``, which now asks the guard as
``_do_write`` does for a local job. There is no local job to hold the command, so
the kernel holds it, keyed by the guard it spawned, and performs or drops the
delivery when the verdict lands. The gate events record ``job=None`` for such a
command, as ``PipeWritten`` already does for deliveries.

It used not to. ``deliver`` wrote the pipe, applied the world object and fired
vectors, and consulted no gate, so a guard declared on the receiving node was
bypassed by any command that crossed the link. Issue #20.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, FrameDelivered, GateAnswered, GateConsulted, PipeWritten
from zeos.core.gates import ALLOW, VETO, GateSpec, GateTable
from zeos.core.ids import DescriptorName, PipeName, Principal, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.topology import LinkSpec, NodeSpec, Topology
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.federated import Federation, Node
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.transport.link import LinkTransport, LossModel
from zeos.world.store import ObjectSet, WorldStore

MS = 1_000_000
BARRIER = PipeName("actuators.barrier")
REQUESTS = PipeName("gates.walkway.requests")
VERDICTS = PipeName("gates.walkway.verdicts")

PIPES = [
    PipeSpec(BARRIER, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
    PipeSpec(REQUESTS, ring=Ring.KERNEL, principal=Principal.KERNEL),
    PipeSpec(VERDICTS, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
]
TOPOLOGY = Topology(
    [
        NodeSpec("platform", "small-local", ObjectSet.of(["robot.*"])),
        NodeSpec("offboard", "large", ObjectSet.of(["mission.*"])),
    ],
    LinkSpec(rtt_p99_ns=120 * MS, ring=Ring.TRUSTED),
)

GUARD: dict[str, Any] = {
    "name": "walkway-guard",
    "priority": 15,
    "pipes": {"stdin": str(REQUESTS)},
    "capabilities": [{"pipe": str(VERDICTS), "min_integrity": 2}],
}
PLANNER: dict[str, Any] = {
    "name": "planner",
    "priority": 50,
    "capabilities": [{"pipe": str(BARRIER), "min_integrity": 2}],
}
SCRIPTS = {
    "planner": [{"write": {"pipe": str(BARRIER), "text": "open the barrier"}}, {"exit": True}],
}


def _guard_script(verdict: str) -> list[dict[str, Any]]:
    return [
        {"read": str(REQUESTS)},
        {"write": {"pipe": str(VERDICTS), "text": verdict}},
        {"exit": True},
    ]


GATES = GateTable(
    [
        GateSpec(
            pipe=BARRIER,
            descriptor=DescriptorName("walkway-guard"),
            requests=REQUESTS,
            verdicts=VERDICTS,
        )
    ]
)


def _kernel(
    node: str,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    link: LinkTransport,
    events: list[Event],
    gates: GateTable | None,
    verdict: str = ALLOW,
) -> Kernel:
    scripts = {**SCRIPTS, "walkway-guard": _guard_script(verdict)}
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine(
            {n: Script.from_spec(scripts[n]) for n in (str(d["name"]) for d in descriptors)},
            block_size=8,
        ),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        node=node,
        topology=TOPOLOGY,
        link=link,
        gates=gates,
        journal_sink=events,
        config=KernelConfig(case=f"remote-gate-{node}"),
    )
    kernel.start()
    return kernel


def _federation(verdict: str = ALLOW) -> tuple[Federation, Kernel, Kernel, list[Event]]:
    """The platform guards the barrier. Only the offboard node's link carries it, so
    a platform-local write stays local and an offboard write crosses."""
    platform_events: list[Event] = []
    offboard_events: list[Event] = []
    to_offboard = LinkTransport(TOPOLOGY.link, pipes=frozenset(), loss=LossModel())
    to_platform = LinkTransport(TOPOLOGY.link, pipes=frozenset({BARRIER}), loss=LossModel())
    platform = _kernel(
        "platform",
        [GUARD, PLANNER],
        link=to_offboard,
        events=platform_events,
        gates=GATES,
        verdict=verdict,
    )
    offboard = _kernel("offboard", [PLANNER], link=to_platform, events=offboard_events, gates=None)
    fed = Federation(topology=TOPOLOGY)
    fed.add(Node("platform", platform, to_offboard))
    fed.add(Node("offboard", offboard, to_platform))
    return fed, platform, offboard, platform_events


def _run(fed: Federation) -> None:
    for _ in range(400):
        if not fed.step():
            return
    raise AssertionError("the federation never settled")


def test_a_command_from_the_peer_faces_the_receiving_node_s_gate() -> None:
    fed, _, offboard, platform_events = _federation()
    offboard.spawn(DescriptorName("planner"))
    _run(fed)

    assert [e for e in platform_events if isinstance(e, FrameDelivered)], (
        "the command never arrived"
    )
    consulted = [e for e in platform_events if isinstance(e, GateConsulted)]
    assert consulted and consulted[0].gate == DescriptorName("walkway-guard")
    assert consulted[0].job is None, "no local job asked; the kernel asked for the link"
    answered = [e for e in platform_events if isinstance(e, GateAnswered)]
    assert answered and answered[0].allowed and answered[0].job is None
    landed = [e for e in platform_events if isinstance(e, PipeWritten) and e.pipe == BARRIER]
    assert [" ".join(e.text) for e in landed] == ["open the barrier"]
    assert platform_events.index(answered[0]) < platform_events.index(landed[0]), (
        "the barrier was written before the guard answered"
    )


def test_a_vetoed_command_from_the_peer_never_reaches_the_actuator() -> None:
    fed, _, offboard, platform_events = _federation(verdict=f"{VETO}: people on the walkway")
    offboard.spawn(DescriptorName("planner"))
    _run(fed)

    answered = [e for e in platform_events if isinstance(e, GateAnswered)]
    assert answered and not answered[0].allowed and "walkway" in answered[0].reason
    assert not [e for e in platform_events if isinstance(e, PipeWritten) and e.pipe == BARRIER]


def test_a_local_command_faces_the_gate() -> None:
    """The control: the same planner on the platform is held and shown to the guard."""
    fed, platform, _, platform_events = _federation()
    platform.spawn(DescriptorName("planner"))
    _run(fed)

    consulted = [e for e in platform_events if isinstance(e, GateConsulted)]
    assert consulted and consulted[0].gate == DescriptorName("walkway-guard")
    assert [e for e in platform_events if isinstance(e, PipeWritten) and e.pipe == BARRIER]
