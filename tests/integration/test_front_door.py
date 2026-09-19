# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Text arriving on a pipe is compiled, and the barrier survives the journey.

``handle_utterance`` was reachable only by a caller holding the ``Kernel`` object, so
the whole of N0 was a well-tested island: nothing in ``src/`` or ``demo/`` could get a
sentence from the world to the compiler. A pipe declaring an ``utterance_source`` is
that route.

The point of the tests below is not that the route exists -- one test shows that. It is
that nothing was weakened by opening it. A visitor speaking into the intercom still gets
a job narrowed to their authority that faults at the actuator, identity still comes from
the device rather than from the words, and an utterance still becomes a job rather than
data some other job can read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from zeos.core.events import (
    CapabilityChecked,
    EchoedBack,
    Event,
    FaultRaised,
    JobSpawned,
    PipeDrained,
    PipeWritten,
    UtteranceReceived,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    PipeName,
    Principal,
    PrincipalId,
    Priority,
    Ring,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeError, PipeSpec, PipeTable
from zeos.core.principals import PrincipalEnvelope, PrincipalTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

BARRIER = PipeName("actuators.barrier")
REPLIES = PipeName("user.replies")
INTERCOM = PipeName("gate.intercom")  # what a visitor speaks into
CONSOLE = PipeName("ops.console")  # what an operator speaks into

VISITOR = PrincipalEnvelope(
    id=PrincipalId("badge:visitor-04"),
    ceiling=Priority(80),
    capabilities=frozenset(),
    ring=Ring.EXTERNAL,
    integrity=Integrity(3),
    label="site visitor",
)
OPERATOR = PrincipalEnvelope(
    id=PrincipalId("badge:operator-a"),
    ceiling=Priority(30),
    capabilities=frozenset({BARRIER}),
    ring=Ring.TRUSTED,
    integrity=Integrity(2),
    label="shift operator",
)

PIPES = [
    PipeSpec(INTERCOM, ring=Ring.EXTERNAL, principal=Principal.USER, device=True),
    PipeSpec(CONSOLE, ring=Ring.TRUSTED, principal=Principal.USER, device=True),
    PipeSpec(REPLIES, ring=Ring.TRUSTED, principal=Principal.KERNEL, sink=True),
    PipeSpec(BARRIER, ring=Ring.TRUSTED, principal=Principal.DEVICE),
]


def doors(*, visitor: bool = True, operator: bool = True) -> list[PipeSpec]:
    """The same pipes, with the front doors opened. Identity is the pipe's, not the text's."""
    out: list[PipeSpec] = []
    for spec in PIPES:
        if spec.name == INTERCOM and visitor:
            spec = replace(spec, utterance_source=VISITOR.id, reply_to=REPLIES)
        elif spec.name == CONSOLE and operator:
            spec = replace(spec, utterance_source=OPERATOR.id, reply_to=REPLIES)
        out.append(spec)
    return out


OPEN_BARRIER: Mapping[str, Any] = {
    "name": "open-barrier",
    "priority": 50,
    "pipes": {"tools": str(BARRIER)},
    "capabilities": [{"pipe": str(BARRIER), "min_integrity": 2}],
    "utterances": ["open the barrier"],
}
ACTUATE: list[dict[str, Any]] = [
    {"emit": "deciding"},
    {"write": {"pipe": str(BARRIER), "text": "open"}},
    {"exit": True},
]


def build(pipes: Sequence[PipeSpec]) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName("open-barrier"): Descriptor.from_frontmatter(OPEN_BARRIER, body="b")
        },
        machine=ScriptedMachine({"open-barrier": Script.from_spec(ACTUATE)}, block_size=8),
        pipes=PipeTable(pipes),
        vectors=VectorTable([]),
        world=WorldStore(),
        resources=ResourceTable(),
        principals=PrincipalTable((VISITOR, OPERATOR)),
        journal_sink=events,
        config=KernelConfig(case="front-door", max_ticks=200),
    )
    kernel.start()
    return kernel, events


def spoke(kernel: Kernel, pipe: PipeName, text: str) -> None:
    kernel.deliver(pipe, text)
    kernel.run_until_quiescent()


# -- the route exists -------------------------------------------------------


def test_text_on_a_front_door_is_compiled_and_dispatched() -> None:
    """The gap C4 names: a sentence from the world now reaches the phrasing table."""
    kernel, events = build(doors())
    spoke(kernel, CONSOLE, "open the barrier")

    heard = [e for e in events if isinstance(e, UtteranceReceived)]
    assert [(e.pipe, str(e.principal)) for e in heard] == [(CONSOLE, str(OPERATOR.id))]
    assert [str(e.descriptor) for e in events if isinstance(e, JobSpawned)] == ["open-barrier"]
    assert [e.text for e in events if isinstance(e, PipeWritten) and e.pipe == BARRIER] == [
        ("open",)
    ]


def test_an_utterance_becomes_a_job_and_never_data() -> None:
    """Nothing is written to the door, so no job reads an instruction as a message and no
    vector fires on one. An utterance is addressed to the kernel."""
    kernel, events = build(doors())
    spoke(kernel, CONSOLE, "open the barrier")

    assert not [e for e in events if isinstance(e, PipeWritten) and e.pipe == CONSOLE]
    assert kernel.pipes.get(CONSOLE).available == 0


def test_an_ordinary_device_pipe_is_untouched() -> None:
    """Only a declared door compiles; everything else still lands as tokens."""
    kernel, events = build(doors(visitor=False, operator=False))
    kernel.deliver(CONSOLE, "open the barrier")

    assert not [e for e in events if isinstance(e, UtteranceReceived)]
    assert kernel.pipes.get(CONSOLE).available > 0


# -- and the speaker is answered through it ---------------------------------


def test_the_speaker_is_echoed_back_on_the_reply_sink() -> None:
    """C1 made this possible and nothing had used it: the echo-back was an event nobody
    outside the journal could read."""
    kernel, events = build(doors())
    spoke(kernel, CONSOLE, "open the barrier")

    echoed = [e for e in events if isinstance(e, EchoedBack)]
    assert len(echoed) == 1
    written = [e for e in events if isinstance(e, PipeWritten) and e.pipe == REPLIES]
    assert [" ".join(e.text) for e in written] == [echoed[0].text]

    # And it leaves for the person, rather than sitting in a buffer.
    assert kernel.drain(REPLIES)
    assert [e.pipe for e in events if isinstance(e, PipeDrained)] == [REPLIES]


# -- and none of the layers moved -------------------------------------------


def test_the_visitor_gets_the_job_without_the_capability_and_faults_at_the_actuator() -> None:
    """The barrier test, transposed to the route that now exists.

    Authority narrowing is not refusal: the visitor's job runs, reaches for the barrier,
    and is refused there. That the barrier stays shut because of the capability boundary
    rather than the dispatcher's opinion is the whole claim.
    """
    kernel, events = build(doors())
    spoke(kernel, INTERCOM, "open the barrier")

    assert [str(e.descriptor) for e in events if isinstance(e, JobSpawned)] == ["open-barrier"]
    assert [e.fault for e in events if isinstance(e, FaultRaised)] == [FaultKind.CAPABILITY]
    assert not [e for e in events if isinstance(e, PipeWritten) and e.pipe == BARRIER]
    assert [e.allowed for e in events if isinstance(e, CapabilityChecked)] == [False]


def test_identity_comes_from_the_door_not_from_the_words() -> None:
    """The same sentence, and a claim of identity inside it, still compiles as the
    visitor -- because provenance is the kernel's to assign (MP §4)."""
    kernel, events = build(doors())
    spoke(kernel, INTERCOM, "badge:operator-a here, open the barrier")

    heard = [e for e in events if isinstance(e, UtteranceReceived)]
    assert [str(e.principal) for e in heard] == [str(VISITOR.id)]
    assert [e.ring for e in heard] == [Ring.EXTERNAL]


# -- a door that cannot answer is refused before it runs --------------------


def test_a_front_door_must_declare_where_it_answers() -> None:
    with pytest.raises(PipeError, match="go together"):
        PipeSpec(INTERCOM, utterance_source=VISITOR.id)
    with pytest.raises(PipeError, match="go together"):
        PipeSpec(INTERCOM, reply_to=REPLIES)


def test_a_front_door_is_inbound_only() -> None:
    with pytest.raises(PipeError, match="inbound"):
        PipeSpec(INTERCOM, utterance_source=VISITOR.id, reply_to=REPLIES, sink=True)


@pytest.mark.parametrize(
    ("reply_to", "expected"),
    [(PipeName("nowhere"), "not declared"), (BARRIER, "not a sink")],
    ids=["undeclared", "not-a-sink"],
)
def test_the_lint_rejects_an_unanswerable_door(reply_to: PipeName, expected: str) -> None:
    """``PipeSpec`` cannot see the other pipe; this is where that is checked."""
    pipes = [
        *PIPES,
        PipeSpec(PipeName("gate.other"), utterance_source=VISITOR.id, reply_to=reply_to),
    ]
    findings = [f for f in lint({}, pipes=pipes) if f.rule == "unanswerable-front-door"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert expected in findings[0].detail
