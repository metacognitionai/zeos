# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A command the seat cannot honour is a ``malformed_request`` fault, not silence."""

from __future__ import annotations

from collections.abc import Sequence

from zeos.core.events import Decoded, Event, FaultRaised, Injected, PipeWritten
from zeos.core.ids import DescriptorName, FaultKind, JobState, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineBackend, MachineRequest, OpKind
from zeos.machine.scripted import Script, ScriptedMachine, Step
from zeos.machine.seat import CommandSeat, Turn
from zeos.world.store import WorldStore

TALKER = DescriptorName("talker")
REPORT = PipeName("ops.report")


class Tape:
    def __init__(self, commands: Sequence[str]) -> None:
        self.commands = list(commands)

    def next_command(self, turn: Turn) -> str:
        return self.commands[turn.issued]


def _run(machine: MachineBackend, **frontmatter: object) -> tuple[list[Event], JobState]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            TALKER: Descriptor.from_frontmatter(
                {"name": str(TALKER), "priority": 50, "pipes": {"stdout": str(REPORT)}}
                | frontmatter,
                body="t",
            )
        },
        machine=machine,
        pipes=PipeTable([PipeSpec(REPORT)]),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="unknown-command", max_ticks=50),
    )
    kernel.start()
    job = kernel.spawn(TALKER)
    kernel.run_until_quiescent()
    return events, job.state


def _spoken(events: Sequence[Event]) -> str:
    return "".join(t for e in events if isinstance(e, Decoded) for t in e.text)


def _faults(events: Sequence[Event]) -> list[FaultRaised]:
    return [e for e in events if isinstance(e, FaultRaised)]


def _notices(events: Sequence[Event]) -> list[str]:
    return [" ".join(e.text) for e in events if isinstance(e, Injected) and "<FAULT" in e.text[0]]


def test_an_unknown_verb_is_a_malformed_fault_naming_the_words() -> None:
    events, state = _run(CommandSeat(source=Tape(["shove stdout hello;", "exit;"])))

    assert "shove stdout hello;" in _spoken(events), "the words were decoded as spoken"
    faults = _faults(events)
    assert [f.fault for f in faults] == [FaultKind.MALFORMED]
    assert "'shove stdout hello'" in faults[0].detail
    assert state is JobState.FAULTED, "the default policy escalates"


def test_a_call_missing_its_pipe_is_a_malformed_fault() -> None:
    events, _ = _run(CommandSeat(source=Tape(["write;", "exit;"])))

    assert [f.fault for f in _faults(events)] == [FaultKind.MALFORMED]
    assert "'write'" in _faults(events)[0].detail


def test_under_retry_the_job_is_told_and_carries_on() -> None:
    """Under ``retry`` the notice lands in the job's context and the job carries on."""
    events, state = _run(
        CommandSeat(source=Tape(["shove stdout hello;", "write stdout hello;", "exit;"])),
        on_fault="retry",
    )

    notices = _notices(events)
    assert len(notices) == 1 and "malformed_request" in notices[0]
    assert "'shove stdout hello'" in notices[0]
    written = [e for e in events if isinstance(e, PipeWritten) and e.pipe == REPORT]
    assert [" ".join(w.text) for w in written] == ["hello"], "the job went on to do its work"
    assert state is JobState.DONE


def test_a_declared_verb_that_asks_for_nothing_is_not_a_fault() -> None:
    events, state = _run(CommandSeat(source=Tape(["say 41;", "exit;"])))

    assert not _faults(events) and state is JobState.DONE


def test_the_kernel_still_faults_a_pipeless_write_it_is_handed_directly() -> None:
    """The neighbouring path, unchanged: a machine handing the kernel a write with no pipe."""
    script = Script(
        steps=(
            Step(emit="write", request=MachineRequest(op=OpKind.WRITE)),
            Step(request=MachineRequest(op=OpKind.EXIT)),
        )
    )
    events, _ = _run(ScriptedMachine({str(TALKER): script}, block_size=8))

    faults = _faults(events)
    assert faults and "names no pipe" in faults[0].detail


def test_a_command_the_abi_knows_leaves_its_effect() -> None:
    events, state = _run(CommandSeat(source=Tape(["write stdout hello;", "exit;"])))

    written = [e for e in events if isinstance(e, PipeWritten) and e.pipe == REPORT]
    assert [" ".join(w.text) for w in written] == ["hello"]
    assert not _faults(events) and state is JobState.DONE
