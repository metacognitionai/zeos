# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The seat and the ABI it speaks: what any model-backed application gets from the kernel.

The ABI is declared once as data, so these tests hold it to that: a verb the declaration
does not name asks the kernel for nothing, and a seat handed a different declaration
speaks it without knowing any words of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from zeos.core.events import Decoded, Event, PipeWritten
from zeos.core.ids import JobId, JobState, PipeName
from zeos.core.pipes import PipeSpec
from zeos.descriptor.loader import load_case
from zeos.descriptor.schema import Descriptor
from zeos.driver import Driver, build_kernel
from zeos.journal.codec import to_line
from zeos.machine.abi import DEFAULT, SyscallABI, Verb
from zeos.machine.base import OpKind
from zeos.machine.scripted import Script, ScriptExhausted
from zeos.machine.seat import (
    CommandSeat,
    SyscallParser,
    TapeSource,
    Turn,
    bound_aliases,
    seat_maps,
    words_of,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "seat"
JOB = JobId(1)

#: A vocabulary with nothing in common with the default: different words, different
#: terminator, no payload cap.
CUSTOM = SyscallABI(
    verbs=(
        Verb("note", text=True),
        Verb("send", OpKind.WRITE, pipe=True, text=True),
        Verb("wait", OpKind.READ, pipe=True),
        Verb("done", OpKind.EXIT),
    ),
    aliases=("in", "out"),
    terminator="!",
    max_text=None,
)


class Tape:
    def __init__(self, commands: Sequence[str]) -> None:
        self.commands = list(commands)
        self.turns: list[Turn] = []

    def next_command(self, turn: Turn) -> str:
        self.turns.append(turn)
        return self.commands[turn.issued]


def _seat(commands: Sequence[str], abi: SyscallABI = DEFAULT) -> tuple[CommandSeat, Tape]:
    tape = Tape(commands)
    seat = CommandSeat(source=tape, abi=abi)
    seat.create_context(JOB, "d")
    return seat, tape


def _drive(seat: CommandSeat, steps: int) -> list[OpKind]:
    return [seat.decode(JOB, allow_control=False).request.op for _ in range(steps)]


# -- the ABI ----------------------------------------------------------------


def test_each_declared_verb_asks_for_its_op() -> None:
    write = DEFAULT.parse("write stdout hello")
    assert write.op is OpKind.WRITE
    assert write.pipe == "stdout"
    assert [t.text for t in write.payload] == ["hello"]
    assert DEFAULT.parse("read stdin").op is OpKind.READ
    assert DEFAULT.parse("exit").op is OpKind.EXIT
    assert DEFAULT.parse("say 41").op is OpKind.NONE


def test_an_undeclared_verb_is_a_malformed_request_carrying_the_words() -> None:
    request = DEFAULT.parse("shove stdout hello;")
    assert request.op is OpKind.MALFORMED
    assert request.text == "shove stdout hello"


def test_a_call_missing_its_pipe_is_a_malformed_request() -> None:
    assert DEFAULT.parse("write").op is OpKind.MALFORMED
    assert DEFAULT.parse("read").op is OpKind.MALFORMED
    assert DEFAULT.parse("write").text == "write"


def test_a_verb_is_recognised_whatever_its_case() -> None:
    assert DEFAULT.parse("WRITE stdout hello").op is OpKind.WRITE


def test_an_unknown_pipe_is_left_for_the_kernel_to_refuse() -> None:
    """The parser knows the verbs, not the pipes; the capability check answers those."""
    assert DEFAULT.parse("write shove hello").pipe == "shove"


def test_the_pattern_finds_one_command_in_prose() -> None:
    match = DEFAULT.pattern().search("Sure! Here you go:\n\n`write stdout go;`\n\nHope it helps.")
    assert match is not None
    assert match.group(1) == "write" and match.group(2) == " stdout go"


def test_the_prose_names_every_verb_once() -> None:
    prose = DEFAULT.prose()
    for verb in DEFAULT.verbs:
        assert prose.count(verb.signature(";")) == 1
        assert verb.doc in prose


def test_an_abi_must_be_well_formed() -> None:
    with pytest.raises(ValueError):
        SyscallABI(verbs=())
    with pytest.raises(ValueError):
        SyscallABI(verbs=(Verb("say"), Verb("SAY")))
    with pytest.raises(ValueError):
        SyscallABI(verbs=(Verb("say"),), terminator=" ")
    with pytest.raises(ValueError):
        SyscallABI(verbs=(Verb("start", OpKind.SPAWN, pipe=True, text=True),))


# -- verbs that name a target rather than carrying a payload ----------------

NAMING = SyscallABI(
    verbs=(
        Verb("start", OpKind.SPAWN, text=True, doc="start a child"),
        Verb("want", OpKind.NEED, text=True, doc="ask for content"),
        Verb("stop", OpKind.EXIT),
    ),
)


def test_a_naming_verb_puts_its_argument_where_the_kernel_reads_it() -> None:
    """The kernel resolves a spawn target from ``text``; a payload it cannot look up."""
    request = NAMING.parse("start clear-bench;")
    assert request.op is OpKind.SPAWN
    assert request.text == "clear-bench"
    assert request.payload == ()


def test_a_naming_verb_keeps_a_multi_word_name_whole() -> None:
    assert NAMING.parse("want maintenance log for pump 4;").text == "maintenance log for pump 4"


def test_a_naming_verb_with_no_name_is_malformed() -> None:
    """As a pipe verb with no pipe is: the command closed, and asks for nothing nameable."""
    request = NAMING.parse("start;")
    assert request.op is OpKind.MALFORMED
    assert request.text == "start"


def test_a_naming_verb_is_offered_a_name_in_the_prose() -> None:
    assert "start <name>;" in NAMING.prose()
    assert "<text>" not in NAMING.prose()


# -- the parser -------------------------------------------------------------


def test_words_carry_a_leading_space_except_the_first() -> None:
    assert words_of("write stdout hello", terminator=";", lead=False) == [
        "write",
        " stdout",
        " hello;",
    ]
    assert words_of("exit;", terminator=";", lead=True) == [" exit;"]


def test_the_parser_fires_on_the_piece_carrying_the_terminator() -> None:
    parser = SyscallParser()
    assert [parser.feed(p).op for p in ["write", " stdout", " hello;"]] == [
        OpKind.NONE,
        OpKind.NONE,
        OpKind.WRITE,
    ]
    assert parser.buffer == "" and parser.lines == ["write stdout hello"]


def test_the_parser_does_not_care_how_the_text_was_cut() -> None:
    """Words from a split string or tokens from a model: only the terminator matters."""
    parser = SyscallParser()
    ops = [parser.feed(p).op for p in ["write", " std", "out", " hell", "o", ";"]]
    assert ops[-1] is OpKind.WRITE and all(op is OpKind.NONE for op in ops[:-1])
    assert parser.lines == ["write stdout hello"]


# -- the seat ---------------------------------------------------------------


def test_a_command_is_spent_one_word_per_decode() -> None:
    seat, tape = _seat(["write stdout hello;", "exit;"])
    assert _drive(seat, 4) == [OpKind.NONE, OpKind.NONE, OpKind.WRITE, OpKind.EXIT]
    assert [t.text for t in seat.transcript(JOB)] == ["write", " stdout", " hello;", " exit;"]
    assert len(tape.turns) == 2, "one call to the source per command, not per word"


def test_the_transcript_is_rebuilt_from_the_tokens() -> None:
    seat, _ = _seat(["write stdout hello;", "exit;"])
    _drive(seat, 4)
    assert seat.render(JOB) == "write stdout hello; exit;"
    assert seat.lines(JOB) == ("write stdout hello", "exit")


def test_a_source_is_told_what_the_job_has_issued() -> None:
    seat, tape = _seat(["say 1;", "say 2;", "exit;"])
    _drive(seat, 5)
    assert [(t.issued, t.last) for t in tape.turns] == [(0, ""), (1, "say 1"), (2, "say 2")]


def test_a_seat_speaks_whatever_abi_it_is_given() -> None:
    seat, _ = _seat(["note thinking hard!", "send out hello world!", "done!"], abi=CUSTOM)
    ops = _drive(seat, 8)
    assert ops == [OpKind.NONE] * 6 + [OpKind.WRITE, OpKind.EXIT]
    assert seat.lines(JOB) == ("note thinking hard", "send out hello world", "done")


def test_a_seat_knows_no_words_but_its_own() -> None:
    seat, _ = _seat(["write stdout hello;"], abi=CUSTOM)
    assert _drive(seat, 3) == [OpKind.NONE, OpKind.NONE, OpKind.MALFORMED]
    assert seat.lines(JOB) == ("write stdout hello;",)


# -- the tape ---------------------------------------------------------------


def _turn(issued: int) -> Turn:
    return Turn(job=JOB, descriptor="d", transcript="", issued=issued)


def test_a_tape_plays_its_emit_steps_in_order() -> None:
    tape = TapeSource({"d": Script.from_spec([{"emit": "say 1;"}, {"emit": "exit;"}])})
    assert [tape.next_command(_turn(i)) for i in range(2)] == ["say 1;", "exit;"]
    with pytest.raises(ScriptExhausted):
        tape.next_command(_turn(2))


def test_a_tape_refuses_a_step_that_is_not_a_command() -> None:
    """A seat asks the kernel through words; a script's ``exit: true`` never reaches it."""
    tape = TapeSource({"d": Script.from_spec([{"exit": True}])})
    with pytest.raises(ValueError):
        tape.next_command(_turn(0))


# -- what a descriptor may name ----------------------------------------------


def test_aliases_are_the_ones_the_descriptor_binds() -> None:
    d = Descriptor.from_frontmatter(
        {"name": "d", "priority": 50, "pipes": {"stdout": "ops.report", "tools": "act.a"}}
    )
    assert bound_aliases(d) == ("stdout", "tools")
    aliases, valued = seat_maps({d.name: d}, [PipeSpec(PipeName("act.a"), world_object="count.a")])
    assert aliases == {"d": ("stdout", "tools")} and valued == {"d": ("tools",)}


# -- a case, through the kernel's own driver --------------------------------


def _play() -> tuple[list[Event], list[JobState]]:
    bundle = load_case(FIXTURE)
    events: list[Event] = []
    kernel, transport = build_kernel(
        bundle, machine=CommandSeat(source=TapeSource(bundle.scripts)), journal_sink=events
    )
    driver = Driver(kernel, transport=transport)
    driver.boot(bundle.boot)
    driver.run()
    return events, [j.state for j in kernel.sched.jobs()]


def test_a_case_runs_through_the_seat_on_the_kernels_driver() -> None:
    events, states = _play()
    written = [e for e in events if isinstance(e, PipeWritten) and e.pipe == "ops.report"]
    assert [" ".join(t.strip() for t in w.text) for w in written] == ["all done"]
    assert all(e.tokens == 1 for e in events if isinstance(e, Decoded)), "one word per decode"
    assert states == [JobState.DONE]


def test_the_seat_run_is_the_same_run_twice() -> None:
    first, _ = _play()
    again, _ = _play()
    assert [to_line(i, e) for i, e in enumerate(first)] == [
        to_line(i, e) for i, e in enumerate(again)
    ]


def test_the_cli_runs_a_case_through_the_seat(tmp_path: Path) -> None:
    from zeos.cli import main

    journal = tmp_path / "seat.jsonl"
    assert main(["run", str(FIXTURE), "--machine", "seat", "--journal", str(journal)]) == 0
    assert main(["replay", str(journal), "--assert-identical"]) == 0
