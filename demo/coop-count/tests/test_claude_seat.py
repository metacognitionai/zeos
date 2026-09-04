# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``CommandSeat`` with the model replaced by a list of canned replies."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import OpKind
from zeos_coop_count.claude import one_command
from zeos_coop_count.seat import CommandSeat, Turn

JOB = JobId(1)


class Replies:
    """A command source that returns one canned reply per turn, cleaned as a model's is."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.asked = 0

    def next_command(self, turn: Turn) -> str:
        self.asked += 1
        return one_command(self._replies.pop(0) if self._replies else "exit;")


def build(replies: Sequence[str]) -> tuple[CommandSeat, Replies]:
    source = Replies(replies)
    seat = CommandSeat(source=source)
    seat.create_context(JOB, "counter-a")
    return seat, source


def drive(seat: CommandSeat, steps: int) -> list[OpKind]:
    """Runs ``steps`` decodes and returns the op each one produced."""
    return [seat.decode(JOB, allow_control=False).request.op for _ in range(steps)]


@pytest.fixture
def seat() -> tuple[CommandSeat, Replies]:
    return build(["say 41;", "write tools 50;", "read stdin;"])


def test_a_command_leaves_one_word_per_decode(seat: tuple[CommandSeat, Replies]) -> None:
    """One decode emits one word, so a two-word command takes two decodes and one call."""
    machine, source = seat
    ops = drive(machine, 2)

    assert ops == [OpKind.NONE, OpKind.NONE]
    assert source.asked == 1, "one call covered both words"
    assert [t.text for t in machine.transcript(JOB)] == ["say", " 41;"]


def test_the_request_lands_on_the_word_that_closes_the_command(
    seat: tuple[CommandSeat, Replies],
) -> None:
    machine, _ = seat
    ops = drive(machine, 5)

    assert ops == [OpKind.NONE, OpKind.NONE, OpKind.NONE, OpKind.NONE, OpKind.WRITE]
    assert machine.lines(JOB) == ("say 41", "write tools 50")


def test_a_read_becomes_a_read_request(seat: tuple[CommandSeat, Replies]) -> None:
    machine, _ = seat
    ops = drive(machine, 7)

    assert ops[-1] is OpKind.READ
    assert machine.lines(JOB)[-1] == "read stdin"


def test_a_reply_that_is_not_a_command_becomes_a_say() -> None:
    """A reply with no command in it is recorded as a ``say`` instead."""
    machine, _ = build(["I think I should probably count to fifty next."])

    ops = drive(machine, 10)

    assert all(op is OpKind.NONE for op in ops)
    assert machine.lines(JOB)[0].startswith("say I think")


def test_prose_around_a_command_is_discarded() -> None:
    machine, _ = build(["Sure! Here you go:\n\n`write stdout go;`\n\nHope that helps."])

    ops = drive(machine, 3)

    assert ops == [OpKind.NONE, OpKind.NONE, OpKind.WRITE]
    assert machine.lines(JOB) == ("write stdout go",)


def test_only_the_first_command_of_a_reply_is_honoured() -> None:
    """A reply with several commands yields only the first, and the rest is dropped."""
    machine, source = build(["say 41; say 42; write tools 50;", "read stdin;"])

    ops = drive(machine, 4)

    assert ops == [OpKind.NONE, OpKind.NONE, OpKind.NONE, OpKind.READ]
    assert machine.lines(JOB) == ("say 41", "read stdin")
    assert source.asked == 2


def test_the_first_word_of_a_context_carries_no_leading_space() -> None:
    """The first token of a new context has no leading space, and later ones do."""
    machine, _ = build(["say 1;", "say 2;"])

    drive(machine, 4)

    assert [t.text for t in machine.transcript(JOB)] == ["say", " 1;", " say", " 2;"]


def test_what_the_job_says_is_what_the_kernel_sees(seat: tuple[CommandSeat, Replies]) -> None:
    """Each decode adds exactly one normal token to the transcript the kernel counts."""
    machine, _ = seat
    drive(machine, 5)

    tokens = machine.transcript(JOB)
    assert len(tokens) == 5
    assert all(t.kind is TokenKind.NORMAL for t in tokens)
    assert machine.stats(JOB).resident_tokens == 5


def test_a_source_is_told_how_far_the_job_has_got() -> None:
    """``Turn.issued`` counts completed commands, which is what lets a tape be stateless."""
    seen: list[int] = []

    class Counting(Replies):
        def next_command(self, turn: Turn) -> str:
            seen.append(turn.issued)
            return super().next_command(turn)

    machine = CommandSeat(source=Counting(["say 1;", "say 2;", "say 3;"]))
    machine.create_context(JOB, "counter-a")
    drive(machine, 6)

    assert seen == [0, 1, 2]
