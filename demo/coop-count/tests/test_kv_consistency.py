# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Decoding still works after an operation that rewrites a context's token history."""

from __future__ import annotations

import pytest

from zeos.core.ids import JobId
from zeos.core.ids import TokenKind
from zeos.machine.base import Token, render, tokens_from_text

from zeos_coop_count import model as model_mod
from zeos_coop_count.machine import LlamaMachine

JOB = JobId(1)


@pytest.fixture
def machine(llama_model, machines) -> LlamaMachine:  # pyright: ignore[reportMissingParameterType]
    m = LlamaMachine(
        llama_model,
        descriptors={"d": ("stdin", "stdout")},
        block_size=16,
        n_ctx=2048,
        n_seq_max=2,
        n_threads=model_mod.DEFAULT_THREADS,
    )
    machines.append(m)
    m.create_context(JOB, "d")
    m.inject(JOB, tokens_from_text(" ".join(str(i) for i in range(60))))
    return m


def test_decode_still_works_after_trunc(machine: LlamaMachine) -> None:
    machine.decode(JOB, allow_control=False)
    machine.trunc(JOB, 20)
    assert machine.stats(JOB).resident_tokens == 20
    machine.decode(JOB, allow_control=False)  # raises if the cache and the transcript differ


def test_decode_still_works_after_splice(machine: LlamaMachine) -> None:
    """Decoding works after a span is replaced by a stub, and the counts are reported."""
    machine.decode(JOB, allow_control=False)
    stub = tuple(Token(t, TokenKind.CONTROL) for t in ("<STUB", "segment=6", "elided)", "</STUB>"))
    result = machine.splice(JOB, 10, 30, stub)

    assert result.tokens_in == len(stub)
    assert result.invalidated_downstream == 61 - 30
    machine.decode(JOB, allow_control=False)


def test_splice_leaves_the_transcript_intact_around_the_stub(machine: LlamaMachine) -> None:
    before_head = render(machine.transcript(JOB)[:10])
    tail_before = render(machine.transcript(JOB)[30:])
    machine.splice(JOB, 10, 30, tokens_from_text("X"))

    transcript = machine.transcript(JOB)
    assert render(transcript[:10]) == before_head
    assert render(transcript[11:]) == tail_before


def test_decode_still_works_after_a_fork(machine: LlamaMachine) -> None:
    child = JobId(2)
    machine.decode(JOB, allow_control=False)
    assert machine.fork(JOB, child) == machine.stats(JOB).resident_tokens
    machine.decode(child, allow_control=False)
    machine.decode(JOB, allow_control=False)
