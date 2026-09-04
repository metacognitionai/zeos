# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Stage A acceptance: a scripted stream runs through the machine, every op lands
in the journal, and the journal folds back to identical state.

"Identical state" means the kernel-visible accounting -- resident token counts, block
counts, offsets -- not the token text. That distinction is deliberate and follows the
specs: the transcript is the source of truth for *content* (core §2.1), while the
journal is what the segment table and scheduler state are recoverable from. A
journal that also carried every token would be a transcript with extra
steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.clock import Clock
from zeos.core.events import (
    BlockBoundary,
    Decoded,
    Event,
    Injected,
    KernelStarted,
    Spliced,
    Truncated,
)
from zeos.core.ids import (
    KERNEL_PIPE,
    Integrity,
    JobId,
    Principal,
    Ring,
    SegmentId,
)
from zeos.journal.writer import Journal, read_journal
from zeos.machine.base import Token, render
from zeos.machine.scripted import Script, ScriptedMachine

JOB = JobId(1)
BLOCK = 4

SCRIPT = [
    {"emit": "checking unit three"},
    {"emit": "reading tags now"},
    {"read": "plant.tags"},
    {"emit": "unit three nominal"},
    {"write": {"pipe": "ops.report", "text": "shift nominal"}},
    {"exit": True},
]


def run_substrate(journal: Journal) -> ScriptedMachine:
    """Drive one scripted job through the machine, journalling every op.

    Stands in for ``driver.py``, which arrives in Stage B. Kept deliberately dumb:
    it exercises the substrate, it does not schedule.
    """
    machine = ScriptedMachine({"d": Script.from_spec(SCRIPT)}, block_size=BLOCK)
    machine.create_context(JOB, "d")
    clock = Clock()
    journal.append(KernelStarted(clock=clock, seed=7, block_size=BLOCK, case="stage-a"))

    segment = SegmentId(1)
    for _ in range(len(SCRIPT)):
        result = machine.decode(JOB, allow_control=False)
        if result.tokens:
            clock = clock.tick_tokens(len(result.tokens))
            journal.append(
                Decoded(clock=clock, job=JOB, segment=segment, tokens=len(result.tokens))
            )
        if result.at_block_boundary:
            journal.append(
                BlockBoundary(
                    clock=clock,
                    job=JOB,
                    block=len(machine.transcript(JOB)) // BLOCK - 1,
                    padding_tokens=0,
                )
            )
        if result.request.pipe is not None and result.request.op.value == "read":
            # Servicing the read: the injected payload is the only entry path for
            # foreign tokens, so it is what makes provenance total.
            payload = [Token("unit_a"), Token("running")]
            start, end = machine.inject(JOB, payload)
            clock = clock.tick_tokens(len(payload))
            segment = SegmentId(segment + 1)
            journal.append(
                Injected(
                    clock=clock,
                    job=JOB,
                    segment=segment,
                    pipe=result.request.pipe,
                    principal=Principal.DEVICE,
                    ring=Ring.EXTERNAL,
                    integrity=Integrity(3),
                    tokens=end - start,
                )
            )

    padding = machine.pad_to_block(JOB)
    if padding:
        journal.append(
            BlockBoundary(
                clock=clock,
                job=JOB,
                block=len(machine.transcript(JOB)) // BLOCK - 1,
                padding_tokens=padding,
            )
        )
    return machine


def fold_resident_tokens(events: list[Event]) -> int:
    """Kernel state as a fold over the journal -- the property the whole design rests on."""
    total = 0
    for event in events:
        match event:
            case Decoded() | Injected():
                total += event.tokens
            case BlockBoundary():
                total += event.padding_tokens
            case Truncated():
                total = event.at
            case Spliced():
                total += event.tokens_in - event.tokens_out
            case _:
                pass  # events that do not change resident size
    return total


def test_every_op_lands_in_the_journal() -> None:
    journal = Journal()
    run_substrate(journal)

    # Three of the six steps emit tokens; read/write/exit carry a request only.
    assert len(journal.of_kind(Decoded)) == 3
    assert len(journal.of_kind(Injected)) == 1  # one serviced read
    assert journal.of_kind(BlockBoundary), "block boundaries must be journalled"
    assert journal.of_kind(KernelStarted)[0].block_size == BLOCK


def test_journal_folds_to_the_machines_actual_state() -> None:
    journal = Journal()
    machine = run_substrate(journal)
    folded = fold_resident_tokens(list(journal.events()))
    assert folded == machine.stats(JOB).resident_tokens


def test_transcript_is_block_aligned_after_padding() -> None:
    journal = Journal()
    machine = run_substrate(journal)
    assert len(machine.transcript(JOB)) % BLOCK == 0


def test_injected_content_carries_its_provenance() -> None:
    """INJECT is the only entry path for foreign tokens, so provenance is total."""
    journal = Journal()
    run_substrate(journal)
    injected = journal.of_kind(Injected)[0]
    assert injected.pipe == "plant.tags"
    assert injected.principal is Principal.DEVICE
    assert injected.ring is Ring.EXTERNAL
    assert injected.pipe != KERNEL_PIPE


@pytest.mark.determinism
def test_two_identical_runs_produce_byte_identical_journals() -> None:
    """Acceptance criterion 1. Compared as bytes, not as parsed structures -- a
    comparison tolerant of reordering would miss nondeterministic iteration order
    leaking into kernel decisions, which is the bug class we most care about."""
    first, second = Journal(), Journal()
    run_substrate(first)
    run_substrate(second)
    assert first.to_bytes() == second.to_bytes()


@pytest.mark.determinism
def test_journal_survives_a_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with Journal(path) as journal:
        machine = run_substrate(journal)
        expected = journal.to_bytes()
        expected_tokens = machine.stats(JOB).resident_tokens

    restored = read_journal(path)
    assert [r.seq for r in restored] == list(range(len(restored)))
    assert fold_resident_tokens([r.event for r in restored]) == expected_tokens

    rewritten = Journal()
    rewritten.extend(r.event for r in restored)
    assert rewritten.to_bytes() == expected


def test_transcript_reads_back_as_written() -> None:
    journal = Journal()
    machine = run_substrate(journal)
    text = render(machine.transcript(JOB))
    assert text.startswith("checking unit three reading tags now unit_a running")
    assert "shift nominal" not in text, "pipe payloads are not part of the transcript"
