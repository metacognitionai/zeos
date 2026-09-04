# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The M0 backend: the five ops, blocks, masking, and control-token gating."""

from __future__ import annotations

import pytest

from zeos.core.ids import JobId, PipeName, SegmentId, TokenKind
from zeos.machine.base import ControlTokenViolation, OpKind, Token, render
from zeos.machine.scripted import (
    PAD_TOKEN,
    Script,
    ScriptedMachine,
    ScriptExhausted,
    Step,
)

JOB = JobId(1)


def machine(steps: list[dict[str, object]], block_size: int = 4) -> ScriptedMachine:
    m = ScriptedMachine({"d": Script.from_spec(steps)}, block_size=block_size)
    m.create_context(JOB, "d")
    return m


def test_decode_emits_tokens_and_advances() -> None:
    m = machine([{"emit": "alpha beta"}, {"emit": "gamma"}, {"exit": True}])
    first = m.decode(JOB, allow_control=False)
    assert render(first.tokens) == "alpha beta"
    second = m.decode(JOB, allow_control=False)
    assert render(second.tokens) == "gamma"
    third = m.decode(JOB, allow_control=False)
    assert third.request.op is OpKind.EXIT
    assert render(m.transcript(JOB)) == "alpha beta gamma"


def test_running_off_the_end_of_a_script_is_an_error() -> None:
    """Silently completing would hide an authoring mistake behind a passing test."""
    m = machine([{"emit": "only"}])
    m.decode(JOB, allow_control=False)
    with pytest.raises(ScriptExhausted, match="past the end"):
        m.decode(JOB, allow_control=False)


def test_control_tokens_are_unavailable_unless_the_kernel_enables_them() -> None:
    m = machine([{"emit": "<FAULT>", "control": True}])
    with pytest.raises(ControlTokenViolation):
        m.decode(JOB, allow_control=False)


def test_control_tokens_pass_when_enabled() -> None:
    m = machine([{"emit": "<FAULT>", "control": True}])
    result = m.decode(JOB, allow_control=True)
    assert result.tokens[0].kind is TokenKind.CONTROL


def test_block_boundary_is_signalled_exactly_on_the_boundary() -> None:
    m = machine([{"emit": "a b c"}, {"emit": "d"}, {"emit": "e"}], block_size=4)
    assert m.decode(JOB, allow_control=False).at_block_boundary is False  # 3 tokens
    assert m.decode(JOB, allow_control=False).at_block_boundary is True  # 4 tokens
    assert m.decode(JOB, allow_control=False).at_block_boundary is False  # 5 tokens


def test_padding_is_bounded_by_block_size_minus_one() -> None:
    """At most block_size - 1 padding tokens per boundary event."""
    for emitted in range(1, 9):
        m = machine([{"emit": " ".join("x" * emitted)}], block_size=4)
        m.decode(JOB, allow_control=False)
        padding = m.pad_to_block(JOB)
        assert padding <= 3
        assert len(m.transcript(JOB)) % 4 == 0
        if padding:
            assert m.transcript(JOB)[-1] == PAD_TOKEN


def test_padding_is_a_no_op_when_already_aligned() -> None:
    m = machine([{"emit": "a b c d"}], block_size=4)
    m.decode(JOB, allow_control=False)
    assert m.pad_to_block(JOB) == 0


def test_blocks_for_range_maps_offsets_to_blocks() -> None:
    m = machine([{"emit": " ".join(str(i) for i in range(10))}], block_size=4)
    m.decode(JOB, allow_control=False)
    assert m.blocks_for_range(JOB, 0, 4) == frozenset({0})
    assert m.blocks_for_range(JOB, 3, 5) == frozenset({0, 1})
    assert m.blocks_for_range(JOB, 8, 10) == frozenset({2})
    assert m.blocks_for_range(JOB, 5, 5) == frozenset()


def test_mask_restricts_visible_blocks() -> None:
    m = machine([{"emit": " ".join(str(i) for i in range(12))}], block_size=4)
    m.decode(JOB, allow_control=False)
    assert m.visible_blocks(JOB) == frozenset({0, 1, 2})
    m.set_mask(JOB, frozenset({0, 2}))
    assert m.visible_blocks(JOB) == frozenset({0, 2})
    m.clear_mask(JOB)
    assert m.visible_blocks(JOB) == frozenset({0, 1, 2})


def test_inject_returns_the_range_it_wrote() -> None:
    m = machine([{"emit": "a b"}])
    m.decode(JOB, allow_control=False)
    start, end = m.inject(JOB, [Token("x"), Token("y"), Token("z")])
    assert (start, end) == (2, 5)
    assert render(m.transcript(JOB)) == "a b x y z"


def test_trunc_drops_the_tail() -> None:
    m = machine([{"emit": "a b c d e"}])
    m.decode(JOB, allow_control=False)
    assert m.trunc(JOB, 2) == 3
    assert render(m.transcript(JOB)) == "a b"
    with pytest.raises(IndexError):
        m.trunc(JOB, 99)


def test_fork_copies_context_and_leaves_parent_alone() -> None:
    m = machine([{"emit": "a b"}, {"emit": "parent-only"}])
    m.decode(JOB, allow_control=False)
    child = JobId(2)
    assert m.fork(JOB, child) == 2
    m.decode(JOB, allow_control=False)
    assert render(m.transcript(JOB)) == "a b parent-only"
    assert render(m.transcript(child)) == "a b"


def test_splice_reports_invalidated_downstream() -> None:
    """The ``d`` term of the cost model: splicing invalidates
    everything downstream of the splice point."""
    m = machine([{"emit": "a b c d e f"}])
    m.decode(JOB, allow_control=False)
    result = m.splice(JOB, 1, 3, [Token("X")])
    assert result.tokens_in == 1
    assert result.invalidated_downstream == 3  # d e f
    assert render(m.transcript(JOB)) == "a X d e f"


def test_script_from_spec_parses_every_request_form() -> None:
    script = Script.from_spec(
        [
            {"read": "plant.tags"},
            {"write": {"pipe": "ops.report", "text": "shift nominal"}},
            {"select": ["a", "b"]},
            {"fault": 7},
            {"need": "maintenance log for pump 4"},
            {"spawn": "clear-bench"},
            {"emit": "thinking", "attend": ["web.fetch"]},
            {"exit": True},
        ]
    )
    ops = [s.request.op for s in script.steps]
    assert ops == [
        OpKind.READ,
        OpKind.WRITE,
        OpKind.SELECT,
        OpKind.FAULT,
        OpKind.NEED,
        OpKind.SPAWN,
        OpKind.NONE,
        OpKind.EXIT,
    ]
    assert script.steps[0].request.pipe == PipeName("plant.tags")
    assert render(script.steps[1].request.payload) == "shift nominal"
    assert script.steps[2].request.pipes == (PipeName("a"), PipeName("b"))
    assert script.steps[3].request.segment == SegmentId(7)
    hint = script.steps[6].hint
    assert hint is not None and hint.tags == ("web.fetch",)


def test_recency_profile_is_normalised_and_tail_weighted() -> None:
    """Synthetic attention. Asserted for shape and determinism only -- no policy
    conclusion rests on this curve."""
    m = machine([{"emit": " ".join(str(i) for i in range(12))}], block_size=4)
    m.decode(JOB, allow_control=False)
    profile = m.recency_profile(JOB, scale=2.0)
    assert profile.keys() == {0, 1, 2}
    assert sum(profile.values()) == pytest.approx(1.0)
    assert profile[2] > profile[1] > profile[0]
    assert m.recency_profile(JOB, scale=2.0) == profile  # deterministic


def test_recency_profile_respects_the_mask() -> None:
    """A masked segment contributes no attention mass -- hard enforcement, not a
    request the job could decline."""
    m = machine([{"emit": " ".join(str(i) for i in range(12))}], block_size=4)
    m.decode(JOB, allow_control=False)
    m.set_mask(JOB, frozenset({0}))
    assert m.recency_profile(JOB, scale=2.0).keys() == {0}


def test_unknown_descriptor_gets_an_empty_script() -> None:
    m = ScriptedMachine({}, block_size=4)
    m.create_context(JobId(9), "nonexistent")
    with pytest.raises(ScriptExhausted):
        m.decode(JobId(9), allow_control=False)


def test_step_tokens_are_empty_when_nothing_is_emitted() -> None:
    assert Step().tokens() == ()
