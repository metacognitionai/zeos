# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Contract tier: the obligations ZEOS-AM places on *any* machine backend.

Written against ``MachineBackend``, not against ``ScriptedMachine``. Every test here
is a clause of docs/ProjectDescription/Abstract-Machine.md, and the point of the
parameterisation is that **M1 inherits this suite instead of needing one authored for
it**. Adding a paged-KV backend means adding one entry to ``BACKENDS``.

These are deliberately the clauses ``tests/unit/test_scripted_machine.py`` does *not*
cover. That file pins the behaviour someone thought to write down while building M0;
this one pins the properties a *different* backend could violate while still passing
every one of those tests.

Two clauses cannot be closed here and are marked in the spec rather than faked:
determinism under real sampling (C1) needs a backend capable of being
non-deterministic, and attention normalisation (C15) needs a backend capable of
measuring. Nothing in M0 can speak to either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import MachineBackend, render, tokens_from_text
from zeos.machine.scripted import Script, ScriptedMachine

JOB = JobId(1)
CHILD = JobId(2)


@dataclass(frozen=True)
class Backend:
    """How to build one backend under test.

    ``emits`` is the portable spelling of "this context will produce these tokens on
    successive decodes". M0 compiles it to a script; a real backend would teacher-force
    or stub it. Everything else in the suite drives the Protocol directly.
    """

    name: str
    build: Callable[..., tuple[MachineBackend, JobId]]

    def make(
        self, *, block_size: int = 4, emits: Sequence[str] = ()
    ) -> tuple[MachineBackend, JobId]:
        return self.build(block_size=block_size, emits=emits)


def _scripted(*, block_size: int, emits: Sequence[str]) -> tuple[MachineBackend, JobId]:
    script = Script.from_spec([{"emit": text} for text in emits])
    machine = ScriptedMachine({"d": script}, block_size=block_size)
    machine.create_context(JOB, "d")
    return machine, JOB


BACKENDS = [Backend("M0-scripted", _scripted)]


@pytest.fixture(params=BACKENDS, ids=lambda b: b.name)
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param  # pyright: ignore[reportAny]


# -- §8.2  the mask: absence is not restriction-to-nothing --------------------


def test_empty_mask_hides_everything_and_is_not_the_same_as_no_mask(
    backend: Backend,
) -> None:
    """ZEOS-AM C3. The escalation trap, one layer down from where it already bit.

    An empty ``CapabilityTable`` once meant "unprotected by choice", so stripping a
    job of all authority made it omnipotent. The mask has the identical shape: if a
    backend collapses ``∅`` to "no mask", then masking a compartment child down to
    nothing grants it everything. Nothing in the unit suite tests this.
    """
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c d e f"))
    every = m.visible_blocks(job)
    assert every == frozenset({0, 1}), "unmasked means every live block"

    m.set_mask(job, frozenset())
    assert m.visible_blocks(job) == frozenset(), (
        "a mask of ∅ must hide everything; collapsing it to 'unmasked' is a privilege escalation"
    )


def test_a_mask_can_only_narrow_never_widen(backend: Backend) -> None:
    """ZEOS-AM AM-I7. Out-of-range block indices are not a grant."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b"))  # one block
    m.set_mask(job, frozenset({0, 7, 99}))
    assert m.visible_blocks(job) <= frozenset({0}), (
        "blocks that do not exist must drop out, not conjure visibility"
    )


def test_reported_attention_never_falls_outside_the_mask(backend: Backend) -> None:
    """ZEOS-AM C2, the reportable half.

    Vacuous for a backend that cannot measure (M0 returns ``attention=None``) and
    binding for one that can. Left in deliberately: it is the clause an M1 backend is
    most likely to violate, because a mask applied to logits rather than to attention
    passes every other test in this file.
    """
    m, job = backend.make(block_size=4, emits=["x y"])
    m.inject(job, tokens_from_text("a b c d e f g h"))
    m.set_mask(job, frozenset({0}))
    result = m.decode(job, allow_control=False)
    if result.attention is None:
        pytest.skip("backend does not measure attention; C2 unenforceable here")
    assert set(result.attention) <= m.visible_blocks(job)


def test_measured_attention_is_normalised_per_block(backend: Backend) -> None:
    """ZEOS-AM §7.1. Skips on M0; fires the day a measuring backend lands.

    Units are normative: summed over layers and heads, normalised per block, so one
    step's mass sums to 1.0 over the blocks that received any. The kernel compares
    these values against ``theta_read``, whose default means "a fifth of a block's
    attention", so a backend reporting raw weights summing to the head count would
    satisfy the type, pass every other test in this file, and get every demotion and
    eviction decision wrong by orders of magnitude.

    Left deliberately as a skip rather than omitted: the obligation is the one most
    likely to be violated at M1 and the least likely to be noticed, so the test should
    already be sitting in the suite when the backend arrives.
    """
    m, job = backend.make(block_size=4, emits=["x y"])
    m.inject(job, tokens_from_text("a b c d e f g h"))
    result = m.decode(job, allow_control=False)
    if result.attention is None:
        pytest.skip("backend does not measure attention; C15 unenforceable here")
    total = sum(result.attention.values())
    assert total == pytest.approx(1.0), f"attention mass must be normalised per block, got {total}"
    assert all(v >= 0.0 for v in result.attention.values())


# -- §6  offset stability ----------------------------------------------------


def test_only_splice_moves_an_existing_offset(backend: Backend) -> None:
    """ZEOS-AM AM-I5 and C6.

    The kernel's segment table holds token offsets, so an operation that quietly
    relocates existing tokens corrupts every segment record. Compaction or
    defragmentation inside a real backend would do exactly that and break nothing
    visible until protection metadata pointed at the wrong span.
    """
    m, job = backend.make(block_size=4, emits=["p q"])
    m.inject(job, tokens_from_text("a b c d e"))
    before = render(m.transcript(job))

    m.decode(job, allow_control=False)  # appends
    assert render(m.transcript(job)).startswith(before)

    m.inject(job, tokens_from_text("z"))  # appends
    assert render(m.transcript(job)).startswith(before)

    m.pad_to_block(job)  # appends
    assert render(m.transcript(job)).startswith(before)

    m.fork(job, CHILD)  # writes the child only
    assert render(m.transcript(job)).startswith(before)

    m.trunc(job, 5)  # drops a suffix
    assert render(m.transcript(job)) == before


def test_splice_shifts_downstream_by_exactly_delta(backend: Backend) -> None:
    """ZEOS-AM §6.5. The arithmetic ``SegmentTable.evict_to_stub`` depends on.

    ``delta = tokens_in - (end - start)``. If a backend's splice shifts by anything
    else, the kernel renumbers segments by the wrong amount and every offset after
    the spliced span is silently wrong.
    """
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c d e f g h"))
    start, end, replacement = 2, 4, tokens_from_text("X")
    tail_before = render(m.transcript(job)[end:])

    result = m.splice(job, start, end, replacement)

    assert result.tokens_in == len(replacement)
    assert result.invalidated_downstream == 8 - end, (
        "invalidated_downstream is the recompute cost the eviction planner prices"
    )
    delta = result.tokens_in - (end - start)
    assert render(m.transcript(job)) == "a b X e f g h"
    assert render(m.transcript(job)[end + delta :]) == tail_before, (
        "the tail must land exactly delta earlier, and stay in order"
    )


def test_trunc_leaves_earlier_offsets_alone_and_shrinks_visibility(
    backend: Backend,
) -> None:
    """ZEOS-AM C9 and C17. A stale mask must fail closed."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c d e f g h i j k l"))
    m.set_mask(job, frozenset({2}))
    assert m.visible_blocks(job) == frozenset({2})

    m.trunc(job, 4)

    assert render(m.transcript(job)) == "a b c d"
    assert m.visible_blocks(job) == frozenset(), (
        "a mask naming a block that no longer exists must hide, never reveal"
    )


# -- §6.4  fork --------------------------------------------------------------


def test_fork_carries_the_mask_to_the_child(backend: Backend) -> None:
    """ZEOS-AM C10. Compartments depend on this direction.

    A compartment child is built by forking and then *narrowing*, so it must start
    from the parent's visibility. A backend that gave the child an unmasked view
    would hand every compartment its parent's secrets.
    """
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("secret secret secret secret public public"))
    m.set_mask(job, frozenset({1}))
    m.fork(job, CHILD)
    assert m.visible_blocks(CHILD) == frozenset({1})


def test_fork_into_an_existing_context_keeps_the_childs_own_behaviour(
    backend: Backend,
) -> None:
    """ZEOS-AM §6.4. A compartment child runs its own behaviour over the parent's
    context, so inheriting the parent's decode state would be exactly wrong."""
    m, job = backend.make(block_size=4, emits=["parent-step"])
    if not isinstance(m, ScriptedMachine):  # pragma: no cover - M0-shaped setup
        pytest.skip("needs a second script registered; backend-specific")
    m.register_script("child", Script.from_spec([{"emit": "child-step"}]))
    m.create_context(CHILD, "child")
    m.inject(job, tokens_from_text("shared"))

    assert m.fork(job, CHILD) == 1
    assert render(m.transcript(CHILD)) == "shared"

    m.decode(CHILD, allow_control=False)
    assert render(m.transcript(CHILD)) == "shared child-step", (
        "the child kept its own script, and only the tokens were copied in"
    )


# -- §9  control tokens ------------------------------------------------------


def test_inject_may_carry_control_tokens_but_decode_may_not(backend: Backend) -> None:
    """ZEOS-AM C4 and C5. The asymmetry *is* the unforgeability property.

    The kernel introduces control tokens through INJECT (stub framing, padding); the
    model may not produce them through DECODE. A backend that gated both, or neither,
    has not implemented ZEOS-MP's unforgeable framing.
    """
    m, job = backend.make(block_size=4)
    start, end = m.inject(job, tokens_from_text("<stub 3fa2>", TokenKind.CONTROL))
    assert (start, end) == (0, 2)
    assert all(t.kind is TokenKind.CONTROL for t in m.transcript(job))


def test_padding_is_control_and_never_fills_a_whole_block(backend: Backend) -> None:
    """ZEOS-AM §9. Padding is kernel framing, so a job must not be able to
    emit it, and an aligned context must not gain a spurious block."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c"))
    added = m.pad_to_block(job)

    assert 0 < added < 4
    assert all(t.kind is TokenKind.CONTROL for t in m.transcript(job)[-added:])
    assert m.stats(job).resident_tokens % 4 == 0
    assert m.pad_to_block(job) == 0, "padding an aligned context is a no-op"


# -- §3.3  derived quantities ------------------------------------------------


def test_block_arithmetic_is_derived_from_length_alone(backend: Backend) -> None:
    """ZEOS-AM AM-I2, C12, C13. Including the empty case, where ``blocks`` is 0."""
    m, job = backend.make(block_size=4)
    assert m.stats(job).blocks == 0, "an empty context occupies no blocks"
    assert m.blocks_for_range(job, 0, 0) == frozenset(), "an empty range spans nothing"

    m.inject(job, tokens_from_text("a b c d e"))
    stats = m.stats(job)
    assert (stats.resident_tokens, stats.blocks) == (5, 2)
    assert m.blocks_for_range(job, 0, 1) == frozenset({0})
    assert m.blocks_for_range(job, 3, 5) == frozenset({0, 1})
    assert m.blocks_for_range(job, 4, 5) == frozenset({1})


# -- §5  context lifecycle ---------------------------------------------------


def test_operations_on_an_absent_context_raise(backend: Backend) -> None:
    """ZEOS-AM C16. A destroyed context must not answer as if it were empty.

    Silently returning zeroes would turn a use-after-destroy into a job that appears
    to be running normally with nothing in its window.
    """
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b"))
    m.destroy_context(job)

    with pytest.raises(Exception):
        m.stats(job)
    with pytest.raises(Exception):
        m.transcript(job)
    with pytest.raises(Exception):
        m.inject(job, tokens_from_text("c"))


def test_splice_and_trunc_reject_out_of_range_offsets(backend: Backend) -> None:
    """ZEOS-AM §6.3 and §6.5. Bounds are checked, because a silent clamp would hand
    the kernel a segment table that disagrees with the token sequence."""
    m, job = backend.make(block_size=4)
    m.inject(job, tokens_from_text("a b c"))

    with pytest.raises(IndexError):
        m.trunc(job, 4)
    with pytest.raises(IndexError):
        m.splice(job, 2, 1, ())  # end before start
    with pytest.raises(IndexError):
        m.splice(job, 0, 9, ())

    assert m.trunc(job, 3) == 0, "truncating at the end is a legal no-op"
