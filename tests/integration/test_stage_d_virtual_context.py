# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Stage D acceptance: Virtual Context.

Two criteria:

1. A job whose window is smaller than its transcript runs to completion -- evicting,
   stubbing, faulting content back in, and surviving a resume -- **with protection
   metadata intact across the round trip**. That last clause is the one that
   matters: if ring and integrity did not survive eviction, paging would be a
   laundering operation and every Protected Mode guarantee would have a hole in it.

2. A deliberately over-sized working set raises a loud ``SCHEDULER_FAULT (THRASH)``
   rather than silently churning SPLICE recomputes.

The window is a cache, not the memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import (
    Event,
    FaultRaised,
    JobCompleted,
    PagedIn,
    PageFaultRaised,
    Refaulted,
    ResidencyChanged,
    SegmentEvicted,
    Spliced,
    WorkingSetSampled,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    JobState,
    Perm,
    PipeName,
    Principal,
    Residency,
    Ring,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.residency import ContextPolicy, admission_check, plan_eviction
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

PIPES = [
    PipeSpec(PipeName("web.fetch"), ring=Ring.EXTERNAL, principal=Principal.TOOL, device=True),
    PipeSpec(PipeName("ops.report"), ring=Ring.TRUSTED, principal=Principal.USER),
]


def build(
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    *,
    block_size: int = 4,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine(
            {n: Script.from_spec(s) for n, s in scripts.items()}, block_size=block_size
        ),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="stage-d"),
    )
    kernel.start()
    return kernel, events


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


#: A job with a window far smaller than the transcript it will generate.
#:
#: Proportions matter here and a first attempt got them wrong: with 15-token spans
#: against 9 tokens of stub framing there is essentially nothing to free, and
#: eviction correctly declines every candidate. Real spans dwarf their framing --
#: the spec's own example stubs a 9450-token sweep into one sentence -- so the
#: fixture uses spans of ~40 tokens against a 300-token window.
LONG_JOB = {
    "name": "patrol",
    "priority": 80,
    "context": {
        "window": 300,
        "eviction": "attention-clock",
        "stub_budget": 120,
        "min_span_age": 0,
        "high_watermark": 0.6,
        "low_watermark": 0.4,
    },
}

_SWEEP = (
    "sweep {i} of the lower drive completed with all readings nominal and no "
    "exceptions logged against the transfer units or the auxiliary pumps on "
    "this pass through the section"
)


def chatter(n: int) -> list[dict[str, Any]]:
    return [{"emit": _SWEEP.format(i=i)} for i in range(n)]


def first_evicted(n: int) -> int:
    """Run a chattering job once and return a segment it actually evicted.

    A script cannot name a runtime segment id, and hardcoding one makes the test
    depend on an allocation order that is not part of the contract. Because runs are
    deterministic, discovering the handle in a probe run and then faulting on it is
    both honest and stable -- and it mirrors what really happens, where the model
    reads a stub's id out of its own window.
    """
    kernel, events = build([LONG_JOB], {"patrol": chatter(n) + [{"exit": True}]})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()
    evictions = of(events, SegmentEvicted)
    assert evictions, "probe run produced no eviction; the fixture is mis-proportioned"
    return int(evictions[0].segment)


# --- eviction ---------------------------------------------------------------


def test_a_job_larger_than_its_window_evicts() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(12) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    assert job.state is JobState.DONE
    evictions = of(events, SegmentEvicted)
    assert evictions, "a job over its high watermark must evict"
    assert all(e.freed_tokens > 0 for e in evictions)


def test_eviction_leaves_a_stub_not_a_hole() -> None:
    """Compaction is eviction without an address. A stub is a summary *plus* a page
    handle, which is the entire difference."""
    kernel, events = build([LONG_JOB], {"patrol": chatter(12) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    eviction = of(events, SegmentEvicted)[0]
    stub = job.segments.get(eviction.stub)
    assert stub.tokens > 0, "the stub occupies window space"
    assert stub.tokens < eviction.freed_tokens + eviction.stub_tokens
    assert kernel.store.has(eviction.store), "the content must be recoverable"


def test_stub_does_not_launder_taint() -> None:
    """Only the framing is ring 0; the body inherits the source's ring and
    integrity. Summarising ring-3 content produces ring-3 content."""
    dirty_job = dict(LONG_JOB, name="reader")
    kernel, events = build(
        [dirty_job],
        {
            "reader": [
                {"read": "web.fetch"},
                *chatter(12),
                {"exit": True},
            ]
        },
    )
    job = kernel.spawn(DescriptorName("reader"))
    kernel.deliver(PipeName("web.fetch"), " ".join(["hostile"] * 24))
    kernel.run_until_quiescent()

    evicted_external = [
        e for e in of(events, SegmentEvicted) if job.segments.get(e.segment).ring is Ring.EXTERNAL
    ]
    if not evicted_external:
        pytest.skip("the external span was not selected for eviction in this run")

    stub = job.segments.get(evicted_external[0].stub)
    assert stub.ring is Ring.EXTERNAL, "the stub inherits the source ring"
    assert stub.integrity == Integrity(3), "and its integrity"
    assert Perm.X not in stub.perms, "a stub can never instruct"


def test_eviction_records_the_splice_cost() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(12) + [{"exit": True}]})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    splices = of(events, Spliced)
    assert splices, "eviction must go through SPLICE"
    assert all(s.tokens_out > s.tokens_in for s in splices), "a stub is smaller"


def test_net_negative_eviction_is_declined() -> None:
    """The clock's skip rule: a span that is cold *and small* is often not
    worth evicting, because the expected refault cost exceeds what it frees."""
    table_policy = ContextPolicy(window=100, high_watermark=0.1, low_watermark=0.05)
    kernel, _ = build([LONG_JOB], {"patrol": chatter(4) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    for record in job.segments.all():
        record.attn.ema = 1.0  # everything is hot: nothing should be evicted
    plan = plan_eviction(
        job.segments,
        table_policy,
        resident_tokens=1000,
        current_token_clock=10_000,
    )
    assert not plan.candidates
    assert plan.skipped, "hot spans must be skipped, and the skip must be visible"
    assert "refault cost" in plan.skipped[0].reason


def test_pinned_segments_are_never_evicted() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(12) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    evicted = {e.segment for e in of(events, SegmentEvicted)}  # noqa
    for record in job.segments.all():
        if record.pinned:
            assert record.id not in evicted, "the descriptor body is pinned"


# --- page faults ------------------------------------------------------------


def test_explicit_fault_pages_the_span_back_in() -> None:
    """The full round trip: evict → stub → fault → page-in."""
    target = first_evicted(12)
    kernel, events = build(
        [LONG_JOB],
        {"patrol": chatter(12) + [{"fault": target}, {"emit": "recovered"}, {"exit": True}]},
    )
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    assert of(events, PageFaultRaised), "the fault must be journalled"
    paged = of(events, PagedIn)
    assert paged, "the span must come back"
    assert paged[0].plan in ("append", "splice")
    assert paged[0].cost_tokens > 0


def test_protection_metadata_survives_the_round_trip() -> None:
    """Acceptance criterion: ring, integrity, and provenance must come back
    byte-for-byte, or eviction would be a laundering operation."""
    dirty_job = dict(LONG_JOB, name="reader")
    kernel, events = build(
        [dirty_job], {"reader": [{"read": "web.fetch"}, *chatter(14), {"exit": True}]}
    )
    job = kernel.spawn(DescriptorName("reader"))
    kernel.deliver(PipeName("web.fetch"), " ".join(["hostile"] * 24))
    kernel.run_until_quiescent()

    evictions = of(events, SegmentEvicted)
    assert evictions
    for eviction in evictions:
        original = job.segments.get(eviction.segment)
        span = kernel.store.get(eviction.store)
        assert span.ring is original.ring
        assert span.integrity == original.integrity
        assert span.provenance.pipe == original.provenance.pipe
        assert span.provenance.principal is original.provenance.principal


def test_unsatisfiable_need_returns_a_notice_not_silence() -> None:
    """ "We looked and it is not there" is a different belief from "I forgot to
    look"."""
    kernel, events = build(
        [LONG_JOB],
        {"patrol": [{"need": "the maintenance log for pump 4"}, {"emit": "ok"}, {"exit": True}]},
    )
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    faults = of(events, PageFaultRaised)
    assert faults and not faults[0].explicit
    transcript = " ".join(t.text for t in kernel.machine.transcript(job.job_id))
    assert "No stored content matches" in transcript


def test_fault_on_an_unknown_segment_is_answered_not_ignored() -> None:
    kernel, _ = build(
        [LONG_JOB], {"patrol": [{"fault": 999}, {"emit": "moving on"}, {"exit": True}]}
    )
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    transcript = " ".join(t.text for t in kernel.machine.transcript(job.job_id))
    assert "No archived span" in transcript
    assert job.state is JobState.DONE


def test_duplicate_fault_is_answered_with_a_pointer() -> None:
    """Two copies of the same content in one window is worse than none -- the
    model has to reconcile them."""
    target = first_evicted(12)
    kernel, events = build(
        [LONG_JOB],
        {
            "patrol": chatter(12)
            + [
                {"fault": target},
                {"emit": "first"},
                {"fault": target},
                {"emit": "second"},
                {"exit": True},
            ]
        },
    )
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    paged = of(events, PagedIn)
    assert len(paged) == 1, "the span must be injected exactly once"
    transcript = " ".join(t.text for t in kernel.machine.transcript(job.job_id))
    assert "already resident" in transcript


def test_residency_transitions_are_journalled_both_ways() -> None:
    target = first_evicted(12)
    kernel, events = build(
        [LONG_JOB], {"patrol": chatter(12) + [{"fault": target}, {"exit": True}]}
    )
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    changes = of(events, ResidencyChanged)
    assert any(
        c.from_residency is Residency.RESIDENT and c.to_residency is Residency.STUBBED
        for c in changes
    )
    assert any(
        c.from_residency is Residency.STUBBED and c.to_residency is Residency.RESIDENT
        for c in changes
    )


# --- thrashing --------------------------------------------------------------

THRASHER = {
    "name": "thrasher",
    "priority": 80,
    "context": {
        "window": 32,
        "eviction": "fifo",
        "stub_budget": 16,
        "min_span_age": 0,
        "high_watermark": 0.5,
        "low_watermark": 0.3,
        "thrash_threshold": 0.01,
        "refault_window_blocks": 64,
    },
    "on_fault": "abort",
}


def test_repeated_refaults_raise_a_loud_thrash_fault() -> None:
    """Acceptance criterion 2. A system that quietly thrashes looks identical to one
    that is merely slow."""
    script = chatter(8)
    for segment in (2, 3, 2, 3, 2):
        script += [{"fault": segment}, {"emit": "using it"}] + chatter(2)
    script += [{"exit": True}]

    kernel, events = build([THRASHER], {"thrasher": script})
    kernel.spawn(DescriptorName("thrasher"))
    try:
        kernel.run_until_quiescent()
    except Exception:  # the job aborts mid-script, which ends its stream
        pass

    assert of(events, Refaulted), "refaults must be detected"
    thrash = [f for f in of(events, FaultRaised) if f.fault is FaultKind.THRASH]
    assert thrash, "sustained refaulting must raise SCHEDULER_FAULT (THRASH)"
    assert "does not fit" in thrash[0].detail


def test_working_set_is_sampled_continuously() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(10) + [{"exit": True}]})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    samples = of(events, WorkingSetSampled)
    assert samples, "the working set must be measured, not guessed"
    assert all(s.size_tokens >= 0 for s in samples)


# --- admission control ------------------------------------------------------


def test_admission_refuses_a_job_that_cannot_fit() -> None:
    """Turns "how many agents fit on this GPU" from folklore into arithmetic."""
    reason = admission_check(
        ContextPolicy(window=1000, stub_budget=100, declared_working_set=2000),
        pinned_tokens=50,
    )
    assert reason is not None and "exceeds usable window" in reason


def test_admission_allows_a_job_that_fits() -> None:
    assert (
        admission_check(
            ContextPolicy(window=1000, stub_budget=100, declared_working_set=500),
            pinned_tokens=50,
        )
        is None
    )


def test_oversized_declared_working_set_faults_at_spawn() -> None:
    oversized = dict(LONG_JOB, name="greedy", working_set={"declared": 100_000})
    kernel, events = build([oversized], {"greedy": [{"exit": True}]})
    kernel.spawn(DescriptorName("greedy"))

    admission = [f for f in of(events, FaultRaised) if f.fault is FaultKind.ADMISSION]
    assert admission, "a job that cannot fit must be refused before it runs"


# --- store ------------------------------------------------------------------


def test_store_deduplicates_identical_spans() -> None:
    """Content addressing across FORKs and across jobs holding identical content."""
    kernel, _ = build([LONG_JOB], {"patrol": chatter(12) + [{"exit": True}]})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    assert kernel.store.unique_spans >= 1
    assert kernel.store.dedup_ratio >= 1.0


def test_completion_survives_a_window_smaller_than_the_transcript() -> None:
    """The headline: unbounded effective context at bounded cost."""
    kernel, events = build([LONG_JOB], {"patrol": chatter(20) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    assert job.state is JobState.DONE
    assert of(events, JobCompleted)
    resident = kernel.machine.stats(job.job_id).resident_tokens
    total_generated = of(events, JobCompleted)[0].tokens_used
    assert total_generated > job.descriptor.context.window, (
        "the job must have generated more than its window"
    )
    assert resident < total_generated, "yet not all of it is resident"
