# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Eviction, stubs, working sets, and thrash detection.

The window is a cache, not the memory. When residency pressure demands it, the
kernel evicts a cold span to the store and leaves a **stub** in its place -- a
summary *plus a page handle*, which is what distinguishes this from compaction.
Compaction is eviction without an address: the summary replaces the content with no
way back. A stub can be faulted in.

Three things here are easy to get subtly wrong, and each has a rule:

**Stubs do not launder taint.** The framing is ring 0 because the kernel wrote it,
but the summary body inherits the *source* segment's ring and integrity.
Summarising ring-3 content produces ring-3 content. Only an endorser raises
integrity, and a summarizer is not an endorser.

**Eviction can be net-negative.** Replacing an `s`-token span with a `σ`-token stub
frees `s - σ` window tokens, but if the job faults it back in, that cost is
`s/p` prefill plus the fault latency. The clock's skip rule declines evictions
whose expected refault cost exceeds what they free -- which is why a span that is
cold *and small* is often not worth evicting at all.

**A throttled or failed eviction must be visible.** Thrashing is a loud
``SCHEDULER_FAULT``, not a silent churn of SPLICE recomputes. A system that
quietly thrashes looks identical to one that is merely slow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from zeos.core.ids import EvictionPolicy, Perm
from zeos.core.segments import SegmentRecord, SegmentTable

__all__ = [
    "ContextPolicy",
    "EvictionCandidate",
    "EvictionPlan",
    "plan_eviction",
    "render_stub",
    "stub_size",
    "STUB_FRAMING_TOKENS",
    "working_set_tokens",
    "ThrashMonitor",
    "admission_check",
    "DEFAULT_THETA_WS",
]

#: Attention EMA above which a segment counts as part of the working set.
DEFAULT_THETA_WS = 0.05

#: Tokens the stub framing costs before any summary text -- the
#: ``<STUB id=... segment=... ring=... integrity=... tokens=...>`` header and
#: its closing tag. It is a floor on stub size, and therefore a floor on how
#: small a span can usefully be: evicting a span barely larger than its own
#: framing frees nothing and costs a refault.
STUB_FRAMING_TOKENS = 9


def stub_size(span_tokens: int) -> tuple[int, int]:
    """Return ``(total_stub_tokens, summary_budget)`` for a span.

    The planner and the executor must agree on this number. They did not in the
    first cut -- the planner assumed a stub a quarter the size of the span while
    the executor emitted framing plus summary -- with the result that every
    "eviction" made the window *larger* and the job evicted itself in circles.
    A single shared function is the fix, and the reason it is exported.
    """
    summary_budget = max(2, span_tokens // 8)
    return STUB_FRAMING_TOKENS + summary_budget, summary_budget


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """A descriptor's ``context:`` block."""

    window: int = 4096
    eviction: EvictionPolicy = EvictionPolicy.ATTENTION_CLOCK
    #: Working-set horizon in blocks, for the attention EMA.
    tau_blocks: float = 8.0
    #: Total resident tokens spent on stubs. A first-class resource, like the HBM
    #: pin budget -- stubs are not free, they are cheaper.
    stub_budget: int = 512
    #: Minimum age (in tokens) before a segment may be evicted. Stops the kernel
    #: evicting content the job is still mid-thought about.
    min_span_age: int = 32
    #: High/low watermarks as fractions of the window.
    high_watermark: float = 0.9
    low_watermark: float = 0.7
    theta_ws: float = DEFAULT_THETA_WS
    declared_working_set: int | None = None
    on_thrash: str = "fault"
    #: Refault rate (per block) above which thrashing is declared.
    thrash_threshold: float = 0.5
    #: A fault on a span evicted within this many blocks counts as a refault.
    refault_window_blocks: int = 16
    #: Downstream recompute the kernel will accept per token of window *permanently*
    #: reclaimed, when retracting a superseded status region (see
    #: ``Kernel._retire_status_region``). The two sides are not commensurable: a
    #: retraction invalidates KV once, while carrying the stale copy costs window for
    #: the life of the job -- so the ratio expresses how many tokens of one-off
    #: prefill a token of permanent window is worth. Crude, and deliberately generous:
    #: the alternative to retracting is not eviction but accumulation, because a span
    #: smaller than ``STUB_FRAMING_TOKENS`` can never be evicted at a profit. Its
    #: calibration is exactly what E3 exists to measure, like ``refault_penalty``.
    retract_recompute_ratio: float = 32.0

    @property
    def high_mark(self) -> int:
        return int(self.window * self.high_watermark)

    @property
    def low_mark(self) -> int:
        return int(self.window * self.low_watermark)


@dataclass(frozen=True, slots=True)
class EvictionCandidate:
    record: SegmentRecord
    stub_tokens: int
    freed: int
    refault_risk: float
    net_positive: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class EvictionPlan:
    """A batch of evictions, planned for the same block boundary.

    Batched because every SPLICE invalidates downstream KV, so planning them
    together and executing at one boundary is what keeps the cost bounded.
    """

    candidates: tuple[EvictionCandidate, ...] = ()
    skipped: tuple[EvictionCandidate, ...] = ()
    resident_before: int = 0
    target: int = 0

    @property
    def total_freed(self) -> int:
        return sum(c.freed for c in self.candidates)

    def __bool__(self) -> bool:
        return bool(self.candidates)


def render_stub(record: SegmentRecord, summary: str, *, store_id: str) -> str:
    """The in-window stand-in for an evicted span.

    The framing is ring 0 and unforgeable; the body inherits the source's ring and
    integrity, and *says so*, so that a model reading a stub can see it is reading
    summarised untrusted content rather than a kernel statement of fact.
    """
    return (
        f"<STUB id={store_id[:8]} segment={record.id} ring={record.ring.name} "
        f"integrity={record.integrity} tokens={record.tokens}> "
        f"{summary} "
        f"</STUB>"
    )


def summarise(tokens: Sequence[object], *, budget: int) -> str:
    """A deterministic stand-in for the kernel summarizer job.

    The real summarizer is an ordinary descriptor -- scheduled, budgeted, and faulted
    like anything else -- and it is what produces a stub worth reading. This
    placeholder exists so eviction can be exercised without a model, and it is
    deliberately dumb rather than plausible: a fake-good summary would make the
    stub-fidelity question (VM OQ-1, benchmark E2) look answered when it is not.
    """
    words = [str(t) for t in tokens]
    head = " ".join(words[: max(1, budget)])
    if len(words) > budget:
        head += f" ... (+{len(words) - budget} tokens elided)"
    return head


def plan_eviction(
    table: SegmentTable,
    policy: ContextPolicy,
    *,
    resident_tokens: int,
    current_token_clock: int,
    refault_penalty: float = 1.0,
) -> EvictionPlan:
    """Choose what to evict, coldest first, skipping net-negative evictions.

    Returns an empty plan when the job is under its high watermark -- eviction is
    triggered by pressure, not run continuously.
    """
    if resident_tokens <= policy.high_mark:
        return EvictionPlan(resident_before=resident_tokens, target=policy.low_mark)
    if policy.eviction is EvictionPolicy.PINNED_ONLY:
        return EvictionPlan(resident_before=resident_tokens, target=policy.low_mark)

    eligible = [
        record
        for record in table.resident()
        if record.is_closed
        and not record.pinned
        and Perm.W not in record.perms  # status regions are refreshed, not evicted
        and record.tokens > 0
        and (current_token_clock - record.provenance.injected_at) >= policy.min_span_age
    ]

    if policy.eviction is EvictionPolicy.FIFO:
        ordered = sorted(eligible, key=lambda r: (r.start, int(r.id)))
    else:
        # Attention clock: coldest first. Ties break on position so the plan is
        # deterministic even when every EMA is identical (which it is at startup).
        ordered = sorted(eligible, key=lambda r: (r.attn.ema, r.start, int(r.id)))

    chosen: list[EvictionCandidate] = []
    skipped: list[EvictionCandidate] = []
    projected = resident_tokens

    for record in ordered:
        if projected <= policy.low_mark:
            break
        stub_tokens, _ = stub_size(record.tokens)
        freed = record.tokens - stub_tokens
        # P(refault) estimated from the EMA: a warm span is likely to be wanted
        # again. Crude, and labelled as such -- it is the clock's skip rule, and its
        # calibration is exactly what E3 exists to measure on real attention.
        risk = min(1.0, record.attn.ema)
        #
        # Both sides are **token counts**: window tokens freed now, against tokens
        # expected to be re-prefilled later. The spec writes the right-hand side as
        # ``P(refault) · s/p``, which mixes tokens with time and cannot be
        # compared to a token count directly; dividing by throughput belongs in a
        # latency budget, not in a space-versus-space decision. ``refault_penalty``
        # is where the extra cost of a refault beyond raw prefill -- fault latency,
        # and the stub still occupying window -- is expressed.
        expected_refault_tokens = risk * record.tokens * refault_penalty
        net_positive = freed > 0 and freed > expected_refault_tokens

        candidate = EvictionCandidate(
            record=record,
            stub_tokens=stub_tokens,
            freed=freed,
            refault_risk=risk,
            net_positive=net_positive,
            reason=(
                ""
                if net_positive
                else (
                    f"expected refault cost {expected_refault_tokens:.1f} tokens "
                    f"exceeds the {freed} freed"
                )
            ),
        )
        if net_positive:
            chosen.append(candidate)
            projected -= freed
        else:
            skipped.append(candidate)

    return EvictionPlan(
        candidates=tuple(chosen),
        skipped=tuple(skipped),
        resident_before=resident_tokens,
        target=policy.low_mark,
    )


def working_set_tokens(table: SegmentTable, theta_ws: float) -> tuple[int, int]:
    """W(τ): (tokens, segment count) currently receiving attention."""
    members = table.working_set(theta_ws)
    return sum(m.tokens for m in members), len(members)


@dataclass
class ThrashMonitor:
    """Refault accounting, and the loud fault it eventually raises.

    Thrashing is defined as faults on spans evicted within the last F blocks,
    exceeding a threshold per block. The definition matters: a job that faults in
    something evicted an hour ago is using its memory hierarchy correctly, while one
    that faults in something evicted four blocks ago is paying SPLICE costs for
    nothing.
    """

    policy: ContextPolicy
    refaults: int = 0
    blocks_observed: int = 0
    recent: list[int] = field(default_factory=list[int])

    def note_block(self) -> None:
        self.blocks_observed += 1

    def note_refault(self, *, at_block: int) -> None:
        self.refaults += 1
        self.recent.append(at_block)
        cutoff = at_block - self.policy.refault_window_blocks
        self.recent = [b for b in self.recent if b > cutoff]

    def rate(self) -> float:
        window = max(1, min(self.blocks_observed, self.policy.refault_window_blocks))
        return len(self.recent) / window

    def is_thrashing(self) -> bool:
        # Require a couple of refaults before declaring it: one refault is a fact of
        # life, and a threshold that fires on the first would be noise.
        return len(self.recent) >= 2 and self.rate() > self.policy.thrash_threshold


def admission_check(
    policy: ContextPolicy, *, pinned_tokens: int, observed_ws: int | None = None
) -> str | None:
    """Refuse to dispatch a job whose working set will not fit.

    This is the arithmetic that turns "how many agents fit on this GPU" from folklore
    into a number. Returns a refusal reason, or None.
    """
    declared = policy.declared_working_set if observed_ws is None else observed_ws
    if declared is None:
        return None
    usable = policy.window - pinned_tokens - policy.stub_budget
    if declared > usable:
        return (
            f"working set {declared} exceeds usable window {usable} "
            f"(window {policy.window} - pins {pinned_tokens} - stub budget "
            f"{policy.stub_budget})"
        )
    return None
