# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Page faults: the mechanism that makes retrieval stop being the model's job.

Today's RAG puts recall inside the model's discretion -- the same mistake as polling,
in the other direction. The model must *decide* to look something up, pay a forward
pass to ask, and frequently does not know that it does not know. Under ZEOS a
missing-page reference blocks the job (costing nothing, by pipe semantics), the
kernel fetches, injects, and resumes.

Two fault kinds, both routed here:

* ``FAULT(segment)`` -- the model referenced a stub's handle. Cheap and exact.
* ``NEED(text)`` -- a structured request for content the model believes exists but
  cannot name. Resolved by search, and a NEED the pager cannot satisfy returns a
  kernel notice saying so -- which is itself valuable state, because "we looked and
  it isn't there" is a different belief from "I forgot to look".

**Placement matters as much as retrieval.** Appending the span at the current
boundary costs a prefill and invalidates nothing, and puts the content near the
point of need -- which is also where attention research says it is most usable.
Splicing it back into its original position costs the prefill *plus* everything
downstream, and is only worth it when downstream is short or is being recomputed
anyway.

In M0 the pager is kernel logic. In the design it is an ordinary descriptor --
scheduled, budgeted, and faulted like anything else -- and the escalation path for
*implicit* faults (a watcher that notices unresolved references) is explicitly an
open question (OQ-4), not something this implements.
"""

from __future__ import annotations

from dataclasses import dataclass

from zeos.core.ids import SegmentId, StoreId
from zeos.core.store import ArchivedSpan, EvictionRecord, SpanStore

__all__ = ["PageInPlan", "choose_plan", "Pager", "PagerResult"]


@dataclass(frozen=True, slots=True)
class PageInPlan:
    """Where a faulted-in span should land, and what it costs."""

    plan: str  # "append" | "splice"
    cost_tokens: int
    invalidated_downstream: int = 0
    rationale: str = ""


def choose_plan(
    *, span_tokens: int, downstream_tokens: int, resume_recompute: bool = False
) -> PageInPlan:
    """Append by default; splice only when it is genuinely cheaper.

    The comparison is ``s`` against ``s + d``: splicing is never cheaper in isolation,
    so it wins only when ``d`` is small enough that in-place coherence is worth the
    extra prefill, or when the downstream is being recomputed anyway (resume-time
    reconstruction), where ``d`` is already sunk.
    """
    if resume_recompute:
        return PageInPlan(
            plan="splice",
            cost_tokens=span_tokens + downstream_tokens,
            invalidated_downstream=downstream_tokens,
            rationale="resume is recomputing downstream regardless",
        )
    if downstream_tokens <= span_tokens // 4:
        return PageInPlan(
            plan="splice",
            cost_tokens=span_tokens + downstream_tokens,
            invalidated_downstream=downstream_tokens,
            rationale="downstream is short relative to the span",
        )
    return PageInPlan(
        plan="append",
        cost_tokens=span_tokens,
        rationale="append costs prefill only and invalidates nothing",
    )


@dataclass(frozen=True, slots=True)
class PagerResult:
    """What servicing a fault produced."""

    span: ArchivedSpan | None
    already_resident: SegmentId | None = None
    notice: str = ""

    @property
    def satisfied(self) -> bool:
        return self.span is not None or self.already_resident is not None


class Pager:
    """Resolves faults against the store. One per kernel in M0."""

    def __init__(self, store: SpanStore) -> None:
        self.store = store
        #: segment → what we know about its eviction, for refault accounting and
        #: for duplicate suppression.
        self.evictions: dict[SegmentId, EvictionRecord] = {}

    def record_eviction(self, record: EvictionRecord) -> None:
        self.evictions[record.segment] = record

    def resolve_fault(self, segment: SegmentId) -> PagerResult:
        """Service an explicit fault on a stub handle."""
        eviction = self.evictions.get(segment)
        if eviction is None:
            return PagerResult(
                span=None,
                notice=(
                    f"<KERNEL> No archived span for segment {segment}; the reference "
                    f"does not correspond to anything that was paged out. </KERNEL>"
                ),
            )
        if eviction.paged_in_segment is not None:
            # Duplicate suppression: answer with a pointer rather than
            # injecting the span twice. Two copies of the same content in one window
            # is worse than none -- the model has to reconcile them.
            return PagerResult(
                span=None,
                already_resident=eviction.paged_in_segment,
                notice=(
                    f"<KERNEL> Content of segment {segment} is already resident above "
                    f"at segment {eviction.paged_in_segment}. </KERNEL>"
                ),
            )
        return PagerResult(span=self.store.get(eviction.store_id))

    def resolve_need(self, text: str) -> PagerResult:
        """Service a NEED by searching the store.

        Naive token-overlap search. A real pager would use whatever retrieval the
        deployment already has; what matters structurally is that the *kernel*
        performs it while the job is descheduled, so the search costs the job
        nothing and cannot be forgotten.
        """
        wanted = {w.lower() for w in text.split() if len(w) > 2}
        if not wanted:
            return PagerResult(span=None, notice=self._miss_notice(text))

        best: ArchivedSpan | None = None
        best_score = 0
        for store_id in sorted(self._store_ids()):
            span = self.store.get(store_id)
            words = {t.text.lower() for t in span.tokens}
            score = len(wanted & words)
            if score > best_score:
                best, best_score = span, score

        if best is None:
            return PagerResult(span=None, notice=self._miss_notice(text))
        return PagerResult(span=best)

    @staticmethod
    def _miss_notice(text: str) -> str:
        """A NEED the pager cannot satisfy still returns state worth having."""
        return (
            f"<KERNEL> No stored content matches the request: {text!r}. "
            f"Proceed without it or state that it is unavailable. </KERNEL>"
        )

    def _store_ids(self) -> list[StoreId]:
        return [record.store_id for record in self.evictions.values()]

    def note_paged_in(self, segment: SegmentId, into: SegmentId) -> None:
        eviction = self.evictions.get(segment)
        if eviction is not None:
            eviction.paged_in_segment = into

    def note_refault(self, segment: SegmentId) -> None:
        eviction = self.evictions.get(segment)
        if eviction is not None:
            eviction.refaults += 1

    def blocks_since_eviction(self, segment: SegmentId, current_block: int) -> int | None:
        eviction = self.evictions.get(segment)
        if eviction is None:
            return None
        return current_block - eviction.evicted_at_block
