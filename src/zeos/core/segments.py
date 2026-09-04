# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Segments: the unit of protection, provenance, integrity, paging, and attention.

A segment is a contiguous, immutable-once-closed span of a job's token sequence.
The per-job segment table is **kernel state -- part of the PCB, not the context**.
The model cannot read it, cannot write it, and cannot forge it;
that separation is what makes provenance total rather than advisory.

Two invariants carry most of the weight:

**Block alignment by construction.** The kernel closes the open segment and
pads to a block boundary before any operation that needs a clean edge -- a new
source being injected, an eviction, a masking change. Padding costs at most
``block_size - 1`` tokens per boundary event, and boundary events are kernel-
scheduled, so the overhead is bounded and measurable. What it buys is that a
segment maps *exactly* onto a set of KV blocks, which is what lets attention masks
work at block granularity and makes per-segment attention accounting exact rather
than approximate.

**INJECT is the only entry path for foreign tokens.** Every segment that did
not come from the model's own decoding arrived through an INJECT that named its
source pipe. Provenance is therefore total and automatic, with no instrumentation
of the model required -- and no way for content to arrive unattributed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import (
    Integrity,
    Perm,
    PipeName,
    Principal,
    Residency,
    Ring,
    SegmentId,
    SegmentState,
)

__all__ = [
    "Provenance",
    "Attention",
    "SegmentRecord",
    "SegmentTable",
    "TAG_SELF",
    "TAG_DESCRIPTOR",
    "map_tag",
]

#: Source tags. A tag names *where content came from* in a form a case author can
#: write down -- a pipe name, or one of these two special sources. The kernel
#: resolves attention hints against them.
TAG_SELF = "self"  # the job's own generated output
TAG_DESCRIPTOR = "descriptor"  # the ring-1 body loaded at spawn


def map_tag(obj: object) -> str:
    """The tag every status region for ``obj`` carries.

    A function rather than an f-string at each use site: the tag is how the kernel
    finds the view it is about to supersede, so a mismatch between the writer and the
    finder would silently turn rewriting into appending.
    """
    return f"map:{obj}"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a segment's tokens came from, and what they were derived from.

    ``derived_from`` links a segment to its inputs: a stub to the segment it
    summarises, an endorsement to the segments the endorser read, a faulted-in span
    to its archived original. The chains are append-only, which is what lets the
    kernel answer "which external inputs could have influenced this effect?" as a
    hard fact at segment granularity.
    """

    pipe: PipeName
    principal: Principal
    injected_at: int  # token_clock
    tag: str = TAG_SELF
    derived_from: tuple[SegmentId, ...] = ()


@dataclass
class Attention:
    """Per-segment reference signal -- the accessed bit upgraded to a float.

    A classical MMU guesses at usage through one bit. A transformer tells us
    exactly what it is using, so eviction policy and integrity demotion get better
    raw material than Denning ever had. In M0 the numbers are synthetic;
    the accounting around them is not.
    """

    ema: float = 0.0
    last_touch: int = 0
    #: Mass accumulated since the last block boundary, folded into the EMA there.
    pending: float = 0.0

    def accumulate(self, mass: float, *, token_clock: int) -> None:
        self.pending += mass
        if mass > 0:
            self.last_touch = token_clock

    def fold(self, tau_blocks: float) -> float:
        """Fold pending mass into the EMA at a block boundary. Returns the mass."""
        alpha = 1.0 / max(tau_blocks, 1.0)
        mass = self.pending
        self.ema = (1.0 - alpha) * self.ema + alpha * mass
        self.pending = 0.0
        return mass


@dataclass
class SegmentRecord:
    """One segment. Mutable: perms, residency, and attention all change over time."""

    id: SegmentId
    start: int
    end: int
    ring: Ring
    integrity: Integrity
    perms: Perm
    provenance: Provenance
    state: SegmentState = SegmentState.OPEN
    residency: Residency = Residency.RESIDENT
    stub_id: SegmentId | None = None
    attn: Attention = field(default_factory=Attention)

    @property
    def tokens(self) -> int:
        return max(0, self.end - self.start)

    @property
    def is_closed(self) -> bool:
        return self.state is SegmentState.CLOSED

    @property
    def readable(self) -> bool:
        """Attendable, and actually present. A stubbed segment's content is gone
        from the window even though its record remains."""
        return Perm.R in self.perms and self.residency is Residency.RESIDENT

    @property
    def pinned(self) -> bool:
        return Perm.P in self.perms

    @property
    def directive(self) -> bool:
        """X: may direct rather than merely inform. Advisory to the model, binding
        at the boundary."""
        return Perm.X in self.perms

    def describe(self) -> str:
        return (
            f"segment {self.id} [{self.start},{self.end}) ring={self.ring.name} "
            f"integrity={self.integrity} perms={self.perms.render()} "
            f"via={self.provenance.pipe}"
        )


class SegmentTable:
    """The per-job segment table.

    Owns segment identity and ranges. It does *not* own the tokens -- the machine
    does -- and it does not decide policy; it records structure and answers questions
    about it.
    """

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self._segments: dict[SegmentId, SegmentRecord] = {}
        self._order: list[SegmentId] = []
        self._open: SegmentId | None = None

    # -- structure -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._segments)

    def __contains__(self, segment_id: SegmentId) -> bool:
        return segment_id in self._segments

    def get(self, segment_id: SegmentId) -> SegmentRecord:
        record = self._segments.get(segment_id)
        if record is None:
            raise KeyError(f"unknown segment {segment_id}")
        return record

    def all(self) -> tuple[SegmentRecord, ...]:
        """Segments in virtual order -- the order they occupy in the sequence."""
        return tuple(self._segments[s] for s in self._order)

    def resident(self) -> tuple[SegmentRecord, ...]:
        return tuple(s for s in self.all() if s.residency is Residency.RESIDENT)

    @property
    def open_segment(self) -> SegmentRecord | None:
        return None if self._open is None else self._segments[self._open]

    def by_tag(self, tag: str) -> tuple[SegmentRecord, ...]:
        """Segments whose content came from a named source.

        The kernel resolves attention hints through this, which is why the tag is
        stored rather than recomputed: after eviction and page-in a segment's pipe
        binding is still what it originally arrived on.
        """
        return tuple(s for s in self.all() if s.provenance.tag == tag)

    def tags(self) -> tuple[str, ...]:
        return tuple(sorted({s.provenance.tag for s in self.all()}))

    # -- the five ops --------------------------------------------------------

    def open_output(
        self,
        segment_id: SegmentId,
        *,
        at: int,
        ring: Ring,
        integrity: Integrity,
        token_clock: int,
        perms: Perm = Perm.R | Perm.X,
    ) -> SegmentRecord:
        """Start a fresh output segment for DECODE to extend.

        Output is ring 2 when the job's code ring is ≤ 2, else ring 3; the
        caller decides, because only it knows the job.
        """
        if self._open is not None:
            raise RuntimeError(f"segment {self._open} is still open")
        record = SegmentRecord(
            id=segment_id,
            start=at,
            end=at,
            ring=ring,
            integrity=integrity,
            perms=perms,
            provenance=Provenance(
                pipe=PipeName("self"),
                principal=Principal.PEER_JOB,
                injected_at=token_clock,
                tag=TAG_SELF,
            ),
        )
        self._segments[segment_id] = record
        self._order.append(segment_id)
        self._open = segment_id
        return record

    def extend_open(self, tokens: int) -> SegmentRecord | None:
        """DECODE extends the single OPEN segment."""
        if self._open is None or tokens <= 0:
            return None
        record = self._segments[self._open]
        record.end += tokens
        return record

    def close_open(self, *, integrity: Integrity | None = None) -> SegmentRecord | None:
        """Close the output segment.

        The kernel closes output at every scheduling boundary event -- any INJECT,
        any returning pipe read, any preemption, any block-boundary refresh.
        The result is that provenance for generated text is "generated by job J
        between events e₁ and e₂", which is the finest granularity defensible
        without per-token attribution.
        """
        if self._open is None:
            return None
        record = self._segments[self._open]
        record.state = SegmentState.CLOSED
        if integrity is not None:
            # Conservative whole-output rule: everything the job attended could have
            # influenced everything it wrote. Finer attention-based
            # attribution is explicitly not trusted for enforcement.
            record.integrity = integrity
        self._open = None
        return record

    def inject(
        self,
        segment_id: SegmentId,
        *,
        start: int,
        end: int,
        ring: Ring,
        integrity: Integrity,
        provenance: Provenance,
        perms: Perm | None = None,
    ) -> SegmentRecord:
        """Record a CLOSED segment for injected foreign tokens.

        Default permissions encode the X-bit discipline: ring-3 content arrives
        **without X**. It may inform beliefs; imperative content within it has no
        standing.
        """
        if self._open is not None:
            raise RuntimeError("close the open segment before injecting")
        if perms is None:
            perms = Perm.R if ring is Ring.EXTERNAL else Perm.R | Perm.X
        record = SegmentRecord(
            id=segment_id,
            start=start,
            end=end,
            ring=ring,
            integrity=integrity,
            perms=perms,
            provenance=provenance,
            state=SegmentState.CLOSED,
        )
        self._segments[segment_id] = record
        self._order.append(segment_id)
        return record

    def trunc(self, at: int) -> tuple[SegmentId, ...]:
        """Drop every segment starting at or after ``at``."""
        dropped: list[SegmentId] = []
        for segment_id in list(self._order):
            record = self._segments[segment_id]
            if record.start >= at:
                dropped.append(segment_id)
                self._order.remove(segment_id)
                del self._segments[segment_id]
                if self._open == segment_id:
                    self._open = None
            elif record.end > at:
                record.end = at
        return tuple(dropped)

    def fork_into(self, other: SegmentTable, *, id_offset: int = 0) -> int:
        """Copy the segment table for a FORK. Returns the number of segments shared.

        Copy-on-write in a real backend shares KV blocks and archived spans; here it
        is the *records* that matter, because they carry ring, integrity, and
        provenance -- which must survive forking exactly, or a compartment could
        launder taint simply by being forked.
        """
        import copy

        for segment_id in self._order:
            record = copy.deepcopy(self._segments[segment_id])
            new_id = SegmentId(int(record.id) + id_offset)
            record.id = new_id
            other._segments[new_id] = record  # pyright: ignore[reportPrivateUsage]
            other._order.append(new_id)  # pyright: ignore[reportPrivateUsage]
        other._open = None  # pyright: ignore[reportPrivateUsage]
        return len(self._order)

    def evict_to_stub(
        self,
        record_id: SegmentId,
        stub_id: SegmentId,
        *,
        stub_tokens: int,
        token_clock: int,
    ) -> SegmentRecord:
        """Replace a resident span with a stub, renumbering what follows.

        Position encodings are assigned over the **resident projection**, not over
        virtual offsets, so closing the gap left by an evicted span shifts
        everything downstream -- which is exactly why eviction batches at boundaries
        and prefers spans with little after them.

        The stub inherits the source's ring and integrity. Summarisation does not
        launder taint: only the framing is ring 0, and the kernel says so in the
        stub text itself.
        """
        record = self.get(record_id)
        if record.residency is not Residency.RESIDENT:
            raise RuntimeError(f"segment {record_id} is not resident")

        delta = stub_tokens - record.tokens
        stub = SegmentRecord(
            id=stub_id,
            start=record.start,
            end=record.start + stub_tokens,
            ring=record.ring,
            integrity=record.integrity,
            perms=Perm.R,  # readable, never directive -- a stub cannot instruct
            provenance=Provenance(
                pipe=record.provenance.pipe,
                principal=record.provenance.principal,
                injected_at=token_clock,
                tag=record.provenance.tag,
                derived_from=(record_id,),
            ),
            state=SegmentState.CLOSED,
        )

        position = self._order.index(record_id)
        self._order[position] = stub_id
        self._segments[stub_id] = stub

        record.residency = Residency.STUBBED
        record.stub_id = stub_id

        for following in self._order[position + 1 :]:
            other = self._segments[following]
            other.start += delta
            other.end += delta

        return stub

    def retract(self, record_id: SegmentId) -> int:
        """Remove a resident span outright, renumbering what follows. Returns tokens freed.

        The counterpart to ``evict_to_stub``, for content the kernel can *regenerate*
        rather than recall. Eviction leaves a stub because an evicted span might be
        wanted again and the only way back is a fault; a retracted span has no way
        back and needs none -- a superseded status region is ``world.get(obj)``, which
        costs nothing to render again and is already in the window in its current form.

        Paying ``STUB_FRAMING_TOKENS`` to remember such a value is not merely wasteful
        but self-defeating: ``stub_size`` exceeds the span for anything under about
        eleven tokens, so a status region handed to ``plan_eviction`` is declined for
        the life of the job and its tokens are never reclaimed at all.

        No record survives. Provenance is not lost with it -- the journal holds the
        ``Injected`` that created the span and the ``Spliced`` that removed it, and the
        journal is the source of truth.
        """
        record = self.get(record_id)
        if record.residency is not Residency.RESIDENT:
            raise RuntimeError(f"segment {record_id} is not resident")
        if not record.is_closed:
            raise RuntimeError(f"segment {record_id} is still open; close it first")

        freed = record.tokens
        position = self._order.index(record_id)
        del self._order[position]
        del self._segments[record_id]

        # ``position`` now indexes the first *following* segment, the deletion having
        # closed the gap.
        for following in self._order[position:]:
            other = self._segments[following]
            other.start -= freed
            other.end -= freed

        return freed

    def reinstate(self, record_id: SegmentId, *, into: SegmentId) -> None:
        """Mark a stubbed segment as paged back in at ``into``."""
        record = self.get(record_id)
        record.residency = Residency.RESIDENT
        record.stub_id = None
        _ = into

    def stubbed(self) -> tuple[SegmentRecord, ...]:
        return tuple(s for s in self._segments.values() if s.residency is Residency.STUBBED)

    def stub_tokens(self) -> int:
        """Resident tokens currently spent on stubs -- a budgeted resource."""
        return sum(
            s.tokens
            for s in self.all()
            if s.provenance.derived_from and s.residency is Residency.RESIDENT
        )

    # -- masking -------------------------------------------------------------

    def allowed_blocks(self) -> frozenset[int]:
        """The allowed-block bitmap: the union of KV blocks of RESIDENT segments
        with R.

        This is the MMU. Enforcement happens per forward pass in the serving stack
        and involves no model cooperation, which is precisely what distinguishes it
        from asking the model nicely not to look.
        """
        blocks: set[int] = set()
        for record in self._segments.values():
            if not record.readable or record.tokens <= 0:
                continue
            blocks.update(self.blocks_for(record))
        return frozenset(blocks)

    def blocks_for(self, record: SegmentRecord) -> frozenset[int]:
        if record.tokens <= 0:
            return frozenset()
        first = record.start // self.block_size
        last = (record.end - 1) // self.block_size
        return frozenset(range(first, last + 1))

    def denied_blocks(self) -> frozenset[int]:
        every: set[int] = set()
        for record in self._segments.values():
            every.update(self.blocks_for(record))
        return frozenset(every) - self.allowed_blocks()

    def grant(self, segment_ids: Iterable[SegmentId], perms: Perm) -> None:
        for segment_id in segment_ids:
            record = self.get(segment_id)
            record.perms |= perms

    def revoke(self, segment_ids: Iterable[SegmentId], perms: Perm) -> None:
        """Drop permission bits.

        Revocation is **prospective**: the blocks leave the bitmap at the next
        boundary, but the segment's influence on tokens already generated of course
        remains. The design pattern is to grant dirty reads to
        compartments rather than to long-lived jobs.
        """
        for segment_id in segment_ids:
            record = self.get(segment_id)
            record.perms &= ~perms

    # -- attention -----------------------------------------------------------

    def accumulate_attention(
        self, mass_by_segment: Sequence[tuple[SegmentId, float]], *, token_clock: int
    ) -> None:
        for segment_id, mass in mass_by_segment:
            record = self._segments.get(segment_id)
            if record is not None:
                record.attn.accumulate(mass, token_clock=token_clock)

    def fold_attention(self, tau_blocks: float) -> dict[SegmentId, float]:
        """Fold pending mass into EMAs at a block boundary; return this block's mass."""
        return {s.id: s.attn.fold(tau_blocks) for s in self.all()}

    def working_set(self, threshold: float) -> tuple[SegmentRecord, ...]:
        """W(τ) = segments whose attention EMA exceeds θ_ws."""
        return tuple(s for s in self.all() if s.attn.ema > threshold)
