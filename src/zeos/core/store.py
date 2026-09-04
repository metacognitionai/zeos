# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The backing store: append-only, content-addressed spans.

``store_id = hash(tokens)``, with the journal mapping ``(job, segment) → store_id``.
Content addressing buys deduplication for free -- across FORKs, and
across jobs holding identical content -- which matters more than it sounds, because
the common case for a fleet is many jobs holding the same org preamble, the same
reference material, and the same safety rules.

The critical property is the one that is easy to get wrong: **archived spans carry
their full segment record.** Ring, integrity, and provenance survive eviction and
reinstatement byte-for-byte. If they did not, eviction would be a laundering
operation -- evict a ring-3 span, page it back in, and watch it return clean -- which
would silently defeat every guarantee Protected Mode makes.

Stdlib-only and pure: this is an in-memory store with the interface a persistent one
would have. Durability is the driver's problem, not the kernel's.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from zeos.core.ids import Integrity, Residency, Ring, SegmentId, StoreId
from zeos.core.segments import Provenance, SegmentRecord
from zeos.machine.base import Token

__all__ = ["ArchivedSpan", "SpanStore"]


@dataclass(frozen=True, slots=True)
class ArchivedSpan:
    """A span in the store, with everything needed to reinstate it exactly."""

    store_id: StoreId
    tokens: tuple[Token, ...]
    ring: Ring
    integrity: Integrity
    provenance: Provenance
    #: The segment this span was evicted from, for refault accounting and for the
    #: derivation chain a reinstated span carries.
    origin: SegmentId

    @property
    def length(self) -> int:
        return len(self.tokens)


class SpanStore:
    """Content-addressed span storage with reference counting.

    Reference counts exist so that dedup is observable: two jobs archiving the same
    reference document produce one entry with a count of two, which is the number a
    capacity planner actually wants.
    """

    def __init__(self) -> None:
        self._spans: dict[StoreId, ArchivedSpan] = {}
        self._refs: dict[StoreId, int] = {}
        self._puts = 0

    @staticmethod
    def address(tokens: Sequence[Token]) -> StoreId:
        """Content address. Includes token kind, so a padding token and a word that
        renders the same do not collide."""
        digest = hashlib.blake2b(digest_size=16)
        for token in tokens:
            digest.update(token.kind.value.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(token.text.encode("utf-8"))
            digest.update(b"\x01")
        return StoreId(digest.hexdigest())

    def put(self, record: SegmentRecord, tokens: Sequence[Token]) -> ArchivedSpan:
        """Archive a span. Idempotent by content: identical spans share one entry."""
        store_id = self.address(tokens)
        existing = self._spans.get(store_id)
        self._puts += 1
        if existing is not None:
            self._refs[store_id] += 1
            return existing
        span = ArchivedSpan(
            store_id=store_id,
            tokens=tuple(tokens),
            ring=record.ring,
            integrity=record.integrity,
            provenance=record.provenance,
            origin=record.id,
        )
        self._spans[store_id] = span
        self._refs[store_id] = 1
        return span

    def get(self, store_id: StoreId) -> ArchivedSpan:
        span = self._spans.get(store_id)
        if span is None:
            raise KeyError(f"no archived span {store_id!r}")
        return span

    def has(self, store_id: StoreId) -> bool:
        return store_id in self._spans

    def refs(self, store_id: StoreId) -> int:
        return self._refs.get(store_id, 0)

    @property
    def unique_spans(self) -> int:
        return len(self._spans)

    @property
    def total_puts(self) -> int:
        return self._puts

    @property
    def dedup_ratio(self) -> float:
        """Puts per unique span. 1.0 means no sharing at all."""
        return self._puts / self.unique_spans if self._spans else 1.0

    def total_tokens(self) -> int:
        return sum(span.length for span in self._spans.values())


@dataclass
class EvictionRecord:
    """What the kernel remembers about an eviction, for refault accounting."""

    segment: SegmentId
    store_id: StoreId
    stub: SegmentId
    evicted_at_block: int
    freed_tokens: int
    stub_tokens: int
    residency: Residency = Residency.STUBBED
    refaults: int = 0
    #: Set once the span has been paged back in and is resident again, so that
    #: duplicate suppression can answer a second fault with a pointer.
    paged_in_segment: SegmentId | None = None
    derived: tuple[SegmentId, ...] = field(default_factory=tuple)
