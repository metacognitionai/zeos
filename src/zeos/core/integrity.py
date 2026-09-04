# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Biba low-water-mark integrity, and the escape hatches that make it livable.

The rule is one line -- *reading dirt makes you dirty; dirty jobs can't touch clean
switches* -- and everything interesting is in the qualifications.

**Attention-thresholded demotion.** At each block boundary the kernel
demotes against the segments *meaningfully attended* since the last boundary, where
"meaningfully" means attention mass ≥ θ_read. Merely *containing* dirt does not
demote; *using* it does. This matters because a long-lived job accumulates hostile
content in its window as a matter of course, and a containment-based rule would
have every such job terminate fully demoted within minutes.

For content that must not even be visible, the R-mask is the fallback -- masking
controls visibility, integrity controls authority, and conflating them is
the most common way to get this wrong.

**Monotone decay is unusable without escape hatches**, so there are three,
in preference order:

1. **Compartment children** -- the parent spawns a low-integrity child with an
   R-grant on just the dirty segments; the child returns a result by pipe and the
   parent's watermark never moves. Cheap, because children and pipes already exist.
2. **Endorsers** -- the only integrity-*raising* operation, and therefore the only
   thing here that can be got badly wrong. An endorser emits under a constrained
   schema, so the schema's width *is* the injection bandwidth.
3. **FORK-and-discard** -- checkpoint before the dirty read, extract what is needed,
   discard the tainted branch.

θ_read is a kernel parameter with a conservative default. Its sensitivity is OQ-7,
and nothing in an M0 run can answer it, because the attention driving it is
synthetic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zeos.core.ids import Integrity, SegmentId
from zeos.core.segments import SegmentTable

__all__ = [
    "DEFAULT_THETA_READ",
    "Demotion",
    "demote_for_boundary",
    "effective_integrity",
    "can_write_at",
]

#: Attention mass at or above which a segment counts as *used* rather than merely
#: present. Conservative default: a segment receiving a fifth of a block's attention
#: is being read, not incidentally resident. A policy dial (OQ-7).
DEFAULT_THETA_READ = 0.2


@dataclass(frozen=True, slots=True)
class Demotion:
    """A watermark movement, with the evidence that caused it.

    ``because`` is not decoration. When a write is later blocked by a privilege
    fault, the first question is always "why is this job dirty?", and the journal
    can answer it exactly -- which segments dragged it down, and via which pipes they
    arrived.
    """

    before: Integrity
    after: Integrity
    because: tuple[SegmentId, ...]

    @property
    def moved(self) -> bool:
        return self.after != self.before


def demote_for_boundary(
    current: Integrity,
    *,
    table: SegmentTable,
    mass_this_block: Mapping[SegmentId, float],
    theta_read: float = DEFAULT_THETA_READ,
) -> Demotion:
    """Apply the low-water-mark rule for one block boundary.

    ``current_integrity = max(current_integrity, max over attended s of s.integrity)``

    ``max`` is *worse* in this numbering -- 0 is most trusted -- which is why the
    convention is shared with rings and priorities. Reading it as "the worst thing
    you meaningfully looked at" is the whole rule.
    """
    attended: list[SegmentId] = []
    worst = int(current)
    for segment_id, mass in sorted(mass_this_block.items()):
        if mass < theta_read:
            continue
        record = table.get(segment_id) if segment_id in table else None
        if record is None or not record.readable:
            # A masked segment cannot demote: it was never attended, because the
            # bitmap excluded its blocks from the forward pass entirely.
            continue
        if int(record.integrity) > worst:
            worst = int(record.integrity)
        if int(record.integrity) > int(current):
            attended.append(segment_id)

    return Demotion(before=current, after=Integrity(worst), because=tuple(attended))


def effective_integrity(current: Integrity, session_floor: Integrity | None) -> Integrity:
    """The confused-deputy rule.

    A job serving requests from a pipe carries that pipe's integrity as a floor for
    the duration of handling one request. A high-trust service answering a low-trust
    requester therefore writes at the *requester's* integrity -- seteuid-drop
    semantics, with no descriptor changes and no cooperation from the service.
    """
    if session_floor is None:
        return current
    return Integrity(max(int(current), int(session_floor)))


def can_write_at(effective: Integrity, required: Integrity) -> bool:
    """Whether a job at ``effective`` integrity may write to a capability requiring
    ``required``. Lower is more trusted, so the test is ``effective <= required``."""
    return int(effective) <= int(required)
