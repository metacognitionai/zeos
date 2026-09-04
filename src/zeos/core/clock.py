# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Time, as data.

The kernel core never reads a wall clock. Both notions of time enter as inputs so
that a run is a pure function of (descriptor tree, event schedule, seed) -- which is
what makes byte-identical replay possible.

Two clocks, because the design needs both and they are not interchangeable:

``token_clock``
    Logical and kernel-wide: the count of tokens the machine has produced or had
    injected, across all jobs. This is the ordering used for ``injected_at``,
    ``last_touch``, segment ranges, and attention EMA horizons -- everything the
    specs measure in tokens rather than seconds. It advances only when the machine
    does work, so it is unaffected by how long the driver took to call us.

``virtual_ns``
    Virtual wall-clock nanoseconds, supplied by the driver. Used only where the
    world's time genuinely matters: deadlines, ``min_interval`` vector throttling,
    and the "Suspended 94s" line in a RESUME notice. In tests the driver
    supplies a scripted schedule; in deployment it supplies the real clock. The
    kernel cannot tell the difference, and that is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

__all__ = ["Clock", "format_duration", "NS_PER_SECOND", "NS_PER_MS"]

NS_PER_MS: Final = 1_000_000
NS_PER_SECOND: Final = 1_000_000_000


@dataclass(frozen=True, slots=True, order=True)
class Clock:
    """An instant in the kernel's two time bases. Immutable; advance returns a new one."""

    token_clock: int = 0
    virtual_ns: int = 0

    def tick_tokens(self, n: int = 1) -> Clock:
        """Advance the logical token clock. Never moves virtual time."""
        if n < 0:
            raise ValueError("token_clock is monotonic; n must be >= 0")
        return replace(self, token_clock=self.token_clock + n)

    def at_virtual(self, virtual_ns: int) -> Clock:
        """Set virtual time, as supplied by the driver. Monotonic by contract."""
        if virtual_ns < self.virtual_ns:
            raise ValueError(f"virtual clock moved backwards: {self.virtual_ns} -> {virtual_ns}")
        return replace(self, virtual_ns=virtual_ns)

    def elapsed_ns_since(self, earlier: Clock) -> int:
        return self.virtual_ns - earlier.virtual_ns

    def tokens_since(self, earlier: Clock) -> int:
        return self.token_clock - earlier.token_clock


def format_duration(ns: int) -> str:
    """Human-readable duration for kernel notices (e.g. the RESUME preamble).

    Kept in the core because the exact string is part of what the model reads, and
    therefore part of the behaviour under test -- not a presentation detail.
    """
    if ns < NS_PER_MS:
        return f"{ns}ns"
    if ns < NS_PER_SECOND:
        return f"{ns / NS_PER_MS:.0f}ms"
    seconds = ns / NS_PER_SECOND
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m"
