# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The interrupt vector table.

An interrupt source is a device pipe with a handler descriptor bound to it at a
priority. A write to that pipe makes the handler runnable, which (if it outranks
the running job) preempts within one token boundary. The vector table is therefore
just the table of (pipe → handler, priority) bindings -- one wake mechanism serving
dataflow, tool completion, and interrupts alike (core §4.4).

Storm control is the interesting part, and the policies are not interchangeable:

* ``coalesce`` is **level-triggered**: N pending firings collapse into one dispatch
  that reads the latest value. Correct for sensors, where what matters is the
  current reading and not how many times it changed on the way here.
* ``queue`` is **edge-triggered**: every firing eventually gets its own dispatch,
  serialised. Correct for commands and discrete events, where dropping one loses
  information.
* ``reentrant`` runs instances in parallel, and is only safe when their read/write
  sets are disjoint across instances (core §5.5).

``min_interval`` throttling **retains** the pending firing rather than discarding
it. A throttle that dropped events would silently convert a storm into data loss,
which is the opposite of failing loudly.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass

from zeos.core.ids import (
    DescriptorName,
    PipeName,
    Priority,
    VectorName,
    VectorPolicy,
)

__all__ = ["VectorSpec", "VectorAction", "VectorDecision", "VectorTable"]


@dataclass(frozen=True, slots=True)
class VectorSpec:
    """One row of the vector table (core §5.1)."""

    name: VectorName
    source: PipeName
    handler: DescriptorName
    priority: Priority
    policy: VectorPolicy = VectorPolicy.COALESCE
    min_interval_ns: int | None = None
    #: Event-to-first-corrective-action budget. Phase 1 records it; it is what
    #: drives the placement rule once there is a link to compare it against
    #:.
    deadline_ns: int | None = None


class VectorAction(enum.StrEnum):
    DISPATCH = "dispatch"
    COALESCE = "coalesce"  # folded into an already-pending dispatch
    THROTTLE = "throttle"  # deferred by min_interval; still pending
    QUEUE = "queue"  # serialised behind an active instance


@dataclass(frozen=True, slots=True)
class VectorDecision:
    spec: VectorSpec
    action: VectorAction
    collapsed: int = 0
    since_last_ns: int = 0


@dataclass
class _VectorState:
    pending: int = 0
    collapsed: int = 0
    active: int = 0
    last_fired_ns: int | None = None


class VectorTable:
    """Bindings plus the per-vector state that storm control needs."""

    def __init__(self, specs: Iterable[VectorSpec] = ()) -> None:
        self._specs: dict[VectorName, VectorSpec] = {}
        self._by_pipe: dict[PipeName, list[VectorName]] = {}
        self._state: dict[VectorName, _VectorState] = {}
        for spec in specs:
            self.bind(spec)

    def bind(self, spec: VectorSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate vector {spec.name!r}")
        self._specs[spec.name] = spec
        self._by_pipe.setdefault(spec.source, []).append(spec.name)
        self._state[spec.name] = _VectorState()

    def specs(self) -> tuple[VectorSpec, ...]:
        return tuple(self._specs[n] for n in sorted(self._specs))

    def get(self, name: VectorName) -> VectorSpec:
        return self._specs[name]

    def for_pipe(self, pipe: PipeName) -> tuple[VectorSpec, ...]:
        # Sorted by name so that two vectors on one pipe fire in a stable order.
        return tuple(self._specs[n] for n in sorted(self._by_pipe.get(pipe, ())))

    def on_write(self, pipe: PipeName, now_ns: int) -> tuple[VectorDecision, ...]:
        """A write landed on ``pipe``. Decide what each bound vector does."""
        decisions: list[VectorDecision] = []
        for spec in self.for_pipe(pipe):
            state = self._state[spec.name]
            since = 0 if state.last_fired_ns is None else now_ns - state.last_fired_ns

            if spec.min_interval_ns is not None and state.last_fired_ns is not None:
                if since < spec.min_interval_ns:
                    state.pending += 1
                    decisions.append(
                        VectorDecision(spec, VectorAction.THROTTLE, since_last_ns=since)
                    )
                    continue

            match spec.policy:
                case VectorPolicy.COALESCE:
                    if state.pending > 0 or state.active > 0:
                        state.collapsed += 1
                        decisions.append(
                            VectorDecision(spec, VectorAction.COALESCE, collapsed=state.collapsed)
                        )
                        continue
                    state.pending += 1
                    decisions.append(
                        VectorDecision(spec, VectorAction.DISPATCH, since_last_ns=since)
                    )
                case VectorPolicy.QUEUE:
                    state.pending += 1
                    if state.active > 0:
                        decisions.append(VectorDecision(spec, VectorAction.QUEUE))
                        continue
                    decisions.append(
                        VectorDecision(spec, VectorAction.DISPATCH, since_last_ns=since)
                    )
                case VectorPolicy.REENTRANT:
                    state.pending += 1
                    decisions.append(
                        VectorDecision(spec, VectorAction.DISPATCH, since_last_ns=since)
                    )
        return tuple(decisions)

    def mark_dispatched(self, name: VectorName, now_ns: int) -> int:
        """Handler instance started. Returns the number of firings it absorbed."""
        state = self._state[name]
        absorbed = state.collapsed
        state.collapsed = 0
        state.pending = max(0, state.pending - 1)
        state.active += 1
        state.last_fired_ns = now_ns
        return absorbed

    def mark_complete(self, name: VectorName) -> None:
        state = self._state[name]
        state.active = max(0, state.active - 1)

    def due(self, now_ns: int) -> tuple[VectorSpec, ...]:
        """Vectors with pending firings whose throttle interval has now elapsed.

        This is what keeps a throttle from becoming data loss: the deferred firing
        comes back rather than evaporating.
        """
        ready: list[VectorSpec] = []
        for name in sorted(self._specs):
            spec, state = self._specs[name], self._state[name]
            if state.pending <= 0:
                continue
            if spec.policy is not VectorPolicy.REENTRANT and state.active > 0:
                continue
            if spec.min_interval_ns is not None and state.last_fired_ns is not None:
                if now_ns - state.last_fired_ns < spec.min_interval_ns:
                    continue
            ready.append(spec)
        return tuple(ready)

    def pending(self, name: VectorName) -> int:
        return self._state[name].pending

    def active(self, name: VectorName) -> int:
        return self._state[name].active
