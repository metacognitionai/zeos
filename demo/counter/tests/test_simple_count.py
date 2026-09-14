# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The shipped ``simple_count`` case runs the two laps the README prints.

``simple_count`` boots two descriptors whose script counts to five, spawns a fresh
instance of itself and exits. The spawn is a capability the descriptor has to declare,
so this asserts the lap the README shows: two jobs completed, each having spawned its
successor, and no fault anywhere.
"""

from __future__ import annotations

from pathlib import Path

from zeos.core.events import Event, FaultRaised, JobCompleted, JobSpawned
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case
from zeos.driver import Driver, build_kernel
from zeos.journal.writer import Journal

CASE = Path(__file__).parent.parent / "simple_count"

#: The two laps the README prints, and the handover between them.
MAX_TICKS = 16


def run_case(case: Path) -> list[Event]:
    bundle = load_case(case)
    events: list[Event] = []
    kernel, transport = build_kernel(
        bundle,
        journal_sink=events,
        config=KernelConfig(case=bundle.name, max_ticks=MAX_TICKS),
    )
    driver = Driver(kernel, transport=transport, journal=Journal(None))
    driver.boot(bundle.boot)
    driver.run()
    return events


def of[E: Event](events: list[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def test_shipped_case_completes_both_laps() -> None:
    events = run_case(CASE)
    assert of(events, FaultRaised) == []
    assert [e.job for e in of(events, JobCompleted)] == [1, 2]
    assert [(e.job, e.parent) for e in of(events, JobSpawned)] == [
        (1, None),
        (2, None),
        (3, 1),
        (4, 2),
    ]
