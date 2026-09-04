# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the tape-driven case: one interrupted turn, replayed the same way every time."""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.events import (
    Decoded,
    Event,
    JobPreempted,
    JobResumed,
    VectorFired,
    WorldWritten,
)
from zeos.core.ids import ResumeKind
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case, split_frontmatter
from zeos.driver import Driver, load_schedule
from zeos.journal.codec import to_line

from zeos_coop_count.boot import build_kernel
from zeos_coop_count.scripted import TapeSource
from zeos_coop_count.seat import CommandSeat

CASE = Path(__file__).resolve().parent.parent / "cases" / "coop-count-scripted"

#: What the console types, and what both counters must read afterwards.
NEW_COUNT = "51"


def play() -> list[Event]:
    """One whole run of the case: the tapes, the scheduled keypress, and nothing else."""
    bundle = load_case(CASE)
    events: list[Event] = []
    kernel, _transport, machine = build_kernel(
        bundle,
        machine=CommandSeat(source=TapeSource(bundle.scripts)),
        journal_sink=events,
        config=KernelConfig(case=bundle.name),
    )
    kernel.start()
    for name in bundle.boot:
        kernel.spawn(name)

    pending = sorted(load_schedule(CASE / "events.jsonl"), key=lambda e: e.at_ns)
    now = 0
    for _ in range(300):
        while pending and pending[0].at_ns <= now:
            event = pending.pop(0)
            kernel.deliver(event.pipe, event.text)
        kernel.advance_time(now)
        ran = kernel.tick()
        now += Driver.DEFAULT_NS_PER_TICK
        if not ran and not pending:
            break
    machine.close()  # pyright: ignore[reportAttributeAccessIssue]
    return events


@pytest.fixture(scope="module")
def run() -> list[Event]:
    return play()


def _of[E: Event](events: list[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def _said(events: list[Event], job: int) -> list[str]:
    """Every number one job spoke, in order, with the syscall words dropped."""
    words = "".join(t for e in _of(events, Decoded) if e.job == job for t in e.text)
    return [w.rstrip(";") for w in words.split() if w.rstrip(";").isdigit()]


def test_the_run_is_the_same_run_twice(run: list[Event]) -> None:
    """No clock, no sampler, no network: the tape is the only thing that decides."""
    again = play()
    assert [to_line(i, e) for i, e in enumerate(run)] == [
        to_line(i, e) for i, e in enumerate(again)
    ]


def test_every_turn_reaches_its_handover(run: list[Event]) -> None:
    """One write to world state per completed turn, plus the handler's two, in order."""
    assert [(w.obj, w.after) for w in _of(run, WorldWritten)] == [
        ("count.a", "10"),
        ("count.a", NEW_COUNT),
        ("count.b", NEW_COUNT),
        ("count.b", "60"),
    ]


def test_the_keypress_preempts_the_counter_mid_turn(run: list[Event]) -> None:
    fired = _of(run, VectorFired)
    assert [f.vector for f in fired] == ["keyboard-interrupt"]

    preemptions = _of(run, JobPreempted)
    assert preemptions, "a handler at priority 5 must displace a counter at 50"
    assert preemptions[0].by_priority == 5
    assert run.index(preemptions[0]) > run.index(fired[0])


def test_the_handler_sets_both_counters_to_the_typed_number(run: list[Event]) -> None:
    written = [w for w in _of(run, WorldWritten) if w.after == NEW_COUNT]
    assert {w.obj for w in written} == {"count.a", "count.b"}


def test_the_interrupted_job_is_told_what_changed_underneath_it(run: list[Event]) -> None:
    dirty = [r for r in _of(run, JobResumed) if r.resume_kind is ResumeKind.DIRTY]
    assert dirty, "the counter it displaced depends on state the handler changed"
    changed = {d.obj: (d.before, d.after) for d in dirty[0].dirty}
    assert changed == {"count.a": ("10", NEW_COUNT)}


def test_the_interrupted_job_carries_on_from_the_new_count(run: list[Event]) -> None:
    """counter-b counts to fifteen, is reset to fifty-one, and resumes at fifty-two."""
    spoken = _said(run, 2)

    said, recorded = spoken[:14], spoken[14]

    assert said == ["11", "12", "13", "14", "15"] + [str(n) for n in range(52, 61)], (
        "the resumed job took its next number from its own working, not from the reset"
    )
    assert recorded == "60", "the interrupted turn still ends on its own target"
    assert "16" not in spoken, "nothing carried on from where the interrupt landed"


def test_the_run_stops_partway_up_the_next_turn(run: list[Event]) -> None:
    """counter-a takes the machine back at sixty and the tape ends before it records."""
    assert _said(run, 1)[-5:] == ["61", "62", "63", "64", "65"]


def test_the_tape_case_carries_the_same_prompts_as_the_case_it_copies() -> None:
    """A golden trace is only golden while both cases give their jobs the same words.

    The frontmatter differs -- this case adds the tape -- but the body below it is the
    prompt, and a change to one case's prose that misses the other quietly turns this
    from a reproduction of the live run into a reproduction of an older one.
    """
    live = CASE.parent / "coop-count-pipe"

    for name in ("goals/counter-a.md", "goals/counter-b.md", "handlers/reset-count.md"):
        _, tape_body = split_frontmatter((CASE / name).read_text(), source=name)
        _, live_body = split_frontmatter((live / name).read_text(), source=name)
        assert tape_body == live_body, f"{name} has drifted from coop-count-pipe"
