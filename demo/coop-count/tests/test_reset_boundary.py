# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A reset lands the count on the next decade boundary, in either direction."""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.events import Event, JobResumed, WorldWritten
from zeos.core.ids import JobState, PipeName, ResumeKind
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case
from zeos.driver import Driver

from zeos_coop_count.boot import build_kernel
from zeos_coop_count import model as model_mod
from zeos_coop_count.machine import LlamaModel

CASE = Path(__file__).resolve().parent.parent / "cases" / "coop-count-pipe"
INTERRUPT, NUMBER = PipeName("keys.interrupt"), PipeName("keys.number")

#: Ticks between the keypress and the typed number, and the length of the whole run.
GAP, TICKS = 20, 900


def next_decade(value: int) -> int:
    """Returns the next multiple of ten above ``value``."""
    return (value // 10 + 1) * 10


def _run_with_reset(model: LlamaModel, number: str) -> list[Event]:
    bundle = load_case(CASE)
    events: list[Event] = []
    kernel, _transport, machine = build_kernel(
        bundle,
        model,
        journal_sink=events,
        config=KernelConfig(case=bundle.name),
        n_threads=model_mod.DEFAULT_THREADS,
    )
    kernel.start()
    for name in bundle.boot:
        kernel.spawn(name)
    now = 0
    pressed_at: int | None = None
    for tick in range(TICKS):
        if (
            pressed_at is None
            and any(isinstance(e, WorldWritten) for e in events)
            and any(job.state is JobState.RUNNING for job in kernel.sched.jobs())
        ):
            kernel.deliver(INTERRUPT, "attention")
            pressed_at = tick
        if pressed_at is not None and tick == pressed_at + GAP:
            kernel.deliver(NUMBER, number)
        kernel.advance_time(now)
        kernel.tick()
        now += Driver.DEFAULT_NS_PER_TICK
    machine.close()
    assert pressed_at is not None, "the counters never exchanged; the fixture pressed nothing"
    return events


@pytest.fixture(scope="module")
def increased(request: pytest.FixtureRequest) -> list[Event]:
    """A run whose reset is far above where the counters had got to."""
    return _run_with_reset(request.getfixturevalue("llama_model"), "503")


@pytest.fixture(scope="module")
def decreased(request: pytest.FixtureRequest) -> list[Event]:
    """A run whose reset is far below where the counters had got to."""
    return _run_with_reset(request.getfixturevalue("llama_model"), "5")


def _progress(events: list[Event]) -> list[tuple[int, int, str, str]]:
    return [
        (i, e.job, e.obj, e.after)
        for i, e in enumerate(events)
        if isinstance(e, WorldWritten) and e.after.isdigit() and e.job is not None
    ]


def _after_the_reset(events: list[Event], number: str) -> list[int]:
    """Values recorded after the reset, by the jobs that took a dirty resume."""
    writes = _progress(events)
    reset_at = max(i for i, _job, _obj, after in writes if after == number)
    told = {
        r.job for r in events if isinstance(r, JobResumed) and r.resume_kind is ResumeKind.DIRTY
    }
    return [int(after) for i, job, _obj, after in writes if i > reset_at and job in told]


@pytest.mark.parametrize("fixture_name, number", [("increased", "503"), ("decreased", "5")])
def test_the_reset_reaches_both_jobs(
    fixture_name: str, number: str, request: pytest.FixtureRequest
) -> None:
    events: list[Event] = request.getfixturevalue(fixture_name)
    written = [e for e in events if isinstance(e, WorldWritten) and e.after == number]
    assert {e.obj for e in written} == {"count.a", "count.b"}

    dirty = [r for r in events if isinstance(r, JobResumed) and r.resume_kind is ResumeKind.DIRTY]
    assert dirty, "a job suspended across the reset must be told what changed"


def test_the_next_segment_ends_on_the_decade_boundary(increased: list[Event]) -> None:
    """After a reset to 503 the next value recorded is 510."""
    following = _after_the_reset(increased, "503")
    assert following, "a counter must record something after the reset"
    assert following[0] == next_decade(503), f"reset to 503: expected 510, got {following[0]}"


@pytest.mark.xfail(
    reason="a downward reset moves the baseline but not the target",
    strict=False,
)
def test_a_downward_reset_also_lands_on_the_boundary(decreased: list[Event]) -> None:
    """After a reset to 5 the next value recorded is 10.

    **Half the instruction survives the resume.** The job restarts from the status value
    plus one, so it has read the new number and abandoned where it had got to; it then
    counts past the target that number implies rather than working one out afresh. That
    is why upward resets pass: when the new value is above where the job had reached,
    restarting from it and overrunning the old target lands on the right answer anyway,
    and the dropped half is invisible.

    The count of stale ``say`` commands is **not** what decides it. Holding the reset and
    the preemption point fixed and varying only how far the job had counted -- one stale
    command against seven -- produces byte-identical behaviour. An earlier reading of this
    case had the job's own recent output outweighing the kernel's diff; that predicts a
    difference between one and seven, and there is none.

    Not strict, because the model is the variable and it sometimes gets this right.
    ``test_a_reset_downwards_does_not_run_away`` below is the bound that must hold either
    way: landing on the wrong decade is recoverable, and chasing a target below yourself
    forever is not.
    """
    following = _after_the_reset(decreased, "5")
    assert following, "a counter must record something after the reset"
    assert following[0] == next_decade(5), f"reset to 5: expected 10, got {following[0]}"


def test_a_reset_downwards_does_not_run_away(decreased: list[Event]) -> None:
    """After a downward reset the recorded values stay within a small bound."""
    following = _after_the_reset(decreased, "5")
    assert following, "a counter must record something after the reset"
    assert following[0] <= 100, f"a downward reset must not run away: {following}"
    assert max(following) <= 1000, f"a downward reset must not run away: {following}"

    # Values after a downward reset are not checked for going only upwards.


def test_a_reset_upwards_does_not_replay_the_gap(increased: list[Event]) -> None:
    """After an upward reset a notified job records 510 and then moves on past 503."""
    following = _after_the_reset(increased, "503")
    assert following[0] == 510
    assert max(following) > 503, "the notified job must move on past the reset"


# Not asserted anywhere here: that progress keeps going up for the rest of the run.
