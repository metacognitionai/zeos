# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Stage B acceptance, driven the way a user drives it: a case directory on disk,
an event schedule, and the CLI.

The point of running this through ``Driver`` rather than poking the kernel directly
is that the driver is where the interesting mistake lives. A driver that runs to
quiescence before each scheduled event can only ever deliver interrupts to an idle
kernel -- which makes preemption structurally unobservable while every unit test
still passes. Ticks must consume virtual time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.events import (
    Event,
    JobCompleted,
    JobPreempted,
    JobResumed,
    VectorFired,
)
from zeos.core.ids import PipeName, Priority, ResumeKind
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case
from zeos.driver import Driver, ScheduledEvent, build_kernel, load_schedule
from zeos.journal.writer import Journal, read_journal

CASE = Path(__file__).parent.parent / "fixtures" / "smoke"
ALARM_AT_NS = 3_000_000  # 3ms -- mid-flight for the supervision job


def run_case(
    *, schedule: tuple[ScheduledEvent, ...], journal_path: Path | None = None
) -> tuple[list[Event], Journal]:
    bundle = load_case(CASE)
    events: list[Event] = []
    kernel, transport = build_kernel(
        bundle, journal_sink=events, config=KernelConfig(case=bundle.name), block_size=8
    )
    journal = Journal(journal_path)
    driver = Driver(kernel, transport=transport, journal=journal)
    driver.boot(bundle.boot)
    driver.run(schedule)
    journal.close()
    return events, journal


def of[E: Event](events: list[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def test_case_runs_to_completion_without_a_schedule() -> None:
    events, _ = run_case(schedule=())
    completed = of(events, JobCompleted)
    assert len(completed) == 1, "only the goal job should run with no interrupts"
    assert not of(events, VectorFired)


def test_scheduled_event_preempts_a_running_job() -> None:
    events, _ = run_case(
        schedule=(
            ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level 42"),
        )
    )
    fired = of(events, VectorFired)
    preempted = of(events, JobPreempted)

    assert len(fired) == 1 and fired[0].handler == "threshold-alarm"
    assert len(preempted) == 1, (
        "the alarm arrived while the goal job was running and must preempt it"
    )
    assert preempted[0].by_priority == Priority(5)


def test_preempted_job_resumes_with_a_dirty_diff() -> None:
    events, _ = run_case(
        schedule=(
            ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level 42"),
        )
    )
    resumed = of(events, JobResumed)
    assert len(resumed) == 1
    assert resumed[0].resume_kind is ResumeKind.DIRTY
    assert [(d.obj, d.before, d.after) for d in resumed[0].dirty] == [
        ("plant.unit_a", "idle", "max")
    ]


def test_both_jobs_finish() -> None:
    events, _ = run_case(
        schedule=(
            ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level 42"),
        )
    )
    assert len(of(events, JobCompleted)) == 2


def test_schedule_loads_from_jsonl() -> None:
    schedule = load_schedule(CASE / "events.jsonl")
    assert len(schedule) == 1
    assert schedule[0].pipe == "sensors.threshold"


@pytest.mark.determinism
def test_two_runs_of_the_same_case_are_byte_identical(tmp_path: Path) -> None:
    schedule = (
        ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level 42"),
    )
    _, first = run_case(schedule=schedule, journal_path=tmp_path / "a.jsonl")
    _, second = run_case(schedule=schedule, journal_path=tmp_path / "b.jsonl")
    assert first.to_bytes() == second.to_bytes()


@pytest.mark.determinism
def test_journal_file_replays_byte_identically(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    schedule = (
        ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level 42"),
    )
    run_case(schedule=schedule, journal_path=path)

    rebuilt = Journal()
    rebuilt.extend(r.event for r in read_journal(path))
    assert rebuilt.to_bytes() == path.read_bytes()


def test_cli_run_replay_and_inspect(tmp_path: Path) -> None:
    from zeos.cli import main

    journal = tmp_path / "cli.jsonl"
    assert main(["lint", str(CASE)]) == 0
    assert (
        main(
            [
                "run",
                str(CASE),
                "--events",
                str(CASE / "events.jsonl"),
                "--journal",
                str(journal),
            ]
        )
        == 0
    )
    assert main(["replay", str(journal), "--assert-identical"]) == 0
    assert main(["inspect", str(journal)]) == 0


def test_cli_refuses_to_run_a_tree_that_does_not_lint(tmp_path: Path) -> None:
    """Load-time rejection is the compiler error that arrives before the robot
    moves -- so it must actually stop the run."""
    from zeos.cli import main

    case = tmp_path / "broken"
    (case / "goals").mkdir(parents=True)
    (case / "goals" / "bad.md").write_text(
        "---\nname: bad\npriority: 10\npreemptible: false\n"
        "budget:\n  tokens: 100000\nscript:\n  - exit: true\n---\nbody\n",
        encoding="utf-8",
    )
    assert main(["run", str(case)]) == 1
    assert main(["run", str(case), "--force"]) == 0


def test_job_states_are_terminal_at_the_end() -> None:
    bundle = load_case(CASE)
    events: list[Event] = []
    kernel, transport = build_kernel(bundle, journal_sink=events, block_size=8)
    driver = Driver(kernel, transport=transport)
    driver.boot(bundle.boot)
    driver.run(
        (ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level"),)
    )

    assert all(j.state.is_terminal for j in kernel.sched.jobs())
    assert kernel.sched.is_quiescent()
    assert kernel.sched.stack_depth == 0
