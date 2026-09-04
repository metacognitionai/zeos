# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Tests for a keypress that spawns a handler, preempts a counter, and resets the count."""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.events import (
    Event,
    JobDispatched,
    JobPreempted,
    JobResumed,
    JobSpawned,
    VectorFired,
    WorldWritten,
)
from zeos.core.ids import JobState, PipeName, ResumeKind
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case
from zeos.driver import Driver

from zeos_coop_count.boot import build_kernel
from zeos_coop_count import model as model_mod
from zeos_coop_count.machine import LlamaModel

CASE = Path(__file__).resolve().parent.parent / "cases" / "coop-count-pipe"
INTERRUPT, NUMBER = PipeName("keys.interrupt"), PipeName("keys.number")

#: Ticks between the keypress and the typed number, so the handler has to wait for it.
GAP = 20
NEW_COUNT = "500"


@pytest.fixture(scope="module")
def run(request: pytest.FixtureRequest) -> list[Event]:
    model: LlamaModel = request.getfixturevalue("llama_model")
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
    for tick in range(600):
        if (
            pressed_at is None
            and any(isinstance(e, WorldWritten) for e in events)
            and any(job.state is JobState.RUNNING for job in kernel.sched.jobs())
            and any(job.state is JobState.READY for job in kernel.sched.jobs())
        ):
            kernel.deliver(INTERRUPT, "attention")
            pressed_at = tick
        if pressed_at is not None and tick == pressed_at + GAP:
            kernel.deliver(NUMBER, NEW_COUNT)
        kernel.advance_time(now)
        kernel.tick()
        now += Driver.DEFAULT_NS_PER_TICK
    machine.close()
    assert pressed_at is not None, (
        "no moment had a job running with its peer ready; the fixture pressed nothing"
    )
    return events


def _of[E: Event](events: list[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def test_a_keypress_fires_the_vector_and_spawns_the_handler(run: list[Event]) -> None:
    fired = _of(run, VectorFired)
    assert [f.vector for f in fired] == ["keyboard-interrupt"]
    assert fired[0].pipe == INTERRUPT
    assert fired[0].priority == 5

    spawned = [s for s in _of(run, JobSpawned) if s.descriptor == "reset-count"]
    assert len(spawned) == 1, "queue policy serialises: one keypress, one handler"
    assert run.index(spawned[0]) > run.index(fired[0])


def test_the_handler_preempts_whatever_was_running(run: list[Event]) -> None:
    """Each preemption is by the priority 5 handler and pushes onto the suspension stack."""
    preemptions = _of(run, JobPreempted)
    assert preemptions, "a handler at priority 5 must displace a counter at 50"
    for event in preemptions:
        assert event.by_priority == 5
        assert event.stack_depth >= 1


def test_the_handler_waits_for_the_human_without_polling(run: list[Event]) -> None:
    """The handler blocks on ``keys.number`` and is woken when the number arrives."""
    from zeos.core.events import JobBlocked, JobWoken

    blocked = [b for b in _of(run, JobBlocked) if b.pipe == NUMBER]
    woken = [w for w in _of(run, JobWoken) if w.pipe == NUMBER]
    assert blocked and woken
    assert blocked[0].reason == "read-empty"
    assert run.index(woken[0]) > run.index(blocked[0])


def test_the_handler_writes_the_typed_number_to_world_state(run: list[Event]) -> None:
    written = [w for w in _of(run, WorldWritten) if w.after == NEW_COUNT]
    assert {w.obj for w in written} == {"count.a", "count.b"}


def test_a_resumed_job_is_told_what_changed_underneath_it(run: list[Event]) -> None:
    """A dirty resume names the objects that changed and carries the new value."""
    dirty = [r for r in _of(run, JobResumed) if r.resume_kind is ResumeKind.DIRTY]
    assert dirty, "the counters depend on state the handler changed"

    for resume in dirty:
        changed = {d.obj: (d.before, d.after) for d in resume.dirty}
        assert changed, "a dirty resume must name what changed"
        assert NEW_COUNT in {after for _, after in changed.values()}


def test_the_interrupted_job_resumes_before_its_peer_runs_again(run: list[Event]) -> None:
    """The first job dispatched after the preemption, handler aside, is the job displaced."""
    preemptions = _of(run, JobPreempted)
    assert preemptions, "nothing was interrupted, so there is no ordering to check"
    # The first preemption is the keypress landing; later ones are a consequence of it.
    interrupted = preemptions[0].job

    handlers = {e.job for e in _of(run, JobSpawned) if str(e.descriptor) == "reset-count"}
    at = next(i for i, e in enumerate(run) if isinstance(e, JobPreempted) and e.job == interrupted)
    after = [e.job for e in run[at:] if isinstance(e, JobDispatched) and e.job not in handlers]
    assert after, "nothing ran after the interrupt"
    assert after[0] == interrupted, (
        "a peer took the machine while the interrupted job sat suspended, so the "
        "resume notice arrived a turn late -- which is the reported symptom"
    )


def test_the_reset_is_the_last_word_on_where_the_count_is(run: list[Event]) -> None:
    """Progress rises up to the reset and never falls back to the start after it."""
    values = [int(w.after) for w in _of(run, WorldWritten) if w.after.isdigit()]
    reset_at = max(i for i, v in enumerate(values) if v == int(NEW_COUNT))
    assert values[: reset_at + 1] == sorted(values[: reset_at + 1]), (
        f"progress before the reset went backwards: {values}"
    )
    assert all(v >= 10 for v in values[reset_at + 1 :]), (
        f"nothing after the reset may fall back to the start: {values}"
    )
