# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Journal tests for a run of two counters that pass a baton over pipes."""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.events import (
    Event,
    JobBlocked,
    JobCompleted,
    JobWoken,
    MapRefreshed,
    PipeReadEvent,
    PipeWritten,
    WorldWritten,
)
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case
from zeos.driver import Driver

from zeos_coop_count.boot import build_kernel
from zeos_coop_count import model as model_mod
from zeos_coop_count.machine import LlamaModel

CASE = Path(__file__).resolve().parent.parent / "cases" / "coop-count-pipe"


#: Enough ticks for the handoff to happen at least once each way, and few enough to keep
#: the run short.
TICKS = 320


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
    for _ in range(TICKS):
        kernel.advance_time(now)
        if not kernel.tick():
            break
        now += Driver.DEFAULT_NS_PER_TICK
    machine.close()
    return events


#: The pipes the two agents block and wake on, kept apart from the actuator pipes below.
SIGNAL_PIPES = {"count.a2b", "count.b2a"}
ACTUATORS = {"count.progress_a", "count.progress_b"}


def _of[E: Event](events: list[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def _signals(events: list[Event]) -> list[PipeWritten]:
    return [w for w in _of(events, PipeWritten) if w.pipe in SIGNAL_PIPES]


def test_a_reader_blocks_on_an_empty_pipe_rather_than_polling(run: list[Event]) -> None:
    """A read on an empty pipe blocks the job with reason ``read-empty``."""
    events = run
    blocked = _of(events, JobBlocked)
    assert blocked, "one of the two agents must have waited on an empty pipe"
    assert all(b.reason == "read-empty" for b in blocked)


def test_an_actuator_never_applies_backpressure_to_its_own_writer(run: list[Event]) -> None:
    """No job blocks with reason ``write-full`` while writing to an actuator pipe."""
    stalled = [b for b in _of(run, JobBlocked) if b.reason == "write-full"]
    assert not stalled, (
        "a counter blocked writing to an actuator nobody can drain; the run would have "
        "gone silently quiet from here"
    )


def test_a_peers_write_is_what_wakes_the_waiter(run: list[Event]) -> None:
    """Every wake follows straight after a write to the same pipe the job blocked on."""
    events = run
    woken = _of(events, JobWoken)
    assert woken, "the blocked agent must have been woken"

    for wake in woken:
        index = events.index(wake)
        cause = events[index - 1]
        assert isinstance(cause, PipeWritten), "a wake is caused by a write, not by a timer"
        assert cause.pipe == wake.pipe
        blocked_on = [b for b in _of(events, JobBlocked) if b.job == wake.job]
        assert blocked_on and blocked_on[-1].pipe == wake.pipe


def test_the_baton_crosses_in_both_directions(run: list[Event]) -> None:
    """Both signal pipes are written, one by each agent."""
    events = run
    written = {(w.pipe, w.job) for w in _signals(events)}
    assert {pipe for pipe, _ in written} == SIGNAL_PIPES
    assert len({job for _, job in written}) == 2, "each agent wrote exactly its own way"


def test_the_receiver_actually_consumed_what_was_sent(run: list[Event]) -> None:
    """Every signal sent is read, bar at most one still in the buffer when the run ends."""
    events = run
    sent, received = len(_signals(events)), len(_of(events, PipeReadEvent))
    assert sent - 1 <= received <= sent, (
        f"{sent} wake signals sent, {received} consumed; "
        "actuator writes are effects, not messages, and are not counted here"
    )


def test_neither_agent_ever_finishes(run: list[Event]) -> None:
    """No job completes during the run."""
    assert _of(run, JobCompleted) == []


def test_the_handoff_keeps_going(run: list[Event]) -> None:
    """The baton crosses more than once, and in both directions."""
    writes = _signals(run)
    assert len(writes) >= 3, "the loop must turn over more than once"
    assert {w.pipe for w in writes} == SIGNAL_PIPES


# -- durable state ------------------------------------------------------------


def test_recording_progress_reaches_the_world_and_comes_back_as_a_view(
    run: list[Event],
) -> None:
    """Each write to world state is followed by a map refresh for the peer job."""
    written = _of(run, WorldWritten)
    assert {w.obj for w in written} == {"count.a", "count.b"}

    for world_write in written:
        after = run[run.index(world_write) + 1 :]
        refreshed = [e for e in after if isinstance(e, MapRefreshed) and e.obj == world_write.obj]
        assert refreshed, f"{world_write.obj} changed without refreshing anyone's view"
        # A counter is mapped only by its peer, never by the job that writes it.
        assert refreshed[0].job != world_write.job


def test_the_peer_carries_on_from_the_recorded_value(run: list[Event]) -> None:
    """Recorded values only ever go up, and end higher than they started."""
    by_object: dict[str, list[int]] = {}
    for write in _of(run, WorldWritten):
        if write.after.isdigit():
            by_object.setdefault(write.obj, []).append(int(write.after))

    assert by_object, "the agents must have recorded something"
    for obj, values in by_object.items():
        assert values == sorted(values), f"{obj} went backwards: {values}"
        assert values[-1] > values[0], f"{obj} never advanced past {values[0]}"


def test_the_pipes_carry_no_numbers(run: list[Event]) -> None:
    """A signal pipe carries one token and no number, and the other pipes are actuators."""
    for write in _signals(run):
        assert write.tokens == 1, "a wake signal is one token and carries no state"
    assert {w.pipe for w in _of(run, PipeWritten)} - SIGNAL_PIPES == ACTUATORS
