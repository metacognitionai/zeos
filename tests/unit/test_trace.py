# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The machine trace: a machine's own account of its windows, sampled by the driver.

The load-bearing test is the round trip: rows written as differences must replay to
exactly what the machine reports at the end. The rest pins the side-file contract --
keyed by journal sequence number, one row per window a tick changed, and never a
change to the journal itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.events import BlockBoundary, Decoded, Event, Forked, Injected, Spliced
from zeos.core.ids import JobId
from zeos.core.kernel import Kernel, KernelConfig
from zeos.descriptor.loader import CaseBundle, load_case
from zeos.driver import Driver, build_kernel, load_schedule
from zeos.journal.writer import Journal
from zeos.machine.base import RawWindow, RawWord, TracesRaw
from zeos.machine.scripted import ScriptedMachine
from zeos.trace import RawTrace, read_trace, replay_trace

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


@pytest.fixture(scope="module")
def bundle() -> CaseBundle:
    return load_case(SMOKE)


@pytest.fixture(scope="module")
def traced(bundle: CaseBundle, tmp_path_factory: pytest.TempPathFactory) -> tuple[Kernel, Path]:
    path = tmp_path_factory.mktemp("trace") / "smoke.trace.jsonl"
    kernel, transport = build_kernel(bundle, config=KernelConfig(case=bundle.name))
    trace = RawTrace(path)
    driver = Driver(kernel, transport=transport, journal=Journal(), trace=trace)
    driver.boot(bundle.boot)
    driver.run(load_schedule(SMOKE / "events.jsonl"))
    trace.close()
    return kernel, path


def test_the_scripted_machine_gives_an_account() -> None:
    """One word is one piece and everything is resident: the trivial account, and the
    one the tests below can run without weights."""
    assert isinstance(ScriptedMachine(), TracesRaw)


def test_the_rows_replay_to_what_the_machine_holds(traced: tuple[Kernel, Path]) -> None:
    kernel, path = traced
    rebuilt = replay_trace(read_trace(path))
    assert rebuilt, "a run with jobs must leave accounts to compare"
    assert isinstance(kernel.machine, TracesRaw)
    for job, account in rebuilt.items():
        window: RawWindow = kernel.machine.raw(JobId(job))
        assert [RawWord(tuple(w["pieces"]), tuple(w["framing"])) for w in account["words"]] == list(
            window.words
        )
        assert tuple(account["trailing"]) == window.trailing
        assert account["kv_resident"] == window.kv_resident
        assert account["model_tokens"] == window.model_tokens


def test_rows_are_keyed_to_the_journal_event_that_changed_the_window(
    traced: tuple[Kernel, Path],
) -> None:
    """A row's ``seq`` names a machine event for that job, so the debugger can line
    the account up with a frame by selection alone."""
    kernel, path = traced
    rows = read_trace(path)
    assert rows
    for row in rows:
        event: Event = kernel.events[int(row["seq"])]
        assert isinstance(event, Decoded | Injected | Spliced | BlockBoundary | Forked)
        touched = event.child if isinstance(event, Forked) else event.job
        assert int(touched) == row["job"]
    seqs = [row["seq"] for row in rows]
    assert seqs == sorted(seqs)


def test_rows_carry_only_what_changed(traced: tuple[Kernel, Path]) -> None:
    """A row carries the words from the first that differs, not the window again.
    Without this the file is quadratic in the run, which is the reason it is written
    as differences."""
    _kernel, path = traced
    rows = read_trace(path)
    assert any(row["from_word"] > 0 for row in rows)
    written = sum(len(row["words"]) for row in rows)
    held = sum(
        len(replay_trace(rows[: index + 1])[row["job"]]["words"]) for index, row in enumerate(rows)
    )
    assert written < held, "the rows re-send whole windows"


def test_a_tick_produces_one_row_per_window_it_changed(traced: tuple[Kernel, Path]) -> None:
    kernel, path = traced
    rows = read_trace(path)
    machine_events = [
        e for e in kernel.events if isinstance(e, Decoded | Injected | Spliced | Forked)
    ]
    # Never more rows than events that could have changed a window, and at least one
    # per job that has any.
    assert len(rows) <= len(machine_events)
    assert {row["job"] for row in rows} == {
        int(e.job) for e in machine_events if not isinstance(e, Forked)
    }


def test_the_journal_is_untouched_by_tracing(bundle: CaseBundle, tmp_path: Path) -> None:
    """The trace is a side file. The journal's bytes are the determinism gate's
    subject and must not depend on whether anyone was asking the machine questions."""

    def run(trace: RawTrace | None) -> bytes:
        kernel, transport = build_kernel(bundle, config=KernelConfig(case=bundle.name))
        journal = Journal()
        driver = Driver(kernel, transport=transport, journal=journal, trace=trace)
        driver.boot(bundle.boot)
        driver.run(load_schedule(SMOKE / "events.jsonl"))
        return journal.to_bytes()

    assert run(None) == run(RawTrace(tmp_path / "t.jsonl"))


def test_a_machine_without_an_account_is_refused_a_trace(bundle: CaseBundle) -> None:
    class Mute(ScriptedMachine):
        raw = None  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]

    kernel, _transport = build_kernel(bundle, config=KernelConfig(case=bundle.name))
    kernel.machine = Mute(bundle.scripts)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError, match="no account"):
        Driver(kernel, trace=RawTrace())


def test_replaying_nothing_is_nothing() -> None:
    assert replay_trace([]) == {}
