# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What every run leaves behind, whoever played it.

These are the invariants a reader -- a comparison, an analysis script, a viewer
-- is allowed to rely on. If one of them breaks, a run stops being replayable.
"""

import json
from pathlib import Path

import pytest

from zeos_space_invaders.clocks import RealtimeRunner, run_episode
from zeos_space_invaders.players import RandomPlayer
from zeos_space_invaders.runlog import (
    SCHEMA_VERSION,
    Decision,
    Frame,
    RunReader,
    RunWriter,
)

#: The world as the clocks hand it over -- `env.info()` and `snapshot()` agree
#: on this shape, which is the only thing `Frame` knows how to read.
INFO = {
    "ticks": 0,
    "score": 0,
    "lives": 3,
    "player": 4,
    "monsters": {},
    "missile": None,
    "dangers": [],
    "can_shoot": True,
    "won": False,
}


def test_a_run_says_what_it_was_before_it_has_finished(tmp_path):
    """meta.json is written up front: a run that dies is still identifiable."""
    with RunWriter(tmp_path / "r", {"player": "random", "seed": 7}) as run:
        meta = json.loads((run.path / "meta.json").read_text())
        assert not (run.path / "summary.json").exists()
    assert meta["schema"] == SCHEMA_VERSION
    assert meta["player"] == "random" and meta["seed"] == 7
    assert meta["started_at"] and "commit" in meta


def test_a_run_records_the_checkout_it_came_from(tmp_path):
    """And says nothing rather than guessing when there is no checkout."""
    from zeos_space_invaders.runlog import git_commit

    here = Path(__file__).resolve().parents[1]
    assert git_commit(here), "the repository's own commit was not readable"
    assert git_commit(tmp_path) is None


def test_the_summary_appearing_is_what_marks_a_run_complete(tmp_path):
    with RunWriter(tmp_path / "r", {"player": "random"}) as run:
        run.finish({"score": 30})
    episode = RunReader(tmp_path / "r")
    assert episode.summary["score"] == 30


def test_the_settings_are_not_copied_into_the_verdict(tmp_path):
    """One copy of each fact: the request in meta, the outcome in summary."""
    with RunWriter(tmp_path / "r", {"player": "random", "seed": 4}) as run:
        run.finish({"score": 30})
    episode = RunReader(tmp_path / "r")
    assert "seed" not in episode.summary and "player" not in episode.summary
    # `row()` is the one place they are put back together.
    assert episode.row() == {**episode.meta, **episode.summary}
    assert episode.row()["seed"] == 4 and episode.row()["score"] == 30


def test_a_row_leaves_the_prompt_out_but_the_meta_keeps_it(tmp_path):
    """A table of runs should not carry kilobytes of prompt per line."""
    with RunWriter(tmp_path / "r", {"prompt": "RULES" * 500}) as run:
        run.finish({"score": 0})
    episode = RunReader(tmp_path / "r")
    assert "prompt" not in episode.row()
    assert episode.meta["prompt"].startswith("RULES")


def test_the_summary_is_read_once_per_reader(tmp_path, monkeypatch):
    """The viewer's index asks for it per row on a timer, so an uncached property
    was three opens a run."""
    from zeos_space_invaders.runlog import reader as reader_module

    with RunWriter(tmp_path / "r", {"player": "random"}) as run:
        run.finish({"outcome": "lost", "score": 0})

    episode = RunReader(tmp_path / "r")
    opened = []
    real = reader_module._json
    monkeypatch.setattr(
        reader_module,
        "_json",
        lambda path: (opened.append(path.name), real(path))[1],
    )
    episode.row()
    assert episode.summary is not None and episode.summary is not None
    assert opened.count("summary.json") == 1


def test_an_unfinished_run_reads_back_as_unfinished(tmp_path):
    with RunWriter(tmp_path / "r"):
        pass
    assert RunReader(tmp_path / "r").summary is None


def test_a_reader_refuses_a_schema_it_was_not_written_against(tmp_path):
    RunWriter(tmp_path / "r").close()
    path = tmp_path / "r" / "meta.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema": 99}))
    with pytest.raises(ValueError, match="schema 99"):
        RunReader(tmp_path / "r")


def test_the_clock_has_to_stamp_a_decision_with_a_tick(tmp_path):
    """A decision with no tick cannot be joined to a frame, so it is refused."""
    run = RunWriter(tmp_path / "r")
    with pytest.raises(ValueError, match="no tick"):
        run.decision(Decision(by="pilot", action="left"))
    run.close()


def test_a_frame_keeps_every_field_even_when_empty(tmp_path):
    """A missing key would read as "not recorded" rather than "nothing there"."""
    info = INFO
    record = Frame.of(info).record()
    assert record["missile"] is None and record["dangers"] == []
    assert set(record) == {"kind", *info} - {"ticks"} | {"tick", "over"}


def test_a_decision_leaves_out_what_its_player_cannot_fill(tmp_path):
    """One shape for every player, without pretending random reasons."""
    line = Decision(by="random", action="left", tick=3).record()
    assert line["kind"] == "decision" and line["by"] == "random"
    assert "reasoning" not in line and "kernel_ticks" not in line


def test_every_decision_names_a_frame_that_exists(tmp_path):
    """The join the whole layout rests on: a decision points at what it saw."""
    with RunWriter(tmp_path / "r") as run:
        run_episode(RandomPlayer(seed=1), seed=1, max_steps=30, run=run, verbose=False)
    episode = RunReader(tmp_path / "r")
    decisions = list(episode.decisions())
    assert decisions, "nothing was logged"
    for decision in decisions:
        frame = episode.frame_at(decision["tick"])
        assert frame, f"decision at tick {decision['tick']} has no frame"


def test_frames_and_decisions_share_one_ordered_stream(tmp_path):
    """Interleaving is the data: it says what the world did between two moves."""
    with RunWriter(tmp_path / "r") as run:
        RealtimeRunner(
            RandomPlayer(seed=2, latency=0.01), seed=2, tick_seconds=0.005
        ).run(run=run)
    kinds = [line["kind"] for line in RunReader(tmp_path / "r").events()]
    assert {"frame", "decision"} == set(kinds)
    seqs = [line["seq"] for line in RunReader(tmp_path / "r").events("decision")]
    assert seqs == sorted(seqs), "decisions are not in the order they were made"


def test_nothing_is_written_after_the_writer_is_closed(tmp_path):
    """The realtime clock is a daemon thread that may still be ticking when the
    run is torn down -- on a Ctrl-C nothing gets to stop it first."""
    run = RunWriter(tmp_path / "r")
    run.frame(INFO)
    run.close()
    run.frame(INFO)  # the late tick, dropped rather than raised
    assert len(list(RunReader(tmp_path / "r").frames())) == 1


def test_a_run_with_no_events_yet_reads_as_empty(tmp_path):
    """A directory being read while it fills must not look corrupt."""
    (tmp_path / "r").mkdir()
    episode = RunReader(tmp_path / "r")
    assert list(episode.events()) == [] and list(episode.kernel()) == []
    assert episode.frame_at(0) is None and episode.meta == {}


def test_the_commit_is_none_outside_a_git_tree(monkeypatch):
    """Recorded when it can be; never a reason to refuse to start a run."""
    import subprocess

    from zeos_space_invaders.runlog import writer

    writer.git_commit.cache_clear()
    monkeypatch.setattr(subprocess, "run", _raise_oserror)
    assert writer.git_commit() is None
    writer.git_commit.cache_clear()


def _raise_oserror(*args, **kw):
    raise OSError("no git here")


def test_the_kernel_lane_is_only_opened_by_a_player_that_has_one(tmp_path):
    with RunWriter(tmp_path / "r") as run:
        run.kernel(0, [])
        assert not (run.path / "kernel.jsonl").exists()
        run.kernel(1, [{"event": "pipe.written"}])
    assert [e["tick"] for e in RunReader(tmp_path / "r").kernel()] == [1]


def test_a_single_run_has_no_episodes_rather_than_no_answer(tmp_path):
    """One loader is handed either kind of directory and asks which it has."""
    with RunWriter(tmp_path / "r") as run:
        run.finish({"score": 0})
    assert RunReader(tmp_path / "r").episodes() == []


def test_a_comparison_is_a_run_of_runs(tmp_path):
    """Every episode of a comparison is a plain run directory."""
    root = tmp_path / "cmp"
    for seed in (0, 1):
        with RunWriter(root / "episodes" / f"random-seed{seed}", {"seed": seed}) as ep:
            ep.finish(
                run_episode(
                    RandomPlayer(seed=seed),
                    seed=seed,
                    max_steps=10,
                    run=ep,
                    verbose=False,
                )
            )
    RunWriter(root).close()
    episodes = RunReader(root).episodes()
    assert [e.meta["seed"] for e in episodes] == [0, 1]
    assert all(e.summary["decisions"] == 10 for e in episodes)
