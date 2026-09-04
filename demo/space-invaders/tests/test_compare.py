# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""`compare`: a run of runs, one table, and the same seeds for everyone.

The layout is the point: each episode is a plain run directory, so anything that
reads one run reads any episode of a comparison.
"""

import json

import pytest

from zeos_space_invaders import compare
from zeos_space_invaders.runlog import Decision, RunReader, RunWriter


def test_every_episode_is_a_plain_run_directory(tmp_path, capsys):
    root = tmp_path / "cmp"
    compare.main(["--seeds", "2", "--players", "random", "--out", str(root)])

    episodes = RunReader(root).episodes()
    assert [e.path.name for e in episodes] == ["random-seed0", "random-seed1"]
    for episode in episodes:
        assert episode.summary["decisions"] == len(list(episode.decisions()))
        assert episode.meta["clock"] == "step"
        assert list(episode.frames()), "no timeline"
    assert "aimed" in capsys.readouterr().out


def test_the_comparison_says_what_it_compared(tmp_path):
    root = tmp_path / "cmp"
    compare.main(
        ["--seeds", "1", "--players", "random", "--max-steps", "20", "--out", str(root)]
    )
    meta = RunReader(root).meta
    assert meta["kind"] == "compare" and meta["players"] == ["random"]
    assert meta["seeds"] == 1 and meta["max_steps"] == 20


def test_a_table_row_says_which_player_and_seed_it_was(tmp_path):
    """Otherwise the position in the file is the only clue."""
    root = tmp_path / "cmp"
    compare.main(["--seeds", "1", "--players", "random", "--out", str(root)])
    row = json.loads((root / "table.json").read_text())["random"][0]
    assert row["player"] == "random" and row["seed"] == 0
    assert row["outcome"] and "score" in row and "per_decision" in row
    assert "prompt" not in row, "kilobytes of prompt per table line"


def test_the_same_seeds_reach_every_player(tmp_path):
    root = tmp_path / "cmp"
    compare.main(["--seeds", "2", "--players", "random", "--out", str(root)])
    seeds = [e.meta["seed"] for e in RunReader(root).episodes()]
    assert seeds == [0, 1]


def test_the_recorded_effort_is_the_one_the_player_got():
    """`default` means "leave the server's own alone", which is None."""
    assert compare.settings("openai-grid-default") == ("grid", None)
    assert compare.settings("openai-cues-none") == ("cues", "none")
    assert compare.settings("random") == (None, None)


def test_the_zeos_players_are_refused_on_the_step_clock(tmp_path):
    """Refused on the clock that waits, which is the same rule `agent` applies."""
    root = tmp_path / "cmp"
    with pytest.raises(SystemExit) as exit:
        compare.main(["--players", "random", "zeos-openai", "--out", str(root)])
    assert "needs --clock realtime" in str(exit.value)
    assert not root.exists(), "a typo cost an episode"


def test_a_scheduled_player_name_parses_like_its_unscheduled_namesake():
    """`zeos-openai-lead-none` is the same player as `openai-lead-none` with the
    kernel wrapped round it, so it has to resolve to the same view and effort."""
    assert compare.settings("zeos-openai-lead-none") == ("lead", "none")
    assert compare.settings("openai-lead-none") == ("lead", "none")


def test_every_clock_runs_the_same_episode_the_agent_would(tmp_path):
    """A comparison that ran an episode its own way would not be comparable
    with anything else in `runs/`."""
    root = tmp_path / "cmp"
    compare.main(
        [
            "--seeds",
            "1",
            "--players",
            "random",
            "--clock",
            "realtime",
            "--tick",
            "0.001",
            "--max-steps",
            "12",
            "--latency",
            "0",
            "--out",
            str(root),
        ]
    )
    episode = RunReader(root).episodes()[0]
    assert episode.meta["clock"] == "realtime"
    assert episode.meta["tick_seconds"] == 0.001
    assert episode.summary["ticks"] <= 12


# --- the aimed-shot measure -------------------------------------------------


def episode_with(tmp_path, shots):
    """A run where `shots` is (player column, monster column) per shot fired."""
    with RunWriter(tmp_path / "r") as run:
        for tick, (player, monster) in enumerate(shots):
            run.frame(
                {
                    "ticks": tick,
                    "score": 0,
                    "lives": 3,
                    "player": player,
                    "monsters": {1: (0, monster)},
                    "missile": None,
                    "dangers": [],
                    "can_shoot": True,
                    "won": False,
                }
            )
            run.decision(Decision(by="model", action="shoot", tick=tick))
    return RunReader(tmp_path / "r")


def test_aimed_counts_shots_with_a_monster_in_the_column(tmp_path):
    episode = episode_with(tmp_path, [(4, 4), (4, 4), (4, 0), (4, 1)])
    assert compare.aimed([episode]) == 50.0


def test_aimed_ignores_moves_and_survives_a_run_with_no_shots(tmp_path):
    with RunWriter(tmp_path / "r") as run:
        run.frame(
            {
                "ticks": 0,
                "score": 0,
                "lives": 3,
                "player": 4,
                "monsters": {1: (0, 4)},
                "missile": None,
                "dangers": [],
                "can_shoot": True,
                "won": False,
            }
        )
        run.decision(Decision(by="model", action="left", tick=0))
    assert compare.aimed([RunReader(tmp_path / "r")]) == 0.0


def test_aimed_pools_across_episodes_rather_than_averaging(tmp_path):
    """A short game must not weigh as much as a long one."""
    long = episode_with(tmp_path / "a", [(4, 4)] * 3)
    short = episode_with(tmp_path / "b", [(4, 0)])
    assert compare.aimed([long, short]) == 75.0
