# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""`play`: a human at the keyboard, recorded the same way a model is.

The screen is faked rather than driven -- what is being tested is the loop and
what it logs, not curses. The two constants are patched to zero so a handful of
keystrokes covers several world ticks instead of taking a second.
"""

import curses
import json

import pytest

from zeos_space_invaders import play
from zeos_space_invaders.runlog import RunReader, RunWriter


class FakeScreen:
    """Absorbs everything `_draw` does and hands back canned keystrokes."""

    def __init__(self, *keys):
        self.keys = list(keys)
        self.drawn = []

    def nodelay(self, flag):
        pass

    def keypad(self, flag):
        pass

    def erase(self):
        pass

    def addstr(self, row, col, text):
        self.drawn.append(text)

    def refresh(self):
        pass

    def getch(self):
        return self.keys.pop(0) if self.keys else ord("q")


@pytest.fixture(autouse=True)
def instant(monkeypatch):
    monkeypatch.setattr(curses, "curs_set", lambda visibility: None)
    monkeypatch.setattr(play, "POLL_SECONDS", 0)
    monkeypatch.setattr(play, "TICK_SECONDS", 0)


def played(*keys):
    return FakeScreen(*keys)


def test_a_human_run_reads_back_like_any_other(tmp_path):
    keys = [curses.KEY_LEFT, ord(" "), curses.KEY_RIGHT, ord("a"), ord("q")]
    with RunWriter(tmp_path / "r", {"player": "human", "clock": "human"}) as run:
        play._loop(played(*keys), seed=1, run=run)

    episode = RunReader(tmp_path / "r")
    decisions = list(episode.decisions())
    assert [d["action"] for d in decisions] == ["left", "shoot", "right", "left"]
    assert {d["by"] for d in decisions} == {"human"}
    assert episode.summary["decisions"] == 4
    assert episode.summary["outcome"] == "quit"
    for decision in decisions:
        assert episode.frame_at(decision["tick"]), "a decision with no world"


def test_several_keystrokes_can_land_on_one_tick(tmp_path, monkeypatch):
    """Unlimited actions per tick is the human's advantage over the agent."""
    monkeypatch.setattr(play, "TICK_SECONDS", 1e6)  # no tick will come due
    with RunWriter(tmp_path / "r") as run:
        play._loop(played(*[curses.KEY_LEFT] * 3, ord("q")), seed=1, run=run)
    ticks = [d["tick"] for d in RunReader(tmp_path / "r").decisions()]
    assert ticks == [0, 0, 0]


def test_an_unknown_key_is_ignored_rather_than_played(tmp_path):
    with RunWriter(tmp_path / "r") as run:
        play._loop(played(ord("z"), ord("j"), ord("q")), seed=1, run=run)
    assert list(RunReader(tmp_path / "r").decisions()) == []


def test_escape_quits_like_q(tmp_path):
    with RunWriter(tmp_path / "r") as run:
        play._loop(played(27), seed=1, run=run)
    assert RunReader(tmp_path / "r").summary["outcome"] == "quit"


def test_playing_to_the_end_reports_the_verdict(tmp_path):
    """Nothing but shooting, until the monsters land or die."""
    with RunWriter(tmp_path / "r") as run:
        play._loop(played(*[ord(" ")] * 4000), seed=1, run=run)
    episode = RunReader(tmp_path / "r")
    assert episode.summary["outcome"] in {"won", "lost"}
    assert episode.summary["ticks"] > 1
    last = list(episode.frames())[-1]
    assert last["over"] is True


def test_recording_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curses, "wrapper", lambda fn, **kw: fn(played(ord("q")), **kw))
    play.main(["--no-record", "--seed", "2"])
    assert not (tmp_path / "runs").exists()


def test_by_default_a_game_is_recorded(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        curses, "wrapper", lambda fn, **kw: fn(played(ord(" "), ord("q")), **kw)
    )
    play.main(["--seed", "2"])

    made = list((tmp_path / "runs").iterdir())
    assert len(made) == 1 and made[0].name.endswith("-human-seed2")
    episode = RunReader(made[0])
    assert episode.meta["player"] == "human" and episode.meta["seed"] == 2
    assert episode.summary["decisions"] == 1
    # The same lines `agent` prints: a game you played is a run like any other.
    printed = capsys.readouterr().out
    assert "10 monsters left" in printed and "lives 3" in printed
    assert "summary written to" in printed


def test_a_settings_file_hands_a_person_the_board_a_model_was_given(
    tmp_path, monkeypatch
):
    """`play` reads `board` alone: a keyboard has no view, effort or history."""
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "board": {"width": 12, "height": 16, "march_group": 1},
                "play": {"view": "lead", "effort": "none"},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curses, "wrapper", lambda fn, **kw: fn(played(ord("q")), **kw))
    play.main(["--settings", str(settings), "--seed", "2"])

    meta = RunReader(next((tmp_path / "runs").iterdir())).meta
    assert (meta["width"], meta["height"]) == (12, 16) and meta["march_group"] == 1


def test_the_module_runs_the_game_without_an_installed_script(tmp_path, monkeypatch):
    """`python -m zeos_space_invaders` is `play`, not `agent`."""
    import runpy
    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["zeos_space_invaders", "--seed", "9"])
    monkeypatch.setattr(
        curses, "wrapper", lambda fn, **kw: fn(played(ord(" "), ord("q")), **kw)
    )
    runpy.run_module("zeos_space_invaders", run_name="__main__")
    made = next((tmp_path / "runs").iterdir())
    assert made.name.endswith("-human-seed9")
