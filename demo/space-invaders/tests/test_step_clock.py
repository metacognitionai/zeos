# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The step clock: one action, one tick, the world waits.

The deterministic mode, and therefore the one a comparison runs in. What is
tested here is the episode -- that it reaches a verdict, that the log it leaves
is complete, and that the summary answers the same questions the other clocks'
summaries do.
"""

import pytest
from stubs import claude_player

from zeos_space_invaders.clocks import run_episode
from zeos_space_invaders.players import RandomPlayer


@pytest.mark.parametrize("effort", ["high", "none"])
def test_a_whole_episode_runs_to_a_verdict(effort, run, written):
    p = claude_player(*["shoot"] * 600, effort=effort)
    summary = run.finish(run_episode(p, seed=3, max_steps=400, run=run, verbose=False))
    episode = written()

    assert summary["outcome"] in {"won", "lost", "truncated"}
    assert summary["decisions"] == len(list(episode.decisions()))
    assert summary["usage"]["input_tokens"] == 100 * summary["decisions"]
    # A frame per tick, plus the one the episode started on.
    assert len(list(episode.frames())) == summary["ticks"] + 1
    assert episode.summary["score"] == summary["score"]


def test_the_random_baseline_drives_a_full_episode(run, written):
    summary = run.finish(
        run_episode(RandomPlayer(seed=7), seed=7, run=run, verbose=False)
    )
    assert summary["outcome"] in {"won", "lost", "truncated"}
    assert summary["usage"] == {}  # nothing was billed
    assert summary["decisions"] == len(list(written().decisions()))


def test_one_action_is_exactly_one_tick(run, written):
    """The property the whole mode rests on: nothing is ever late here."""
    summary = run.finish(
        run_episode(RandomPlayer(seed=1), seed=1, max_steps=25, run=run, verbose=False)
    )
    assert summary["ticks"] == summary["decisions"]
    for decision in written().decisions():
        assert decision["tick_applied"] == decision["tick"]


def test_the_episode_stops_at_max_steps():
    summary = run_episode(RandomPlayer(seed=2), seed=2, max_steps=12, verbose=False)
    assert summary["outcome"] == "truncated" and summary["decisions"] == 12


def test_the_reward_is_summed_onto_the_verdict():
    """The env shapes a reward; only this clock has one to report."""
    summary = run_episode(RandomPlayer(seed=7), seed=7, max_steps=60, verbose=False)
    assert "reward" in summary and isinstance(summary["reward"], float)


def test_it_runs_without_a_writer_at_all():
    """A sweep drives episodes with nothing to log to."""
    assert (
        run_episode(RandomPlayer(seed=3), seed=3, max_steps=5, verbose=False)[
            "decisions"
        ]
        == 5
    )


def test_verbose_and_render_do_not_change_the_outcome(capsys):
    quiet = run_episode(RandomPlayer(seed=4), seed=4, max_steps=10, verbose=False)
    loud = run_episode(RandomPlayer(seed=4), seed=4, max_steps=10, verbose=True)
    drawn = run_episode(
        RandomPlayer(seed=4), seed=4, max_steps=10, render=True, delay=0.001
    )
    assert quiet["score"] == loud["score"] == drawn["score"]
    assert capsys.readouterr().out.count("\n") > 10


def test_every_clock_answers_the_same_questions():
    """Reported as the constants they are under this clock rather than omitted: a
    blank cell reads as "not measured" when the truth is "always 1 here"."""
    from zeos_space_invaders.clocks.realtime import RealtimeRunner
    from zeos_space_invaders.game import Rules

    quiet = Rules(fire_chance=0.0)
    stepped = run_episode(
        RandomPlayer(seed=1), seed=1, max_steps=3, verbose=False, rules=quiet
    )
    timed = RealtimeRunner(
        RandomPlayer(seed=1), seed=1, tick_seconds=0.0, max_ticks=3, rules=quiet
    ).run()
    shared = {
        "outcome",
        "score",
        "lives",
        "monsters_left",
        "ticks",
        "decisions",
        "actions_per_tick",
        "dropped_after_game_over",
        "unparseable",
        "seconds",
        "usage",
    }
    assert shared <= set(stepped), f"the step clock omits {shared - set(stepped)}"
    assert shared <= set(timed), f"the realtime clock omits {shared - set(timed)}"
