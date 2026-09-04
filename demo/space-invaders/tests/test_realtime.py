# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Real-time mode: the world runs on its own clock and does not wait.

Timing assertions are deliberately loose — these run on a shared machine.
"""

from zeos_space_invaders.clocks import RealtimeRunner
from zeos_space_invaders.game import Game, snapshot
from zeos_space_invaders.players import RandomPlayer
from zeos_space_invaders.runlog import RunReader, RunWriter


def test_snapshot_matches_the_env_info_shape():
    """Players are written against SpaceInvadersEnv.info(); this must match it."""
    from zeos_space_invaders.game import SpaceInvadersEnv

    assert set(snapshot(Game(seed=1))) >= set(SpaceInvadersEnv(seed=1).info()) - {
        "steps"
    }


def test_an_episode_finishes_and_reports_timing():
    runner = RealtimeRunner(RandomPlayer(seed=1), seed=1, tick_seconds=0.005)
    summary = runner.run()
    assert summary["outcome"] in {"won", "lost", "timeout"}
    assert summary["ticks"] > 0 and summary["decisions"] > 0
    assert summary["mean_ticks_waited"] >= 0


def test_the_world_does_not_wait_for_a_slow_agent():
    """The whole point: thinking costs ticks."""
    quick = RealtimeRunner(
        RandomPlayer(seed=1, latency=0.0), seed=1, tick_seconds=0.02
    ).run()
    slow = RealtimeRunner(
        RandomPlayer(seed=1, latency=0.1), seed=1, tick_seconds=0.02
    ).run()
    assert slow["mean_ticks_waited"] > quick["mean_ticks_waited"]
    assert slow["decisions"] < quick["decisions"]


def test_a_slow_agent_gets_fewer_actions_per_tick():
    slow = RealtimeRunner(
        RandomPlayer(seed=2, latency=0.05), seed=2, tick_seconds=0.01
    ).run()
    assert slow["actions_per_tick"] < 1.0


def test_every_decision_says_when_it_was_taken_and_when_it_landed(tmp_path):
    with RunWriter(tmp_path / "rt") as run:
        summary = RealtimeRunner(RandomPlayer(seed=3), seed=3, tick_seconds=0.005).run(
            run=run
        )
    episode = RunReader(tmp_path / "rt")
    decisions = list(episode.decisions())
    assert len(decisions) == summary["decisions"]
    assert all(
        {"latency", "tick", "tick_applied", "applied"} <= set(d) for d in decisions
    )


def test_the_timeline_is_complete_even_between_two_slow_decisions(tmp_path):
    """The reason frames exist: a slow agent leaves ticks nobody decided on."""
    with RunWriter(tmp_path / "slow") as run:
        summary = RealtimeRunner(
            RandomPlayer(seed=3, latency=0.05), seed=3, tick_seconds=0.005
        ).run(run=run)
    episode = RunReader(tmp_path / "slow")
    frames = list(episode.frames())
    assert summary["decisions"] < summary["ticks"], "not actually a slow agent"
    assert [f["tick"] for f in frames] == list(range(summary["ticks"] + 1))
    for decision in episode.decisions():
        assert episode.frame_at(decision["tick"]), "a decision with no world"


def test_token_usage_is_summed_over_a_realtime_episode():
    """A run that reports no tokens reads as a free one, and it is not."""
    from stubs import claude_player

    summary = RealtimeRunner(
        claude_player(*["shoot"] * 200), seed=8, tick_seconds=0.005, max_seconds=0.2
    ).run()
    assert summary["usage"]["input_tokens"] == 100 * summary["decisions"]
    assert summary["usage"]["output_tokens"] == 5 * summary["decisions"]


def test_rendering_draws_a_frame_without_changing_the_run(capsys):
    """Drawn from both threads, so a fast agent does not look laggy."""
    drawn = RealtimeRunner(
        RandomPlayer(seed=6), seed=6, tick_seconds=0.005, max_seconds=0.2, render=True
    ).run()
    printed = capsys.readouterr().out
    # The same three lines a human sees: board, status, what was played.
    assert "left   missile" in printed, "not the shared status line"
    assert drawn["ticks"] > 0


def test_the_footer_reports_this_tick_and_not_the_last_one(capsys):
    """Edge-triggered, like the stick it reports on: a tick nobody acted on draws
    a dash."""
    from zeos_space_invaders.clocks.render import played
    from zeos_space_invaders.game import Game

    game = Game(seed=1)
    assert played(game, "left") == "tick    0   left "
    assert played(game, None) == "tick    0   —", "the footer held a stale action"

    # Through the runner: a slow agent leaves ticks nobody acted on.
    frames = RealtimeRunner(
        RandomPlayer(seed=6, latency=0.05),
        seed=6,
        tick_seconds=0.005,
        max_seconds=0.2,
        render=True,
    )
    frames.run()
    printed = capsys.readouterr().out
    assert "   —" in printed, "no tick went by without a decision"


def test_the_clock_thread_stops_when_the_game_does():
    import threading

    before = threading.active_count()
    RealtimeRunner(RandomPlayer(seed=4), seed=4, tick_seconds=0.005).run()
    assert threading.active_count() <= before


# --- the actions-per-tick throttle ------------------------------------------


def test_uncapped_a_fast_agent_gets_many_actions_per_tick():
    r = RealtimeRunner(
        RandomPlayer(seed=5, latency=0.001), seed=5, tick_seconds=0.05, max_seconds=1
    )
    r.run()
    assert _worst_tick(r) > 1


def test_capping_at_one_holds_strictly():
    """Grouped by the tick the action LANDED on, not the tick it was decided on."""
    r = RealtimeRunner(
        RandomPlayer(seed=5, latency=0.001),
        seed=5,
        tick_seconds=0.05,
        max_seconds=1,
        actions_per_tick=1,
    )
    r.run()
    assert _worst_tick(r) == 1


def test_capping_at_two_holds_strictly():
    r = RealtimeRunner(
        RandomPlayer(seed=5, latency=0.001),
        seed=5,
        tick_seconds=0.05,
        max_seconds=1,
        actions_per_tick=2,
    )
    r.run()
    assert _worst_tick(r) <= 2


def test_a_throttled_action_is_delayed_not_discarded():
    r = RealtimeRunner(
        RandomPlayer(seed=5, latency=0.001),
        seed=5,
        tick_seconds=0.05,
        max_seconds=1,
        actions_per_tick=1,
    )
    r.run()
    live = [rec for rec in r.records if rec.applied]
    assert live, "nothing was applied at all"
    assert all(rec.tick_applied >= rec.tick for rec in live)
    assert any(rec.tick_applied > rec.tick for rec in live), (
        "throttling should push some actions to a later tick"
    )


def _worst_tick(runner):
    from collections import Counter

    landed = Counter(rec.tick_applied for rec in runner.records if rec.applied)
    return max(landed.values()) if landed else 0
