# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The stick: one write, one move, and a budget that is the game's, not a runner's.

What is under test is the rule itself; the drivers are tested where they live.
"""

import pytest

from zeos_space_invaders.game import Controls, Game


def test_a_write_moves_the_ship_once():
    game = Game(seed=1)
    controls = Controls(game)
    before = game.player
    assert controls.write("left") is True
    assert game.player == before - 1
    assert controls.applied == 1


def test_a_tick_nobody_wrote_on_moves_nothing():
    """Holding the last write and replaying it emits moves nobody decided on."""
    game = Game(seed=1)
    controls = Controls(game)
    controls.write("left")
    where = game.player
    for _ in range(10):
        controls.tick()
    assert game.player == where, "the ship moved with nobody at the stick"
    assert controls.applied == 1


def test_a_budget_of_one_holds_strictly():
    game = Game(seed=1)
    controls = Controls(game, per_tick=1)
    assert controls.write("left") is True
    assert controls.full
    assert controls.write("left") is False, "a second move landed in one tick"
    controls.tick()
    assert not controls.full
    assert controls.write("left") is True


def test_no_budget_lets_a_human_out_mash_the_clock():
    """The one place a person legitimately differs: holding left crosses several
    columns while the world advances once."""
    game = Game(seed=1)
    controls = Controls(game)
    start = game.player
    for _ in range(3):
        controls.write("left")
    assert game.player == start - 3
    assert not controls.full


def test_a_write_after_the_episode_ended_is_refused_rather_than_silent():
    """A decision that arrived too late is not the same as one that was played,
    and the caller records the difference."""
    game = Game(seed=1)
    controls = Controls(game)
    game.over = True
    assert controls.write("left") is False
    assert controls.applied == 0


@pytest.mark.parametrize("per_tick", [None, 1, 3])
def test_the_budget_never_admits_more_than_it_promises(per_tick):
    game = Game(seed=2)
    controls = Controls(game, per_tick=per_tick)
    for _ in range(20):
        landed = sum(controls.write("shoot") for _ in range(5))
        assert per_tick is None or landed <= per_tick
        controls.tick()
        if game.over:
            break
