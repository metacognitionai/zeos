# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The interception search: where a shot connects, and how soon.

Incoming fire is switched off by building a game that way, not by patching the
module: what is under test is aiming, and leaving fire on would test dodging and
make the result depend on the seed's luck.
"""

import pytest

from zeos_space_invaders.game import Game, Rules, snapshot
from zeos_space_invaders.game.aim import intercept

#: The game these tests are about: the standard rules with the fire turned off.
QUIET = Rules(fire_chance=0.0)


def board(monsters, player, ticks, direction):
    game = Game(seed=7, rules=QUIET)
    game.monsters = {i: list(p) for i, p in monsters.items()}
    game.player, game.ticks, game.dir = player, ticks, direction
    return game


# --- the search itself -------------------------------------------------------


def test_it_finds_the_shot_the_old_formula_could_not():
    """The answer needs a turn spent waiting, which no formula over positions
    produces."""
    game = board({5: [4, 4]}, player=4, ticks=52, direction=1)
    shot = intercept(snapshot(game), 1)
    assert shot.column == 8 and shot.target == "5"
    assert shot.steps == 4, "four columns to walk"
    assert shot.turns == 6, "and a turn spent waiting before firing"


def test_firing_from_the_column_it_names_actually_kills():
    game = board({5: [4, 4]}, player=4, ticks=52, direction=1)
    shot = intercept(snapshot(game), 1)
    for _ in range(shot.turns):
        game.act(intercept(snapshot(game), game.dir).action)
        game.tick()
    assert not game.monsters, "the line it found does not connect"


def test_nothing_to_aim_at_is_not_an_answer():
    game = board({}, player=4, ticks=10, direction=1)
    assert intercept(snapshot(game), 1) is None


def test_a_block_that_has_not_moved_yet_has_no_lead():
    """Direction is watched for, not known: without it there is nothing to lead."""
    game = board({5: [4, 4]}, player=4, ticks=52, direction=1)
    assert intercept(snapshot(game), 0) is None


def test_a_search_that_finds_nothing_says_so_rather_than_guessing():
    game = board({5: [0, 0]}, player=8, ticks=1, direction=1)
    assert intercept(snapshot(game), 1, depth=2) is None


# --- the end-to-end check ----------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_playing_by_the_search_alone_wins(seed):
    """A recommendation that misses shows up here and nowhere else."""
    game = Game(seed=seed, rules=QUIET)
    for _ in range(400):
        if game.over:
            break
        shot = intercept(snapshot(game), game.dir)
        # Before the first march there is no direction to lead by.
        game.act(shot.action if shot else "shoot")
        game.tick()
    assert game.won, f"seed {seed}: {len(game.monsters)} left, score {game.score}"
    assert game.score == 100 and game.lives == 3


def test_it_is_cheap_enough_to_run_every_turn():
    import time

    game = Game(seed=7, rules=QUIET)
    for _ in range(6):
        game.act("shoot")
        game.tick()
    started = time.monotonic()
    for _ in range(20):
        intercept(snapshot(game), game.dir)
    assert (time.monotonic() - started) / 20 < 0.05


def test_the_search_plays_the_board_it_was_given():
    """Asserted on the column rather than on the advice because the wrong width
    does not clip the answer, it searches a different game."""
    small = Rules(w=5, h=6, monster_rows=1, monster_cols=2, monster_col_offset=1)
    game = Game(seed=3, rules=small)
    for _ in range(4):  # enough for the block to have marched, so there is a lead
        game.act("right")
        game.tick()
    shot = intercept(snapshot(game), game.dir)
    assert shot is not None, "nothing can be hit on a board this small"
    assert 0 <= shot.column < small.w, (
        f"col {shot.column} is off a {small.w}-wide board"
    )
    assert shot.steps <= small.w, "walking further than the board is wide"


# --- fire already in the air -------------------------------------------------


def test_the_line_never_steps_under_a_bomb_landing_next_turn():
    """A step that ends under a bomb already falling is no line."""
    game = board({5: [4, 6]}, player=4, ticks=52, direction=1)
    game.dangers = [[game.rules.h - 2, 5]]  # lands on col 5 this tick
    shot = intercept(snapshot(game), 1)
    assert shot is not None
    assert shot.action != "right", "told to step into the bomb"


def test_a_bomb_on_the_way_costs_a_turn_and_not_the_target():
    """With the way blocked for one tick the search waits it out -- a step aside
    or a shot standing still -- and still names the same kill."""
    clear = board({5: [4, 6]}, player=4, ticks=52, direction=1)
    blocked = board({5: [4, 6]}, player=4, ticks=52, direction=1)
    blocked.dangers = [[blocked.rules.h - 2, 5]]
    a, b = intercept(snapshot(clear), 1), intercept(snapshot(blocked), 1)
    assert a.target == b.target
    assert b.turns >= a.turns


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_following_the_line_under_fire_never_walks_into_a_bomb(seed):
    """New fire is not simulated, so a bomb can still fall on a standing ship;
    a ship that *walks* into one, though, was sent there by the line."""
    game = Game(seed=seed)
    for _ in range(300):
        if game.over:
            break
        shot = intercept(snapshot(game), game.dir)
        game.act(shot.action if shot else "shoot")
        game.tick()
    assert game.hits_walking_in == 0, f"seed {seed}: walked into a bomb"
