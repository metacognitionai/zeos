# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

import pytest

from zeos_space_invaders.game import (
    DEFAULTS,
    LEFT,
    RIGHT,
    SHOOT,
    Game,
    H,
    Rules,
    W,
)


def test_initial_board():
    game = Game(seed=1)
    assert len(game.monsters) == 10
    assert set(game.monsters) == set(range(1, 11))
    assert game.render().splitlines()[-1].strip().split() == ["."] * (W // 2) + [
        "p"
    ] + ["."] * (W // 2)


def test_seed_is_reproducible():
    a, b = Game(seed=42), Game(seed=42)
    for _ in range(50):
        a.tick()
        b.tick()
    assert a.render() == b.render() and a.score == b.score


def test_movement_clamps_to_walls():
    game = Game(seed=1)
    for _ in range(W):
        game.act(LEFT)
    assert game.player == 0
    for _ in range(2 * W):
        game.act(RIGHT)
    assert game.player == W - 1


def test_only_one_missile_in_flight():
    game = Game(seed=1)
    game.act(SHOOT)
    first = list(game.missile)
    game.act(SHOOT)
    assert game.missile == first


def test_shot_kills_the_monster_above():
    game = Game(seed=1)
    row, col = game.monsters[6]  # bottom-row monster
    game.player = col
    game.missile = [row + 1, col]
    game.tick()
    assert 6 not in game.monsters and game.score == 10


def test_monsters_speed_up_as_they_die():
    game = Game(seed=1)
    intervals = []
    for _ in range(0, 10, 3):
        for key in list(game.monsters)[:3]:
            del game.monsters[key]
        intervals.append(game._interval())
    assert intervals == sorted(intervals, reverse=True)
    assert intervals[-1] == 1


def test_clearing_the_board_wins():
    game = Game(seed=1)
    last = max(game.monsters)
    for key in list(game.monsters)[:-1]:
        del game.monsters[key]
    row, col = game.monsters[last]
    game.missile = [row + 1, col]
    game.tick()
    assert game.over and game.won


def test_every_game_terminates():
    for seed in range(20):
        game = Game(seed=seed)
        for _ in range(1000):
            game.tick()
            if game.over:
                break
        assert game.over, f"seed {seed} never ended"


def test_winning_while_taking_a_hit_still_ends_the_game():
    """The missile ends the game; a survivable hit on the same tick must not undo it."""
    game = Game(seed=1)
    game.rng.random = lambda: 1.0  # no fresh monster fire
    game.monsters = {1: [4, 4]}  # last monster, one row above the missile
    game.player = 4
    game.missile = [5, 4]
    game.dangers = [[H - 2, 4]]  # lands on us this tick, and we survive it
    game.tick()
    assert game.monsters == {} and game.won
    assert game.over, "a win must stay a win even when you take a hit on the way"
    assert game.lives == 2


def test_acting_after_the_game_is_over_does_nothing():
    """The runners can land an action after the last tick; it must be inert."""
    game = Game(seed=1)
    game.lives = 0
    game.over = True
    before = game.player
    game.act(LEFT)
    game.act(SHOOT)
    assert game.player == before and game.missile is None


def test_a_game_is_the_rules_it_was_built_with():
    """`Rules()` is the game as it has always been played, which keeps every
    existing score reproducible."""
    assert Game(seed=1).rules == DEFAULTS
    tiny = Rules(w=5, h=5, monster_rows=1, monster_cols=2, monster_col_offset=1)
    game = Game(seed=1, rules=tiny)
    assert game.rules is tiny
    assert len(game.monsters) == 2
    assert game.player == 2  # the middle of a 5-wide board, not a 9-wide one
    assert len(game.render().splitlines()) == 5


def test_the_rules_reach_every_part_of_a_tick():
    """A field still read off the module shows up here and nowhere else, because
    the defaults and a changed value agree wherever the board is not involved."""
    rules = Rules(w=4, h=6, monster_rows=1, monster_cols=1, monster_col_offset=0)
    game = Game(seed=1, rules=rules)
    game.act(RIGHT)
    game.act(RIGHT)
    game.act(RIGHT)
    assert game.player == 3, "the player walked off a 4-wide board"
    game.act(SHOOT)
    assert game.missile == [4, 3], "the missile spawned on the wrong row"


def test_a_faster_fall_shortens_the_warning():
    """A method rather than a module function: one reading the default `h` and
    `danger_rows` would answer for a board nobody is playing on."""
    slow = Rules(danger_rows=1)
    fast = Rules(danger_rows=3)
    assert [slow.turns_to_land(r) for r in range(1, 7)] == [6, 5, 4, 3, 2, 1]
    assert [fast.turns_to_land(r) for r in range(1, 7)] == [2, 2, 2, 1, 1, 1]
    assert Rules(danger_rows=2).turns_to_land(2) == 3, "rounded down, not up"
    # A shorter board is a shorter warning, for the same fall speed.
    assert Rules(h=4).turns_to_land(1) == 2


def test_explicit_monsters_override_the_grid():
    """One monster over the gun, on a board with nowhere else to be, is the
    shortest route to a preemption worth reading."""
    rules = Rules(w=3, h=4, monsters=((0, 1),))
    game = Game(seed=1, rules=rules)
    assert game.monsters == {1: [0, 1]}


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"w": 0}, "at least 1x2"),
        ({"h": 1}, "at least 1x2"),
        ({"lives": 0}, "lives must be >= 1"),
        ({"danger_rows": 0}, "danger_rows must be >= 1"),
        ({"fire_chance": 1.5}, "fire_chance must be in"),
        ({"w": 3, "monster_cols": 5}, "outside a 3x8 board"),
        ({"h": 3, "monster_rows": 4}, "outside a 9x3 board"),
    ],
)
def test_a_board_that_cannot_hold_the_game_is_refused(kwargs, message):
    """A contract at construction, so a typo at a command line is reported before
    an episode starts."""
    with pytest.raises(ValueError, match=message):
        Rules(**kwargs)


def test_fire_chance_is_the_game_it_was_built_with(tmp_path):
    from zeos_space_invaders.game import FIRE_CHANCE, LIVES

    assert Game(seed=1).fire_chance == FIRE_CHANCE
    assert Game(seed=1, rules=Rules(fire_chance=0.75)).fire_chance == 0.75

    quiet = Game(seed=1, rules=Rules(fire_chance=0.0))
    loud = Game(seed=1, rules=Rules(fire_chance=1.0))
    for _ in range(30):
        quiet.tick()
        loud.tick()
    assert not quiet.dangers and quiet.lives == LIVES, "fire fell where there is none"
    # Counted by what it did rather than what is on the board: each hit clears
    # the bomb that landed.
    assert loud.over and loud.lives == 0, "no fire fell in a game that is all fire"


def test_two_games_at_different_fire_rates_do_not_share_state():
    """The obvious way to get this wrong is a module constant one of them set."""
    quiet = Game(seed=1, rules=Rules(fire_chance=0.0))
    loud = Game(seed=1, rules=Rules(fire_chance=1.0))
    for _ in range(10):
        loud.tick()
        quiet.tick()
    assert quiet.fire_chance == 0.0 and loud.fire_chance == 1.0
    assert not quiet.dangers


def test_two_bombs_on_one_tick_cannot_take_lives_below_zero():
    game = Game(seed=1)
    game.rng.random = lambda: 1.0
    game.lives = 1
    game.dangers = [[H - 2, game.player], [H - 2, game.player]]
    game.tick()
    assert game.over and game.lives == 0


def test_a_hit_says_whether_the_ship_stood_or_stepped_into_it():
    """Counted apart because the three arms of the comparison lose lives in very
    different proportions of the two."""
    game = Game(seed=1)
    game.rng.random = lambda: 1.0
    game.dangers = [[H - 2, game.player]]
    game.tick()
    assert (game.hits_standing, game.hits_walking_in) == (1, 0)
    game.dangers = [[H - 2, game.player + 1]]
    game.act(RIGHT)
    game.tick()
    assert (game.hits_standing, game.hits_walking_in) == (1, 1)
    assert game.lives == 1
