# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The views: what the model is shown, and how the prompt is assembled."""

import pytest

from zeos_space_invaders.game import Game, Rules, SpaceInvadersEnv
from zeos_space_invaders.players import OpenAIPlayer, PromptPlayer
from zeos_space_invaders.utils.views import (
    VIEWS,
    CoordsView,
    CuesView,
    GridView,
    LeadView,
)


@pytest.fixture
def board():
    """A game a few steps in, so a missile and monster movement are in play."""
    return frames(6)[-1]


def frames(n):
    """n successive (obs, info) states, so views that infer motion can see it."""
    env = SpaceInvadersEnv(seed=1)
    obs, info = env.reset()
    out = [(obs, info)]
    for step in range(n):
        obs, _, _, _, info = env.step("shoot" if step == 0 else "right")
        out.append((obs, info))
    return out


# --- what each view shows ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_every_view_reports_the_shared_status_line(name, board):
    obs, info = board
    state = VIEWS[name]().state(obs, info)
    assert f"step {info['steps']}" in state
    assert f"score {info['score']}" in state
    assert f"lives {info['lives']}" in state


def test_grid_shows_the_board_and_nothing_invented(board):
    obs, info = board
    assert GridView().state(obs, info).endswith(obs)


def test_coords_states_every_monster_position(board):
    obs, info = board
    state = CoordsView().state(obs, info)
    for ident, (row, col) in info["monsters"].items():
        assert f"m{ident} at row {row} col {col}" in state
    assert f"you are at col {info['player']}" in state


def test_coords_carries_no_picture(board):
    """If the grid leaks in, the comparison between views means nothing."""
    obs, info = board
    state = CoordsView().state(obs, info)
    assert "   .   ." not in state


def test_cues_names_a_monster_in_your_column_when_there_is_one():
    game = Game(seed=1)
    info = _info(game)
    info["player"] = game.monsters[6][1]  # stand under m6
    state = CuesView().state(game.render(), info)
    assert "monster in your column: m6" in state


def test_cues_says_none_when_the_column_is_empty():
    game = Game(seed=1)
    info = _info(game)
    info["player"] = 0  # monsters start at columns 2-6
    assert "monster in your column: none" in CuesView().state(game.render(), info)


def test_cues_counts_the_turns_until_fire_lands():
    game = Game(seed=1)
    game.dangers = [[4, 3]]
    info = _info(game)
    info["player"] = 3
    # row 4 -> row 7 is three rows, so three turns
    assert "incoming fire in your column: lands in 3 turns" in CuesView().state(
        game.render(), info
    )


# --- the lead view, which is the one that changed the result ----------------


def test_lead_has_no_aim_point_before_the_block_has_moved():
    """Direction is unknowable from a single frame; it must not guess."""
    game = Game(seed=1)
    state = LeadView().state(game.render(), _info(game))
    assert "aim point: not known yet" in state


def test_the_aim_point_names_a_column_a_shot_really_connects_from():
    """Checked against the game, not against the formula that produced it."""
    from zeos_space_invaders.game import Game, snapshot

    view = LeadView()
    game = Game(seed=7)
    for _ in range(9):  # long enough for a march to happen
        game.act("right")
        game.tick()
        state = view.state(game.render(), snapshot(game))
    assert view.direction != 0, "the block moved, so direction must be known"

    line = next(x for x in state.splitlines() if x.startswith("aim point"))
    column = int(line.split("col ")[1].split(",")[0].split(" ")[0])
    target = line.split("m")[1].split(" ")[0]

    # Walk there and follow the line; the monster it named has to die.
    for _ in range(20):
        info = snapshot(game)
        line = next(
            x
            for x in view.state(game.render(), info).splitlines()
            if x.startswith("aim point")
        )
        if target not in game.monsters:
            break
        game.act(
            "shoot"
            if "— shoot" in line
            else ("right" if info["player"] < column else "left")
        )
        game.tick()
    assert target not in game.monsters, f"m{target} survived the aim point"


def test_the_aim_point_does_not_send_you_chasing_what_you_cannot_catch():
    """With three monsters or fewer the block marches every tick -- the player's
    own speed -- so "walk to where it will be" never converges."""
    from zeos_space_invaders.game import Game, snapshot

    view = LeadView()
    view.direction = 1
    # Fire off: a bomb in the way changes the line, which has a test of its own.
    game = Game(seed=7, rules=Rules(fire_chance=0.0))
    game.monsters, game.player, game.ticks, game.dir = {5: [4, 4]}, 4, 52, 1

    targets = []
    for _ in range(8):
        if not game.monsters:
            break
        info = snapshot(game)
        line = next(
            x
            for x in view.state(game.render(), info).splitlines()
            if x.startswith("aim point")
        )
        targets.append(line.split("col ")[1].split(",")[0].split(" ")[0])
        # The line says either `— shoot` or `play <move> now`.
        game.act("shoot" if "— shoot" in line else line.split("play ")[1].split()[0])
        game.tick()

    assert len(set(targets)) == 1, f"the aim column wandered: {targets}"
    assert not game.monsters, "following the aim point did not land the kill"


def test_lead_remembers_direction_between_marches():
    """The block moves every few turns; direction must survive the turns in between."""
    view = LeadView()
    states = [view.state(obs, info) for obs, info in frames(9)]
    learned = next(i for i, s in enumerate(states) if "direction not seen yet" not in s)
    assert learned > 0, "direction cannot be known from the very first frame"
    # once learned it must never revert, including on turns with no movement
    assert all("direction not seen yet" not in s for s in states[learned:])


# --- prompt assembly --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_the_prompt_carries_the_view_format_and_the_shared_rules(name):
    player = OpenAIPlayer(effort="none", view=VIEWS[name](), client=object())
    sections = [line for line in player.prompt.splitlines() if line.startswith("## ")]
    assert "## Your actions" in sections
    assert "## Winning and losing" in sections
    assert "## Your reply" in sections
    assert sections[0] in ("## The board", "## What you are told")


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_the_prompt_never_describes_a_format_the_view_does_not_produce(name):
    """A coords prompt showing an ascii board would teach the model a lie."""
    player = OpenAIPlayer(effort="none", view=VIEWS[name](), client=object())
    shows_grid = "   .   .  m1" in player.prompt
    assert shows_grid == (name == "grid")


def test_views_differ_only_in_format_and_decision_sections():
    """The point of the split: the shared rules cannot drift between views."""
    prompts = {
        n: OpenAIPlayer(effort="none", view=VIEWS[n](), client=object()).prompt
        for n in sorted(VIEWS)
    }
    shared = "## Your actions"
    tails = {
        n: text[text.index(shared) : text.index("## How to choose")]
        for n, text in prompts.items()
    }
    assert len(set(tails.values())) == 1, "the shared rules block differs between views"


def test_the_decision_section_is_the_one_that_varies():
    drill = OpenAIPlayer(effort="none", view=VIEWS["drill"](), client=object()).prompt
    cues = OpenAIPlayer(effort="none", view=VIEWS["cues"](), client=object()).prompt
    assert "you must move" in drill and "you must move" not in cues


# --- the prompt states the game that will run, not the defaults --------------

#: Every rule the prompt states, turned away from its default at once, so a
#: hardcoded number shows up as wrong rather than as a coincidental match.
ODD = Rules(
    w=5,
    h=5,
    monster_rows=1,
    monster_cols=2,
    monster_col_offset=1,
    lives=9,
    fire_chance=1.0,
)


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_no_default_geometry_survives_into_a_resized_prompt(name):
    """Asserted as an absence because a hardcoded fact does not raise; it tells
    the model something false about the game it is playing."""
    prompt = OpenAIPlayer(
        effort="none", view=VIEWS[name](), client=object(), rules=ODD
    ).prompt
    for stale in ("row 7", "col 8", "column 8", "15%", "ten monsters", "seven turns"):
        assert stale not in prompt, f"{name} still states {stale!r}"


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_the_prompt_states_the_rules_it_was_built_with(name):
    prompt = OpenAIPlayer(
        effort="none", view=VIEWS[name](), client=object(), rules=ODD
    ).prompt
    assert "5 columns" in prompt and "row 4" in prompt
    assert "100% chance" in prompt, "the bomb rate is a rule the model is told"
    assert "you start with 9" in prompt


def test_the_prompt_still_states_the_defaults_it_used_to_hardcode():
    """A default run must still be told the same game, or every score recorded
    before is unmoored from the prompt that produced it."""
    prompt = OpenAIPlayer(effort="none", view=VIEWS["grid"](), client=object()).prompt
    flat = " ".join(prompt.split())
    assert "The board is 9 columns wide and 8 rows tall" in flat
    assert "reaches row 7 costs you a life" in flat
    assert "already at column 8" in flat
    assert "destroy all 10 monsters" in flat
    assert "45% chance" in flat and "15%" not in flat


def test_the_grid_shown_in_the_prompt_is_the_board_that_will_be_dealt():
    """Rendered from a game rather than drawn by hand, so it cannot disagree."""
    from zeos_space_invaders.game.rules import Game

    prompt = OpenAIPlayer(
        effort="none", view=VIEWS["grid"](), client=object(), rules=ODD
    ).prompt
    assert Game(seed=0, rules=ODD).render() in prompt


def test_every_placeholder_in_every_prompt_file_is_filled():
    """`str.format` raises for an unknown name, so this catches the other
    direction: a brace meant as prose that is silently eaten."""
    for name in sorted(VIEWS):
        prompt = OpenAIPlayer(
            effort="none", view=VIEWS[name](), client=object(), rules=ODD
        ).prompt
        assert "{" not in prompt and "}" not in prompt, name


def test_history_is_compact_for_the_coordinate_views(board):
    """Five turns of full coordinates would swamp the prompt."""
    obs, info = board
    assert len(CoordsView().history(obs, info)) < len(CoordsView().state(obs, info)) / 2
    assert GridView().history(obs, info) == obs


def test_the_player_uses_the_view_for_both_now_and_history(board):
    obs, info = board
    player = PromptPlayer(prompt="RULES", view=CoordsView())
    player.record(obs, info, "left", "", "stop", {})
    rendered = player.render_turn(obs, info)
    assert "Recent turns" in rendered
    assert "you col" in rendered  # the compact history form
    assert "monsters: m" in rendered  # the full form for the current turn


def _info(game):
    from zeos_space_invaders.game import snapshot

    return snapshot(game)


def test_the_aim_point_names_the_first_step_of_the_line_not_a_direction():
    """When a bomb is about to land in the column the line walks through, the
    first step is the other way or a standstill, and the text has to say so."""
    from zeos_space_invaders.game import Game, snapshot

    view = LeadView()
    view.direction = 1
    # The line is four steps right to the wall, with a bomb about to land on the
    # first of those columns, so the first step cannot be `right`.
    game = Game(seed=7)
    game.monsters, game.player, game.ticks, game.dir = {5: [4, 4]}, 4, 52, 1
    game.dangers = [[game.rules.h - 2, 5]]
    line = next(
        x
        for x in view.state(game.render(), snapshot(game)).splitlines()
        if x.startswith("aim point")
    )
    assert "play " in line and "play right" not in line, line
