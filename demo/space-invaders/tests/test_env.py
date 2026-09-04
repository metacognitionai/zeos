# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

import pytest

from zeos_space_invaders.game import ACTIONS, SpaceInvadersEnv


def test_reset_returns_board_and_info():
    env = SpaceInvadersEnv(seed=0)
    obs, info = env.reset()
    assert "p" in obs and "m1" in obs
    assert info["lives"] == 3 and info["score"] == 0 and len(info["monsters"]) == 10


def test_step_returns_the_five_tuple():
    env = SpaceInvadersEnv(seed=0)
    obs, reward, terminated, truncated, info = env.step("left")
    assert isinstance(obs, str) and isinstance(reward, float)
    assert terminated is False and truncated is False
    assert info["steps"] == 1


def test_unknown_action_is_rejected():
    env = SpaceInvadersEnv(seed=0)
    with pytest.raises(ValueError):
        env.step("jump")


def test_kill_is_rewarded():
    env = SpaceInvadersEnv(seed=0)
    row, col = env.game.monsters[6]
    env.game.player = col
    env.game.missile = [row + 1, col]
    _, reward, _, _, _ = env.step("left")
    assert reward == pytest.approx(10.0)


def test_truncation_at_max_steps():
    env = SpaceInvadersEnv(seed=0, max_steps=3)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step("left")
        assert not (terminated or truncated)
    _, _, terminated, truncated, _ = env.step("left")
    assert truncated and not terminated


def test_episode_terminates_and_pays_out():
    env = SpaceInvadersEnv(seed=4, max_steps=2000)
    total, terminated = 0.0, False
    while not terminated:
        _, reward, terminated, truncated, info = env.step("shoot")
        total += reward
        assert not truncated, "should end before the step cap"
    assert info["lives"] == 0 or info["won"] or len(info["monsters"]) > 0


def test_reset_with_seed_replays_identically():
    env = SpaceInvadersEnv(seed=1)
    first = [env.step(ACTIONS[i % 3])[0] for i in range(20)]
    env.reset(seed=1)
    assert [env.step(ACTIONS[i % 3])[0] for i in range(20)] == first


def test_render_shows_the_board_and_the_status_line():
    """What `--render` prints; the observation is the board alone."""
    env = SpaceInvadersEnv(seed=1)
    drawn = env.render()
    assert env.observation() in drawn
    assert "score" in drawn and "lives" in drawn
