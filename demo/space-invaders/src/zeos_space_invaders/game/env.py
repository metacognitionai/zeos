# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Step-based environment wrapper. Stdlib only.

The `reset` / `step` signatures follow the Gymnasium convention, but nothing
here depends on that package.

    env = SpaceInvadersEnv(seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("shoot")

No wall clock and therefore no deadline: a policy may take as long as it likes,
and one `step` is exactly one action plus one world tick. That is stricter than
`play.py`, where a human can squeeze many keystrokes into one 0.2 s tick — bear
it in mind before comparing a policy's score against your own.
"""

from .controls import Controls
from .rules import LEFT, RIGHT, SHOOT, Game, Rules

ACTIONS = (LEFT, RIGHT, SHOOT)
MAX_STEPS = 300  # default episode cap; `agent` and `compare` share it


class SpaceInvadersEnv:
    """One `step` is one action plus one world tick.

    The observation is the rendered board; `info` carries the same state in
    structured form.
    """

    action_space = ACTIONS

    def __init__(
        self,
        seed=None,
        max_steps=MAX_STEPS,
        kill_reward=10.0,
        life_penalty=-5.0,
        win_reward=100.0,
        loss_penalty=-10.0,
        rules=None,
    ):
        self.max_steps = max_steps
        self.kill_reward = kill_reward
        self.life_penalty = life_penalty
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self._seed = seed
        #: What game this is. See `game.rules.Rules`; `None` is the default game.
        self.rules = Rules() if rules is None else rules
        self.reset()

    def reset(self, seed=None):
        """Start a fresh episode. Returns (observation, info)."""
        if seed is not None:
            self._seed = seed
        self.game = Game(seed=self._seed, rules=self.rules)
        # One write and one tick per step *is* this clock's pacing, expressed
        # through the same stick every other driver writes to.
        self.controls = Controls(self.game)
        self.steps = 0
        return self.observation(), self.info()

    def step(self, action):
        """Returns (observation, reward, terminated, truncated, info)."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}, expected one of {ACTIONS}")
        game = self.game
        kills_before, lives_before = len(game.monsters), game.lives
        self.controls.write(action)
        self.controls.tick()
        self.steps += 1

        reward = self.kill_reward * (kills_before - len(game.monsters))
        reward += self.life_penalty * (lives_before - game.lives)
        if game.over:
            reward += self.win_reward if game.won else self.loss_penalty
        truncated = not game.over and self.steps >= self.max_steps
        return self.observation(), reward, game.over, truncated, self.info()

    def observation(self):
        """The board as text — the same view `render` prints."""
        return self.game.render()

    def render(self):
        return f"{self.game.render()}\n{self.game.status()}"

    def info(self):
        game = self.game
        return {
            "score": game.score,
            "lives": game.lives,
            "ticks": game.ticks,
            "steps": self.steps,
            "won": game.won,
            "player": game.player,
            "monsters": {i: tuple(pos) for i, pos in game.monsters.items()},
            "missile": tuple(game.missile) if game.missile else None,
            "dangers": [tuple(d) for d in game.dangers],
            "can_shoot": game.missile is None,
            # The live rules, so nothing downstream answers from the module
            # defaults; `Frame.of` picks named fields, so the object never
            # reaches `events.jsonl`.
            "rules": game.rules,
        }


def snapshot(game):
    """The same shape `SpaceInvadersEnv.info()` returns, so players are unchanged."""
    return {
        "steps": game.ticks,
        "score": game.score,
        "lives": game.lives,
        "ticks": game.ticks,
        "won": game.won,
        "player": game.player,
        "monsters": {i: tuple(pos) for i, pos in game.monsters.items()},
        "missile": tuple(game.missile) if game.missile else None,
        "dangers": [tuple(d) for d in game.dangers],
        "can_shoot": game.missile is None,
        # The live rules, so nothing downstream answers from the module
        # defaults; `Frame.of` picks named fields, so the object never reaches
        # `events.jsonl`.
        "rules": game.rules,
    }
