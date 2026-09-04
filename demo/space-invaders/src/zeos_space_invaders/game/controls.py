# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The stick everyone writes to, and the rule that turns a write into a move.

Edge-triggered, deliberately: one write, one move; no write, no move. Every
driver goes through it, because re-applying the last write every tick emits
moves nobody decided on, and a player that decided nothing should score as one.
"""


class Controls:
    """One game's stick. Not thread-safe; the caller holds whatever lock it has.

    controls = Controls(game, per_tick=1)
    controls.write("left")      # -> True if the ship moved
    controls.tick()             # world advances, budget reopens
    """

    def __init__(self, game, per_tick=None):
        self.game = game
        #: None means "as many as you can write", which is what a human at a
        #: keyboard gets: holding left crosses several columns in one tick.
        self.per_tick = per_tick
        self.used = 0  # writes applied since the last tick
        self.applied = 0  # writes applied all episode

    @property
    def full(self):
        """Whether this tick's budget is spent.

        Separate from `write` because a keyboard drops the keystroke while an
        agent waits for the next tick rather than throw away a reply.
        """
        return self.per_tick is not None and self.used >= self.per_tick

    def write(self, action):
        """Move the ship. Returns whether the move happened.

        False means the budget is spent or the episode is over, and a decision
        that arrived too late is not the same as one that was played.
        """
        if self.full or self.game.over:
            return False
        self.game.act(action)
        self.used += 1
        self.applied += 1
        return True

    def tick(self):
        """Advance the world one step and open a fresh budget."""
        self.game.tick()
        self.used = 0
