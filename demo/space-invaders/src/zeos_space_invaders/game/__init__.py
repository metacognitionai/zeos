# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The game: rules, the stick, a step wrapper, and a snapshot of live state.

Stdlib only, and it knows nothing about models. Everything is measured in ticks,
never seconds -- whoever drives it decides how fast a tick is.

`Rules` is what makes one game different from another, and it is the thing to
reach for rather than the constants: those are only its defaults. Downstream code
gets the live rules from `snapshot(game)["rules"]`.
"""

from .controls import Controls
from .env import ACTIONS, MAX_STEPS, SpaceInvadersEnv, snapshot
from .rules import (
    COL_OFFSET,
    COLS,
    DANGER_ROWS,
    DEFAULTS,
    FIRE_CHANCE,
    LEFT,
    LIVES,
    MARCH_GROUP,
    MISSILE_ROWS,
    RIGHT,
    ROWS,
    SHOOT,
    Game,
    H,
    Rules,
    W,
)

__all__ = [
    "ACTIONS",
    "COLS",
    "COL_OFFSET",
    "DANGER_ROWS",
    "DEFAULTS",
    "FIRE_CHANCE",
    "LEFT",
    "LIVES",
    "MARCH_GROUP",
    "MAX_STEPS",
    "MISSILE_ROWS",
    "RIGHT",
    "ROWS",
    "SHOOT",
    "Controls",
    "Game",
    "H",
    "Rules",
    "SpaceInvadersEnv",
    "W",
    "snapshot",
]
