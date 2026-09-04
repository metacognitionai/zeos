# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""How well can an LLM play Space Invaders?

The modules at this level are the entry points; everything they build on sits
in a package of its own: `game/`, `players/`, `clocks/`, `runlog/`, `web/` and
`utils/`.
"""

from .game import ACTIONS, Game, SpaceInvadersEnv

__all__ = ["ACTIONS", "Game", "SpaceInvadersEnv"]
