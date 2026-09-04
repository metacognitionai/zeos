# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""One board, drawn the same way whoever is playing.

`board()` is the plain-terminal form. `play.py` keeps its own curses call
because it has to address a window rather than a stream, but takes the same
two lines from `status()`, so the two cannot drift.
"""

CLEAR = "\033[H\033[J"


def status(game):
    """The line under the board: score, lives, monsters left, missile state."""
    loaded = "ready" if game.missile is None else "in flight"
    return (
        f"score {game.score:>3}   lives {game.lives}   "
        f"{len(game.monsters)} left   missile {loaded}"
    )


def played(game, action=None, by=None, flag=""):
    """The line under the status: this tick, and what was played *on this tick*.

    Edge-triggered, to match the stick: `None` is a tick nobody acted on and
    draws as a dash. `by` names the actor where more than one can act; `flag`
    is for what the kernel did about it.
    """
    if action is None:
        return f"tick {game.ticks:>4}   —"
    who = f" by {by:<6}" if by else ""
    return f"tick {game.ticks:>4}   {action:<5}{who}{flag}"


def board(game, footer=""):
    """The board, the status line and the caller's footer, printed over the last."""
    return f"{CLEAR}{game.render()}\n\n{status(game)}\n{footer}"
