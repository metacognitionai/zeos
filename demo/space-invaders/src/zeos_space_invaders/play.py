# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Play the game in a terminal. The world runs on a wall clock.

Keys are applied the instant they are read and there is no cap on actions per
tick, so holding left crosses several columns while the world advances once.
That is the one place a human differs from `SpaceInvadersEnv`, where one step is
exactly one action plus one tick however long the policy takes.

A human run is logged the same way a model's is -- same directory, same records,
`by: human` on the decisions -- so the two can be put side by side, and a person
playing the same seed is a baseline rather than an anecdote.
"""

import argparse
import curses
import sys
import time
from pathlib import Path

from .clocks.outcome import announce, outcome_of
from .clocks.render import played, status
from .game import FIRE_CHANCE, LEFT, RIGHT, SHOOT, Controls, Game, snapshot
from .runlog import Decision, RunWriter
from .utils import add_rules_flags, rules_of, settings_flags

TICK_HZ = 5.0
TICK_SECONDS = 1.0 / TICK_HZ  # the deadline for a keystroke
POLL_SECONDS = 0.01

KEYS = {
    curses.KEY_LEFT: LEFT,
    ord("a"): LEFT,
    curses.KEY_RIGHT: RIGHT,
    ord("d"): RIGHT,
    ord(" "): SHOOT,
    curses.KEY_UP: SHOOT,
    ord("w"): SHOOT,
}
HELP = "left/right or a/d to move, space to shoot, q to quit"


def _draw(scr, game, action, footer):
    scr.erase()
    scr.addstr(0, 0, game.render())
    # The board's own height, so a resized game does not draw its status line
    # over its bottom row.
    height = game.rules.h
    # The same three lines a model's run prints, from the same helpers.
    scr.addstr(height + 1, 0, status(game))
    scr.addstr(height + 2, 0, played(game, action))
    scr.addstr(height + 3, 0, footer)
    scr.refresh()


def _loop(scr, seed=None, run=None, rules=None):
    curses.curs_set(0)
    scr.nodelay(True)
    scr.keypad(True)
    game = Game(seed=seed, rules=rules)
    # The same stick an agent writes to; a human gets no per-tick budget, which
    # is the one place a person legitimately differs.
    controls = Controls(game)
    started = time.monotonic()
    records = []
    #: What was played since the last tick, cleared by it; several keys can land
    #: on one tick, and the last of them is the one the display names.
    action = None
    last = time.monotonic()
    if run:
        run.frame(snapshot(game))
    while not game.over:
        key = scr.getch()
        if key in (ord("q"), 27):
            break
        if key in KEYS:
            controls.write(KEYS[key])
            # Unlimited actions per tick is the human's advantage, and the
            # record keeps it visible: several decisions can name one tick.
            decision = Decision(
                by="human",
                action=KEYS[key],
                tick=game.ticks,
                tick_applied=game.ticks,
            )
            records.append(decision)
            action = KEYS[key]
            if run:
                run.decision(decision)
        now = time.monotonic()
        if now - last >= TICK_SECONDS:
            controls.tick()
            action = None
            last = now
            if run:
                run.frame(snapshot(game), over=game.over)
        _draw(scr, game, action, HELP)
        time.sleep(POLL_SECONDS)
    # The same shape every clock writes, so a human's run reads back through
    # `RunReader` exactly as a model's does.
    summary = {
        **outcome_of(game, records, time.monotonic() - started, over="quit"),
        # A human cannot misread their own keyboard, and nothing is billed.
        "unparseable": 0,
        "usage": {},
    }
    if run:
        run.finish(summary)
    if game.over:
        _draw(
            scr,
            game,
            action,
            ("YOU WIN" if game.won else "GAME OVER") + " - press any key",
        )
        scr.nodelay(False)
        scr.getch()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, help="fixes the episode, so an agent can be given the same"
    )
    parser.add_argument(
        "--out", type=Path, help="run directory (default: runs/<stamp>-human)"
    )
    parser.add_argument(
        "--no-record", action="store_true", help="play without writing a run directory"
    )
    parser.add_argument(
        "--fire-chance",
        type=float,
        default=None,
        help=f"chance a monster drops a bomb on any tick (default {FIRE_CHANCE})",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="the `board` object of a JSON file of flags, so a person can be "
        "given the board a model was given; a flag here wins over the file",
    )
    add_rules_flags(parser, "The board, the same flags `agent` takes.")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(settings_flags(argv, ("board",)) + argv)
    # One object, built once, handed to the loop and recorded in the metadata,
    # so the game cannot drift between what ran and what was written down.
    rules = rules_of(args)

    run = None
    if not args.no_record:
        config = {
            "player": "human",
            "clock": "human",
            "seed": args.seed,
            "tick_seconds": TICK_SECONDS,
            # The whole game, flat, the same shape `agent` records.
            **rules.fields(),
        }
        run = (
            RunWriter(args.out, config)
            if args.out
            else RunWriter.create(f"human-seed{args.seed}", config)
        )
    try:
        summary = curses.wrapper(_loop, seed=args.seed, run=run, rules=rules)
    finally:
        if run:
            run.close()
    # The same lines `agent` prints: a game you played is a run like any other.
    announce(summary, clock="human", path=run.path if run else None)


if __name__ == "__main__":
    main()
