# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Where a shot connects, and how soon -- found by playing the rules forward.

Leading a moving target is the one calculation this game asks for that a model
with no reasoning budget cannot do, so `utils.views.LeadView` hands it over. It
is a search rather than a formula because walking takes turns and the block
keeps marching while you walk, and because the winning line often needs a turn
spent waiting, which is not an action: the only no-ops are pressing into a wall
and firing with a missile already up. So the three actions are played forward
breadth-first and the first sequence that kills something is taken. `Game`
itself does the stepping, which keeps the one ordering that matters: the
missile moves and is tested before the block marches, so a monster that drops
onto a missile's square is not hit.
"""

from collections import deque

from .rules import Game


class _NoFire:
    """A generator that never rolls a monster shot.

    New fire is random, so rolling it would make the answer a guess; the bombs
    already in the air fall by the rules, so `_restore` keeps them and a line
    that walks under one is no line at all.
    """

    @staticmethod
    def random():
        return 1.0


class Shot:
    """A shot that lands: fire from `column`, `turns` from now, killing `target`.

    `steps` is how far the player has to walk and is not always `turns`, because
    arriving early and waiting is sometimes the whole trick. `action` is what to
    play this turn to be on that line, and is what `LeadView` prints rather than
    a direction to the column, which contradicts the line whenever it begins
    with a step the other way.
    """

    __slots__ = ("action", "column", "steps", "target", "turns")

    def __init__(self, column, turns, steps, target, action):
        self.column = column
        self.turns = turns
        self.steps = steps
        self.target = target
        self.action = action

    def __repr__(self):
        return (
            f"Shot(column={self.column}, turns={self.turns}, "
            f"steps={self.steps}, target={self.target!r}, "
            f"action={self.action!r})"
        )


def _restore(info, direction):
    """A throwaway game in the state `info` describes, under its rules.

    `info["rules"]` and not the defaults: the board's width is what reverses the
    march, so the wrong width does not clip the answer, it searches a different
    game.
    """
    game = Game(seed=0, rules=info["rules"])
    game.rng = _NoFire()
    game.monsters = {i: list(pos) for i, pos in info["monsters"].items()}
    game.player = info["player"]
    game.ticks = info["ticks"]
    game.dir = direction
    game.missile = list(info["missile"]) if info["missile"] else None
    game.dangers = [list(d) for d in info["dangers"]]
    return game


def _key(game):
    """What makes two futures the same. The tick counter only matters modulo the
    march interval, or the search never revisits anything."""
    interval = game.rules.march_interval(len(game.monsters))
    return (
        game.player,
        game.ticks % interval,
        game.dir,
        None if game.missile is None else tuple(game.missile),
        tuple(sorted((i, r, c) for i, (r, c) in game.monsters.items())),
        tuple(sorted((r, c) for r, c in game.dangers)),
    )


def intercept(info, direction, depth=16):
    """The soonest shot that kills something, or None if none does in `depth`.

    `direction` is which way the block is marching, which `info` does not carry
    -- the caller watches for it. Without it there is nothing to lead.
    """
    if not info.get("monsters") or not direction:
        return None

    start = _restore(info, direction)
    alive = len(start.monsters)
    frontier = deque([(start, ())])
    seen = {_key(start)}

    while frontier:
        game, path = frontier.popleft()
        if len(path) >= depth:
            continue
        for action in ("left", "right", "shoot"):
            future = _restore(_snapshot(game), game.dir)
            future.act(action)
            future.tick()
            if future.lives < game.lives:
                # That step ends under a bomb. Not a line, however soon it kills.
                continue
            taken = (*path, (action, game.player))
            if len(future.monsters) < alive:
                killed = set(game.monsters) - set(future.monsters)
                # The column the shot went up is where the player stood when it
                # fired, which is the last turn `shoot` was played.
                fired = next(
                    (col for act, col in reversed(taken) if act == "shoot"), None
                )
                column = future.player if fired is None else fired
                return Shot(
                    column=column,
                    turns=len(taken),
                    steps=abs(column - info["player"]),
                    target=str(next(iter(killed))),
                    action=taken[0][0],
                )
            marker = _key(future)
            if marker not in seen:
                seen.add(marker)
                frontier.append((future, taken))
    return None


def _snapshot(game):
    """The fields `_restore` needs, straight off a live game."""
    return {
        "monsters": game.monsters,
        "player": game.player,
        "ticks": game.ticks,
        "missile": game.missile,
        "dangers": game.dangers,
        "rules": game.rules,
    }
