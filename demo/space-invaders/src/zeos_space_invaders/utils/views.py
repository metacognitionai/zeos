# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Different ways of showing the same game state to a model.

The board is one representation, not the only one, and working out which column
a symbol sits in is the one thing a small model reliably fails at; these views
let that be tested rather than assumed. Each supplies a `state()` that renders
one turn, and `describe(view, rules)` builds its section of the system prompt
from the `Rules` that will run, so switching view changes only the
representation.
"""

from ..game.aim import intercept


def _status(info):
    return (
        f"step {info['steps']} | score {info['score']} | lives {info['lives']} "
        f"| missile in flight: {'no' if info['can_shoot'] else 'yes'}"
    )


class GridView:
    """The ASCII board. The control."""

    name = "grid"
    choose_file = "choose_default.md"

    def state(self, obs, info):
        return f"{_status(info)}\n{obs}"

    def history(self, obs, info):
        """What a past turn looks like. The board alone; the status is implied."""
        return obs


class CoordsView:
    """Positions as integers, no picture at all."""

    name = "coords"
    choose_file = "choose_default.md"

    def state(self, obs, info):
        monsters = "  ".join(
            f"m{i} at row {r} col {c}" for i, (r, c) in sorted(info["monsters"].items())
        )
        missile = (
            f"row {info['missile'][0]} col {info['missile'][1]}"
            if info["missile"]
            else "none in flight"
        )
        dangers = "  ".join(f"row {r} col {c}" for r, c in info["dangers"]) or "none"
        return (
            f"{_status(info)}\n"
            f"you are at col {info['player']} (row {info['rules'].player_row})\n"
            f"monsters: {monsters}\n"
            f"your missile: {missile}\n"
            f"incoming fire: {dangers}"
        )

    def history(self, obs, info):
        """Compact: the point of history is to show what moved, not to repeat it all."""
        columns = ",".join(
            str(c) for c in sorted({c for _, c in info["monsters"].values()})
        )
        fire = (
            "; fire at " + " ".join(f"{r},{c}" for r, c in info["dangers"])
            if info["dangers"]
            else ""
        )
        return f"you col {info['player']}; monster cols {columns}{fire}"


class CuesView(CoordsView):
    """Coordinates plus the facts the picture exists to convey.

    A human reads alignment off the board instantly; this states it outright,
    leaving the model the tactical choice rather than the arithmetic.
    """

    name = "cues"

    def __init__(self):
        self.previous = None  # monster columns last turn, to infer direction
        # Remembered rather than recomputed, because the block only moves every
        # few turns and a fresh reading is "unknown" most of the time.
        self.direction = 0

    def state(self, obs, info):
        base = super().state(obs, info)
        return f"{base}\n{self._cues(info)}"

    def _cues(self, info):
        player, rules = info["player"], info["rules"]
        lines = []

        above = sorted((r, i) for i, (r, c) in info["monsters"].items() if c == player)
        if above:
            row, ident = above[-1]  # the lowest one is the one a shot reaches
            lines.append(
                f"monster in your column: m{ident} at row {row}; "
                f"a shot fired now reaches it in {rules.turns_to_reach(row)} turns"
            )
        else:
            lines.append("monster in your column: none")

        incoming = [r for r, c in info["dangers"] if c == player]
        if incoming:
            lines.append(
                "incoming fire in your column: lands in "
                f"{min(rules.turns_to_land(r) for r in incoming)} turns"
            )
        else:
            lines.append("incoming fire in your column: none")

        columns = sorted({c for _, c in info["monsters"].values()})
        if columns:
            nearest = min(columns, key=lambda c: abs(c - player))
            gap = nearest - player
            where = (
                "your column"
                if gap == 0
                else f"{abs(gap)} step{'s' if abs(gap) > 1 else ''} "
                f"{'right' if gap > 0 else 'left'}"
            )
            lines.append(f"nearest monster column: {nearest} ({where})")

        lines.append(self._march(info))
        return "\n".join(lines)

    def _march(self, info):
        alive = len(info["monsters"])
        interval = info["rules"].march_interval(alive)
        step = info["steps"]
        nxt = ((step // interval) + 1) * interval
        self._track(info)
        named = {1: "right", -1: "left"}.get(self.direction, "not seen yet")
        return (
            f"the block moves every {interval} steps, next on step {nxt}; "
            f"direction {named}"
        )

    def _track(self, info):
        """Update the remembered direction, only on turns the block moved."""
        columns = {i: c for i, (_, c) in info["monsters"].items()}
        if self.previous:
            shared = set(columns) & set(self.previous)
            moved = sum(columns[i] - self.previous[i] for i in shared)
            if moved:
                self.direction = 1 if moved > 0 else -1
        self.previous = columns
        return self.direction


class LeadView(CuesView):
    """Cues plus where the target will be when a shot fired now would arrive.

    Chasing where a monster currently is never wins: with few monsters left the
    block moves a column every tick while a shot takes six to climb. Leading is
    the whole skill and the one calculation a model with no reasoning tokens
    cannot do, so this hands it over and asks only for the decision.
    """

    name = "lead"

    def _cues(self, info):
        base = super()._cues(info)  # this also updates the remembered direction
        return f"{base}\n{self._lead(info, self.direction)}"

    def _lead(self, info, direction):
        """Where a shot connects, asked of the rules rather than guessed at.

        `game/aim.py` plays the rules forward, so the column and the timing are
        the real ones; what is handed over is the calculation, not the game --
        the column, when it connects, and what to do about it this turn.
        """
        if not info["monsters"] or not direction:
            return "aim point: not known yet, the block has not moved"
        shot = intercept(info, direction)
        if shot is None:
            return (
                "aim point: nothing can be hit from anywhere in the next few "
                "turns; stay clear and wait for the block to turn"
            )
        if shot.action == "shoot" and shot.column == info["player"]:
            return (
                f"aim point: m{shot.target} is lined up from col {shot.column} "
                f"— shoot; the hit lands in {shot.turns} "
                f"turn{'s' if shot.turns > 1 else ''}."
            )
        # The first step of the line, not a direction to the column: the two
        # differ when a bomb is falling on the way, and the direction walks the
        # player into it.
        gap = shot.column - info["player"]
        if gap == 0:
            return (
                f"aim point: m{shot.target} can be hit from col {shot.column}, "
                f"{shot.turns} turns from now. You are in col {info['player']} "
                f"already: do not fire yet — play {shot.action}."
            )
        return (
            f"aim point: m{shot.target} can be hit from col {shot.column}, "
            f"{shot.turns} turns from now. You are in col {info['player']}, "
            f"{abs(gap)} step{'s' if abs(gap) > 1 else ''} away: "
            f"play {shot.action} now."
        )


class DrillView(CuesView):
    """The cues, plus a rules section that is an order rather than advice.

    The ablation for `lead`: same cues, imperative rules, no aim point. Tests
    whether the collapse to always-shoot is a reading or a compliance problem.
    """

    name = "drill"
    choose_file = "choose_strict.md"


GRID_DESCRIPTION = """\
## The board

The board is {width} columns wide and {height} rows tall, printed as
fixed-width 4-character cells. Rows are numbered 0 at the top
to {player_row} at the bottom, columns 0 at the left to {rightmost} at the
right.

```
{board}
```

`m1`..`m{monster_count}` are monsters, `p` is you, `^` is your missile,
`d` is monster fire falling toward you, `.` is empty. One square holds one
glyph, so a monster hides a `d` in the same square.
"""

COORDS_DESCRIPTION = """\
## What you are told

There is no picture. Every position is given to you as numbers.

The board is {width} columns (0 at the left to {rightmost} at the right)
and {height} rows (0 at the top to {player_row} at the bottom). You are always
on row {player_row}. Each turn you are told your column, where every surviving
monster is, where your missile is if one is in flight, and where any incoming
fire is:

```
{example}
```

A monster threatens you when its column equals yours. Incoming fire hits you
when it reaches row {player_row} and its column equals yours.
"""

CUES_DESCRIPTION = (
    COORDS_DESCRIPTION
    + """
After the positions you are given four facts worked out for you, so you never
have to count anything:

```
{cues}
```

`monster in your column: none` means a shot now would hit nothing. `incoming
fire in your column: none` means you are safe to stand still this turn.
"""
)

LEAD_DESCRIPTION = (
    CUES_DESCRIPTION
    + """
Finally you are given the aim point — the column a shot has to be fired from to
actually hit something, and how long from now that hit lands:

```
{aim}
```

This matters more than anything else on the board. The block keeps moving while
your missile is in the air, so firing at where a monster is now misses. The aim
point already accounts for the time you spend walking there and for every bomb
already falling, so the move it names is the move to play this turn: it is
sometimes a step away from the column, or a shot to stand still, and the line
says so when it is.
"""
)

LeadView.choose_file = "choose_lead.md"

#: Which template each view's section comes from. A mapping rather than a class
#: attribute because `DrillView` deliberately shares `CuesView`'s text -- it is
#: the ablation, differing only in `choose_file`.
TEMPLATES = {
    "grid": GRID_DESCRIPTION,
    "coords": COORDS_DESCRIPTION,
    "cues": CUES_DESCRIPTION,
    "drill": CUES_DESCRIPTION,
    "lead": LEAD_DESCRIPTION,
}


def _played(rules, turns=9):
    """Every turn of a short scripted game, as `(board, info)` pairs.

    Playing the real game makes the sample the format by construction, and
    deterministically. Every turn rather than the last, because the cue views
    infer the block's direction by comparing consecutive turns; handed one
    snapshot they report "direction not seen yet".
    """
    from ..game.env import snapshot
    from ..game.rules import Game

    game, out = Game(seed=0, rules=rules), []
    for turn in range(turns):
        game.act("shoot" if turn == turns - 3 else "right")
        game.tick()
        out.append((game.render(), snapshot(game)))
        if game.over:  # a board small enough to finish inside the sample
            break
    return out


def describe(view, rules):
    """`view`'s section of the system prompt, with everything taken from `rules`.

    A function rather than a method because it needs a game played, which a view
    has no business constructing; `_cues` is reached directly rather than through
    `state` so the cues block does not repeat the coordinates shown above it.
    """
    from ..game.rules import Game
    from ..players.rules_text import fields

    turns = _played(rules)
    board, info = turns[-1]
    # Walked through every turn so the views learn which way the block is going.
    cues, lead = CuesView(), LeadView()
    for _, earlier in turns:
        cues._cues(earlier)
        lead._cues(earlier)
    return TEMPLATES[view.name].format(
        board=Game(seed=0, rules=rules).render(),
        example=CoordsView().state(board, info),
        cues=cues._cues(info),
        aim=lead._lead(info, lead.direction),
        **fields(rules),
    )


VIEWS = {v.name: v for v in (GridView, CoordsView, CuesView, DrillView, LeadView)}
