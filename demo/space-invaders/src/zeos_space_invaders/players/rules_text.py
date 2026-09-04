# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The rules, written out for a model, generated from the `Rules` that will run.

Every placeholder in `prompts/rules_common.md` is filled here and nowhere else;
a new rule the model needs to be told becomes a field in `fields()` and a
`{placeholder}` in the file. `str.format` over a flat mapping rather than a
template engine, because a prompt is read by a person as often as by a model.
"""

from ..game.rules import DEFAULTS, Rules


def _plural(n: int) -> str:
    return "" if n == 1 else "s"


def flight(rules: Rules) -> list[tuple[int, int]]:
    """`(row, turns after firing)` for every row the missile is visible on.

    The same arithmetic `Game._move_missile` performs -- spawn at `h - 2`, climb by
    `missile_rows` each tick, gone once it leaves the top -- so a prompt that
    disagreed with the game would have to disagree with that method first. The
    first entry is already one climb up from the spawn row, because the world
    advances before the board is shown.
    """
    out, turn, row = [], 1, rules.h - 2 - rules.missile_rows
    while row >= 0:
        out.append((row, turn))
        row -= rules.missile_rows
        turn += 1
    return out


def missile_table(rules: Rules) -> str:
    """`flight` as a markdown table.

    Generated rather than written because `missile_rows` decides both the row
    sequence and how many columns the table has.
    """
    rows = flight(rules)
    header = "| the missile is at row | " + " | ".join(str(r) for r, _ in rows) + " |"
    rule = "| --------------------- | " + " | ".join("-" for _ in rows) + " |"
    body = "| turns after firing    | " + " | ".join(str(t) for _, t in rows) + " |"
    return "\n".join((header, rule, body))


def march_table(rules: Rules) -> str:
    """The march interval at every possible number of survivors.

    One column per monster, so the table shrinks with the board rather than
    claiming ten monsters exist; `Rules.march_interval` is the only place the
    formula lives.
    """
    counts = list(range(rules.monster_count, 0, -1))
    header = "| monsters alive           | " + " | ".join(str(c) for c in counts) + " |"
    rule = "| ------------------------ | " + " | ".join("-" for _ in counts) + " |"
    body = (
        "| moves once every N turns | "
        + " | ".join(str(rules.march_interval(c)) for c in counts)
        + " |"
    )
    return "\n".join((header, rule, body))


def fields(rules: Rules) -> dict[str, object]:
    """Every `{placeholder}` in `prompts/rules_common.md`, and nothing else.

    On the shortest board `Rules` allows (`h=2`) `flight` is empty and the timing
    section describes a shot that can never connect, which is a true description
    of that board: there is nowhere for a monster to stand.
    """
    seen = flight(rules)
    return {
        "width": rules.w,
        "height": rules.h,
        "rightmost": rules.w - 1,
        "player_row": rules.player_row,
        "monster_count": rules.monster_count,
        "lives": rules.lives,
        "missile_rows": rules.missile_rows,
        "missile_plural": _plural(rules.missile_rows),
        "danger_rows": rules.danger_rows,
        "danger_plural": _plural(rules.danger_rows),
        # A percentage, because that is how a person states odds.
        "fire_percent": f"{rules.fire_chance:.0%}",
        "missile_spawn": rules.h - 2,
        "missile_first": seen[0][0] if seen else rules.h - 2,
        # The wasted-shot cost: one turn per row the missile is visible on, plus
        # the turn it was fired on.
        "missile_blocked": len(seen) + 1,
        # An expression rather than words, because `danger_rows` changes the
        # divisor as well as the subtrahend.
        "danger_formula": (
            f"{rules.player_row} - r"
            if rules.danger_rows == 1
            else f"({rules.player_row} - r) / {rules.danger_rows}, rounded up"
        ),
        # The last row a bomb can be dodged from: one fall short of the gun.
        "danger_last": rules.player_row - rules.danger_rows,
        "missile_table": missile_table(rules),
        "march_table": march_table(rules),
    }


def rules_section(text: str, rules: Rules = DEFAULTS) -> str:
    """Fill `rules_common.md`'s placeholders from `rules`.

    Takes the text rather than reading the file, so `base.py` keeps the one place
    that knows where prompt files live and this module stays a pure function.
    """
    return text.format(**fields(rules))
