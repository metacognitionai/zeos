# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The board flags, and the `Rules` they build.

Here rather than in `cli.py` because `play` takes the same board and must not
import the kernel to get it: a keyboard game needs the rules and nothing else.
"""

from ..game import DEFAULTS, Rules


def add_rules_flags(parser, description):
    """The board flags, on whichever parser is being built.

    `play` and `agent` share them so a person and a model can be handed the same
    game, from the same `--settings` file.
    """
    rules = parser.add_argument_group("the game", description)
    for flag, field in (
        ("--width", "w"),
        ("--height", "h"),
        ("--monster-rows", "monster_rows"),
        ("--monster-cols", "monster_cols"),
        ("--monster-col-offset", "monster_col_offset"),
        ("--lives", "lives"),
        ("--missile-rows", "missile_rows"),
        ("--danger-rows", "danger_rows"),
        ("--march-group", "march_group"),
    ):
        rules.add_argument(
            flag,
            dest=field,
            type=int,
            default=None,
            help=f"{field} (default {getattr(DEFAULTS, field)})",
        )


def rules_of(args) -> Rules:
    """One `Rules` from the flags, built once and passed everywhere.

    `None` means "leave the default", so `Rules()` and an untouched command line
    are the same run; construction is where a nonsensical board is refused, so
    a typo is reported before an episode starts.
    """
    asked = {
        field: getattr(args, field)
        for field in (
            "w",
            "h",
            "monster_rows",
            "monster_cols",
            "monster_col_offset",
            "lives",
            "missile_rows",
            "danger_rows",
            "march_group",
        )
        if getattr(args, field, None) is not None
    }
    if args.fire_chance is not None:
        asked["fire_chance"] = args.fire_chance
    return Rules(**asked)
