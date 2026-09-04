# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Minimal ASCII Space Invaders. Pure state and rules, no I/O.

Everything here is measured in ticks, never seconds; whoever drives the game
decides how fast a tick is. A game rule is a `Rules` field, never a module
constant read at tick time -- the constants below are only the defaults
`Rules()` reads once -- so whoever builds the game says what the game was, and
`meta.json` records it.
"""

import random
from dataclasses import dataclass

W, H = 9, 8  # board width, height; player sits on row H-1
ROWS, COLS = 2, 5  # 10 monsters, m1..m10, left to right
COL_OFFSET = 2  # the monster grid starts this far in from the left
LIVES = 3  # hits the player survives before the game ends

# all per tick
MISSILE_ROWS = 1  # rows the player's '^' climbs each tick
DANGER_ROWS = 1  # rows each monster's 'd' falls each tick
#: Default chance that some monster drops a 'd' this tick: high enough that the
#: reflex has something to react to, low enough that a prompt loop plays a
#: whole episode before it dies.
FIRE_CHANCE = 0.45
MARCH_GROUP = 3  # monsters move every 1 + alive // MARCH_GROUP ticks

LEFT, RIGHT, SHOOT = "left", "right", "shoot"


@dataclass(frozen=True, slots=True)
class Rules:
    """What game this is. Frozen, so it can be shared and read anywhere.

    Reached downstream through `snapshot(game)["rules"]`, never the constants
    above: a view or a reflex that read the module while the game was 5x5 would
    answer for a board nobody was playing on.
    """

    w: int = W
    h: int = H
    #: The monster grid: `monster_rows` x `monster_cols`, starting
    #: `monster_col_offset` in from the left.
    monster_rows: int = ROWS
    monster_cols: int = COLS
    monster_col_offset: int = COL_OFFSET
    #: Explicit positions as `((row, col), ...)`, overriding the grid: a monster
    #: put directly over the gun is the shortest route to a preemption.
    monsters: tuple[tuple[int, int], ...] | None = None
    lives: int = LIVES
    missile_rows: int = MISSILE_ROWS
    danger_rows: int = DANGER_ROWS
    march_group: int = MARCH_GROUP
    fire_chance: float = FIRE_CHANCE

    def __post_init__(self) -> None:
        """Refuse a board that cannot hold the game described.

        A contract rather than a precaution: a monster already on the bottom row
        is a plausible typo, and far harder to recognise from the wrong answer it
        produces than from being told.
        """
        if self.w < 1 or self.h < 2:
            raise ValueError(
                f"a board must be at least 1x2 (the player's row plus one above "
                f"it for the missile to spawn on); got {self.w}x{self.h}"
            )
        for name in ("lives", "missile_rows", "danger_rows", "march_group"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1; got {getattr(self, name)}")
        if not 0.0 <= self.fire_chance <= 1.0:
            raise ValueError(f"fire_chance must be in [0, 1]; got {self.fire_chance}")
        for row, col in self.starting_monsters().values():
            if not (0 <= col < self.w and 0 <= row < self.h - 1):
                raise ValueError(
                    f"a monster at ({row}, {col}) is outside a {self.w}x{self.h} "
                    f"board, or on the player's own row {self.h - 1}"
                )

    def starting_monsters(self) -> dict[int, list[int]]:
        """`id -> [row, col]`, ids numbered from 1 in reading order.

        Ids are what label a monster in the rendered board and in every frame of
        the runlog, so they are assigned here, once, and never renumbered when one
        dies -- a reader following `m3` across a scrubber is following one monster.
        """
        if self.monsters is not None:
            return {i + 1: [r, c] for i, (r, c) in enumerate(self.monsters)}
        return {
            r * self.monster_cols + c + 1: [r, c + self.monster_col_offset]
            for r in range(self.monster_rows)
            for c in range(self.monster_cols)
        }

    @property
    def player_row(self) -> int:
        """The row the gun sits on, and the row a `d` has to reach to hurt you."""
        return self.h - 1

    @property
    def monster_count(self) -> int:
        """How many monsters a game starts with."""
        return len(self.starting_monsters())

    def march_interval(self, alive: int) -> int:
        """Ticks between one march and the next, with `alive` monsters left.

        On `Rules` because the prompt states the schedule as a table, and a table
        computed from a stale copy of this line is exactly the kind of wrong the
        model cannot detect.
        """
        return max(1, 1 + alive // self.march_group)

    def turns_to_reach(self, row: int) -> int:
        """Ticks for a missile fired now to arrive at `row`.

        Divided by `missile_rows` rather than assumed to be one row per tick, in
        the one cue the model is told to aim by.
        """
        return -(-(self.h - 2 - row) // self.missile_rows)  # ceiling division

    def turns_to_land(self, row: int) -> int:
        """Ticks until a `d` on this row reaches the player's row.

        A rule, not a scheduling parameter: how much warning counts as "about to
        land" is the driver's (`REFLEX_HORIZON` in the zeos player). Deliberately
        no module-level counterpart, which would answer from the default board.
        """
        return -(-(self.h - 1 - row) // self.danger_rows)  # ceiling division

    def fields(self) -> dict[str, object]:
        """The rules as flat scalars, for `meta.json`.

        Flat so that every rule is a column the same way every other setting is;
        `monsters` is rendered as a string because a table cell is not a list of
        pairs.
        """
        return {
            "width": self.w,
            "height": self.h,
            "monster_rows": self.monster_rows,
            "monster_cols": self.monster_cols,
            "monster_col_offset": self.monster_col_offset,
            "monsters": (
                None
                if self.monsters is None
                else " ".join(f"{r},{c}" for r, c in self.monsters)
            ),
            "lives": self.lives,
            "missile_rows": self.missile_rows,
            "danger_rows": self.danger_rows,
            "march_group": self.march_group,
            "fire_chance": self.fire_chance,
        }


#: The game as it is played by default: one object, so "the standard rules" is a
#: thing to point at rather than a set of constants to re-read.
DEFAULTS = Rules()


class Game:
    """The rules, and the state they act on.

    `rules` is taken at construction, never read off the module at tick time. Not
    a dataclass, deliberately: a generated `__eq__` would make two games built
    from the same seed compare equal while holding different state.
    """

    def __init__(self, seed=None, rules=None):
        self.seed = seed
        self.rules = DEFAULTS if rules is None else rules
        self.rng = random.Random(self.seed)
        self.monsters = self.rules.starting_monsters()
        self.dir = 1  # monster march direction
        self.player = self.rules.w // 2
        self.missile = None  # [row, col] of the single '^' in flight
        self.dangers = []  # [row, col] of each falling 'd'
        self.score = self.ticks = 0
        self.lives = self.rules.lives
        self.over = self.won = False
        #: `standing`: the ship was already in the bomb's column last tick;
        #: `walking_in`: it stepped into the column, the hit a stale decision makes.
        self.hits_standing = self.hits_walking_in = 0
        self._column_last_tick = self.player

    #: Kept as a property for the runs and the tests that read it directly.
    @property
    def fire_chance(self) -> float:
        return self.rules.fire_chance

    def act(self, action):
        """Apply one of LEFT / RIGHT / SHOOT. Does not advance the world."""
        if self.over:
            return
        if action == LEFT:
            self.player = max(0, self.player - 1)
        elif action == RIGHT:
            self.player = min(self.rules.w - 1, self.player + 1)
        elif action == SHOOT and self.missile is None:
            self.missile = [self.rules.h - 2, self.player]

    def tick(self):
        """Advance the world one step."""
        if self.over:
            return
        self.ticks += 1
        self._move_missile()
        self._move_dangers()
        if not self.over and self.ticks % self._interval() == 0:
            self._move_monsters()
        if not self.over and self.rng.random() < self.rules.fire_chance:
            r, c = self.rng.choice(list(self.monsters.values()))
            self.dangers.append([r + 1, c])
        self._column_last_tick = self.player

    def _interval(self):
        """Ticks between monster moves — fewer monsters left, faster they march."""
        return self.rules.march_interval(len(self.monsters))

    def _move_missile(self):
        if self.missile is None:
            return
        self.missile[0] -= self.rules.missile_rows
        hit = next((i for i, m in self.monsters.items() if m == self.missile), None)
        if hit is not None:
            del self.monsters[hit]
            self.score += 10
            self.missile = None
            if not self.monsters:
                self.over = self.won = True
        elif self.missile[0] < 0:
            self.missile = None

    def _move_dangers(self):
        alive = []
        for d in self.dangers:
            d[0] += self.rules.danger_rows
            if d[0] == self.rules.h - 1 and d[1] == self.player:
                # Two bombs landing on one tick must not take the count below 0.
                self.lives = max(0, self.lives - 1)
                if self.player == self._column_last_tick:
                    self.hits_standing += 1
                else:
                    self.hits_walking_in += 1
                # `or`, not `=`: the missile may already have won the game this
                # same tick, and surviving a hit must not undo that.
                self.over = self.over or self.lives <= 0
            elif d[0] < self.rules.h - 1:
                alive.append(d)
        self.dangers = alive

    def _move_monsters(self):
        cols = [c for _, c in self.monsters.values()]
        if min(cols) + self.dir < 0 or max(cols) + self.dir >= self.rules.w:
            self.dir *= -1
            for m in self.monsters.values():
                m[0] += 1
        else:
            for m in self.monsters.values():
                m[1] += self.dir
        if max(r for r, _ in self.monsters.values()) >= self.rules.h - 1:
            self.over = True

    def render(self):
        """The board as text, one fixed-width 4-char cell per square."""
        w, h = self.rules.w, self.rules.h
        grid = [["."] * w for _ in range(h)]
        for r, c in self.dangers:
            grid[r][c] = "d"
        if self.missile:
            grid[self.missile[0]][self.missile[1]] = "^"
        for i, (r, c) in self.monsters.items():
            grid[r][c] = f"m{i}"
        grid[h - 1][self.player] = "p"
        return "\n".join("".join(f"{cell:>4}" for cell in row) for row in grid)

    def status(self):
        return f"score {self.score}   lives {self.lives}   tick {self.ticks}"
