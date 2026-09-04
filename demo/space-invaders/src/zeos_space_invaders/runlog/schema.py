# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""One record shape for every run, whoever is at the controls.

`events.jsonl` is a single stream of `frame` and `decision` records in the order
they happened. Frames are emitted by the clock, one per world tick, and carry
structured state rather than the drawn board; the rendered prompt stays on the
decision, because that is what the model saw. The kernel's own journal is per
token boundary and goes to `kernel.jsonl` beside it, keyed by tick.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar

#: Bumped whenever a reader written against this layout would misread a run.
#: `meta.json` carries it, so a reader can refuse rather than guess.
SCHEMA_VERSION = 2

#: Who chose an action: one per player, plus the two zeos jobs. A tick nobody
#: chose on has a frame and no decision.
AUTHORS = ("human", "random", "model", "pilot", "evade")


def _line(kind, values):
    """A JSON-ready dict: `kind` first, and nothing that was never filled in."""
    return {"kind": kind, **{k: v for k, v in values.items() if v is not None}}


@dataclass
class Frame:
    """The world after `tick` ticks. Everything needed to redraw the board."""

    kind: ClassVar[str] = "frame"

    tick: int
    score: int
    lives: int
    player: int
    monsters: dict  # str(id) -> [row, col], ids kept so labels are stable
    missile: list | None
    dangers: list
    can_shoot: bool
    over: bool
    won: bool

    @classmethod
    def of(cls, info, over=False):
        """From an `env.info()` or `snapshot()` dict -- the two have one shape."""
        return cls(
            tick=info["ticks"],
            score=info["score"],
            lives=info["lives"],
            player=info["player"],
            monsters={str(i): list(pos) for i, pos in info["monsters"].items()},
            missile=list(info["missile"]) if info["missile"] else None,
            dangers=[list(d) for d in info["dangers"]],
            can_shoot=info["can_shoot"],
            over=over,
            won=info["won"],
        )

    def record(self):
        # Unfiltered, unlike a decision: every field of a frame is always
        # measured, and `missile: null` has to survive the round trip.
        return {"kind": self.kind, **asdict(self)}


@dataclass
class Decision:
    """One choice of action, by whoever made it.

    Who chose and what is intrinsic; *when* is the clock's to fill in -- `tick`
    is the frame the chooser looked at, `tick_applied` where the action landed,
    and they differ exactly when the player was late or throttled.
    """

    kind: ClassVar[str] = "decision"

    by: str
    action: str
    tick: int | None = None
    parsed: bool = True
    applied: bool = True
    tick_applied: int | None = None
    latency: float | None = None
    #: Time to the first piece of a streamed reply: what moves when a server
    #: serves the prompt prefix from cache, which not every server reports.
    ttft: float | None = None
    #: The request that produced this decision, verbatim.
    prompt: str | None = None
    reply: str | None = None
    reasoning: str | None = None
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    retried: bool | None = None  # openai: reasoning ate the budget
    preempted: bool | None = None  # zeos: a pass was abandoned for the reflex
    kernel_ticks: int | None = None  # zeos: token boundaries spent this tick
    reward: float | None = None  # step clock only; the env shapes it
    seq: int | None = None  # filled by the writer

    def record(self):
        return _line(self.kind, asdict(self))
