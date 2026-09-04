# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A command source that reads each command off a tape written in the case.

Plugged into the same ``CommandSeat`` as the model sources, so a case plays out over the
same token boundaries whether a tape or a model is answering. What it buys is a run that
needs no weights, no key and no network, and that comes out byte-identical every time --
which is what makes a scheduled interrupt land in the same place twice.

A tape is the ``script:`` list in a descriptor's frontmatter, one command per step::

    script:
      - emit: "say 1;"
      - emit: "write tools 10;"
      - emit: "read stdin;"

It is the whole behaviour: nothing here reads the descriptor body or the status region,
so a tape is only ever right for the event schedule it was written against.
"""

from __future__ import annotations

from collections.abc import Mapping

from zeos.machine.scripted import Script, ScriptExhausted
from zeos_coop_count.seat import Turn

__all__ = ["TapeSource"]


class TapeSource:
    """A command source whose next command is the next step of the job's tape."""

    def __init__(self, scripts: Mapping[str, Script]) -> None:
        self._scripts = dict(scripts)

    def next_command(self, turn: Turn) -> str:
        steps = self._scripts.get(turn.descriptor, Script()).steps
        if turn.issued >= len(steps):
            # An authoring mistake rather than an implicit exit, exactly as the kernel's
            # own scripted backend treats it: a tape ends where its run ends, so running
            # off it means the schedule moved and the tape did not.
            raise ScriptExhausted(
                f"job {turn.job} ({turn.descriptor}) asked for command {turn.issued + 1} "
                f"of a tape with {len(steps)}"
            )
        return steps[turn.issued].emit
