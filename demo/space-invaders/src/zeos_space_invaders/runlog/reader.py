# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Read a run directory back; what compare, an analysis script and the viewer use.

A compare directory is a run of runs: `episodes()` yields the same reader for
each episode under it, so nothing downstream needs to know whether a run came
from `agent` or from `compare`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import SCHEMA_VERSION


class RunReader:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = _json(self.path / "meta.json") or {}
        found = self.meta.get("schema")
        if found is not None and found != SCHEMA_VERSION:
            raise ValueError(
                f"{self.path} was written by schema {found}, this is "
                f"{SCHEMA_VERSION}: read it with a matching checkout"
            )

    @property
    def summary(self):
        """The verdict, or None for a run that did not finish.

        Read once per reader: the index asks for it several times a row and
        reloads every few seconds, and a reader is built fresh per request, so
        the cache cannot go stale.
        """
        if not hasattr(self, "_summary"):
            self._summary = _json(self.path / "summary.json")
        return self._summary

    def row(self):
        """Settings and verdict as one flat dict -- one line of a table.

        Joined here rather than read from a second copy someone remembered to
        keep in step; the prompt is left out because it is kilobytes, and
        `meta["prompt"]` is right there.
        """
        return {
            **{k: v for k, v in self.meta.items() if k != "prompt"},
            **(self.summary or {}),
        }

    def events(self, kind=None):
        for line in _lines(self.path / "events.jsonl"):
            if kind is None or line["kind"] == kind:
                yield line

    def frames(self):
        return self.events("frame")

    def decisions(self):
        return self.events("decision")

    def kernel(self):
        return _lines(self.path / "kernel.jsonl")

    def frame_at(self, tick):
        """The world the chooser saw. Built once, on first use."""
        if not hasattr(self, "_by_tick"):
            self._by_tick = {f["tick"]: f for f in self.frames()}
        return self._by_tick.get(tick)

    def episodes(self):
        """Every episode under a compare run, in directory order.

        Empty for a single run rather than an error, so one loader can be handed
        either kind of directory and ask.
        """
        root = self.path / "episodes"
        if not root.is_dir():
            return []
        return [RunReader(d) for d in sorted(root.iterdir()) if d.is_dir()]


def aimed(episodes):
    """Share of shots fired with a monster in the player's column.

    Pooled over the episodes rather than averaged per episode, so a short game
    does not weigh as much as a long one. It reads the frame of the tick the
    decision names, which is exactly what the chooser was looking at.
    """
    hit = total = 0
    for episode in episodes:
        for decision in episode.decisions():
            if decision["action"] != "shoot":
                continue
            total += 1
            frame = episode.frame_at(decision["tick"])
            columns = [col for _, col in frame["monsters"].values()] if frame else []
            hit += frame["player"] in columns
    return (100 * hit / total) if total else 0.0


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _lines(path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
