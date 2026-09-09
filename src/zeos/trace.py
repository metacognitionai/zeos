# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A machine's own account of each window, sampled beside the journal.

The journal counts in kernel words. What the model is fed -- sub-word pieces, chat
framing, how much of it the cache still holds -- is the machine's business. A machine
implementing ``TracesRaw`` is asked after every tick, and what changed is written to a
side file keyed by journal sequence number: a separate machine's account, kept apart
from the byte-identical record the determinism gate compares. ``replay_trace`` is the
specification of how the rows become windows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from zeos.core.events import BlockBoundary, Decoded, Event, Forked, Injected, Spliced
from zeos.core.ids import JobId
from zeos.machine.base import RawWindow, TracesRaw

__all__ = ["RawTrace", "read_trace", "replay_trace"]


def _touched(events: Sequence[Event], first_seq: int) -> dict[int, int]:
    """Which jobs' windows the machine changed, and the last event that did so."""
    touched: dict[int, int] = {}
    for offset, event in enumerate(events):
        match event:
            case Decoded() | Injected() | Spliced():
                touched[int(event.job)] = first_seq + offset
            case BlockBoundary() if event.padding_tokens:
                touched[int(event.job)] = first_seq + offset
            case Forked():
                touched[int(event.child)] = first_seq + offset
            case _:
                continue
    return touched


def _word(word: Any) -> dict[str, Any]:
    return {"pieces": list(word.pieces), "framing": list(word.framing)}


class RawTrace:
    """Collects a machine's account after each tick, streaming rows to a file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self._last: dict[int, RawWindow] = {}
        self._file = path.open("w", encoding="utf-8", newline="\n") if path else None

    def sample(self, machine: TracesRaw, events: Sequence[Event], first_seq: int) -> None:
        """Write one row per window the events changed: the words from the first that
        differs from the last sample, the trailing framing and the cache mark, keyed
        by the sequence number of the last event that changed it."""
        for job, seq in sorted(_touched(events, first_seq).items(), key=lambda item: item[1]):
            window = machine.raw(JobId(job))
            before = self._last.get(job)
            from_word = 0
            if before is not None:
                limit = min(len(before.words), len(window.words))
                while from_word < limit and before.words[from_word] == window.words[from_word]:
                    from_word += 1
            row = {
                "seq": seq,
                "job": job,
                "kv_resident": window.kv_resident,
                "from_word": from_word,
                "words": [_word(w) for w in window.words[from_word:]],
                "trailing": list(window.trailing),
            }
            self.rows.append(row)
            if self._file is not None:
                self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
                self._file.flush()
            self._last[job] = window

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self) -> int:
        return len(self.rows)


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replay_trace(
    rows: Sequence[Mapping[str, Any]], *, upto: int | None = None
) -> dict[int, dict[str, Any]]:
    """Rebuild every job's account through sequence number ``upto``: each row cuts the
    words at ``from_word``, appends its own, and replaces the trailing framing and the
    cache mark."""
    windows: dict[int, dict[str, Any]] = {}
    for row in rows:
        if upto is not None and row["seq"] > upto:
            break
        held = windows.setdefault(
            row["job"], {"words": [], "trailing": [], "kv_resident": 0, "model_tokens": 0}
        )
        held["words"] = held["words"][: row["from_word"]] + [dict(w) for w in row["words"]]
        held["trailing"] = list(row["trailing"])
        held["kv_resident"] = row["kv_resident"]
        held["model_tokens"] = sum(
            len(w["pieces"]) + len(w["framing"]) for w in held["words"]
        ) + len(held["trailing"])
    return windows
