# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The append-only journal.

Kernel state is a fold over this sequence. The kernel core never touches a file --
it emits events and the driver appends them here -- so the core stays pure while the
journal stays authoritative.

Sequence numbers are assigned on append rather than carried by events: an event is
a fact, its sequence number is a position, and only the journal knows positions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from zeos.core.events import Event
from zeos.journal.codec import from_line, to_line

__all__ = ["JournalRecord", "Journal", "read_journal", "read_journal_lines"]


@dataclass(frozen=True, slots=True)
class JournalRecord:
    seq: int
    event: Event


class Journal:
    """Collects events in order, optionally streaming them to a file as they land.

    Streaming matters for the thrash and starvation cases: a run that is killed
    mid-flight should still leave an analysable journal behind.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._records: list[JournalRecord] = []
        self._path = path
        self._fh: IO[str] | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # The newline argument is part of the determinism gate, not a style
            # choice. The gate compares `to_bytes()` against the file's bytes, and
            # `to_bytes()` joins lines with a bare newline; default text mode
            # translates that to CRLF on Windows, so every line would differ by one
            # byte and a byte-identical replay would be impossible there. It also
            # keeps a journal portable between platforms, which a byte comparison
            # requires.
            self._fh = path.open("w", encoding="utf-8", newline="\n")

    def append(self, event: Event) -> int:
        seq = len(self._records)
        self._records.append(JournalRecord(seq=seq, event=event))
        if self._fh is not None:
            self._fh.write(to_line(seq, event))
            self._fh.write("\n")
        return seq

    def extend(self, events: Iterable[Event]) -> None:
        for event in events:
            self.append(event)

    @property
    def records(self) -> Sequence[JournalRecord]:
        return self._records

    def events(self) -> Iterator[Event]:
        for record in self._records:
            yield record.event

    def of_kind[E: Event](self, cls: type[E]) -> list[E]:
        """All events of one type, in order.

        The workhorse of integration assertions: tests state journal properties
        ("an alarm preempted supervision within one boundary") rather than
        inspecting transcripts.
        """
        return [r.event for r in self._records if isinstance(r.event, cls)]

    def lines(self) -> list[str]:
        return [to_line(r.seq, r.event) for r in self._records]

    def to_bytes(self) -> bytes:
        """The canonical serialisation compared by the determinism gate."""
        return ("".join(line + "\n" for line in self.lines())).encode("utf-8")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._records)


def read_journal_lines(lines: Iterable[str]) -> list[JournalRecord]:
    records: list[JournalRecord] = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            seq, event = from_line(stripped)
        except Exception as exc:
            raise ValueError(f"journal line {lineno}: {exc}") from exc
        records.append(JournalRecord(seq=seq, event=event))
    return records


def read_journal(path: Path) -> list[JournalRecord]:
    with path.open("r", encoding="utf-8") as fh:
        return read_journal_lines(fh)
