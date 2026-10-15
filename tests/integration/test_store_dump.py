# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The span store written beside the journal, and the debugger's view of it.

The load-bearing test is the equivalence: the store rebuilt from the journal alone
must equal the store the kernel holds. The journal is the record; the dump is the
kernel's own object; if they ever differ, one of them is lying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_stage_d_virtual_context import (
    LONG_JOB,
    build,
    chatter,
    first_evicted,
    of,
)
from zeos.core.events import Event, SegmentEvicted
from zeos.core.ids import DescriptorName
from zeos.core.kernel import Kernel
from zeos.debugger.payload import frames, replay_context
from zeos.journal.writer import Journal
from zeos.store_dump import read_store, write_store

STATIC = Path(__file__).resolve().parents[2] / "src" / "zeos" / "debugger" / "static"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Kernel, list[Event], Path]:
    """A job that evicts, faults one span back in, and evicts on: one entry is reused."""
    target = first_evicted(12)
    script = chatter(12) + [{"fault": target}] + chatter(12) + [{"exit": True}]
    kernel, events = build([LONG_JOB], {"patrol": script})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()
    path = tmp_path_factory.mktemp("store") / "patrol.store.jsonl"
    write_store(kernel.store, path)
    return kernel, events, path


def rebuilt_from_journal(events: list[Event]) -> dict[str, tuple[list[str], int]]:
    """Store id -> (tokens, refs), from the context log alone: the text a splice
    removed, keyed by the store id the stub row that follows it names."""
    journal = Journal()
    journal.extend(events)
    log = frames(journal.records)["contexts"]
    removed: dict[tuple[int, int], list[str]] = {}
    store: dict[str, tuple[list[str], int]] = {}
    for index, row in enumerate(log):
        if row[1] == "splice":
            _, _, job, start, stop, *_ = row
            window = replay_context(log[:index])[job]
            region = next(r for r in window["regions"] if r["segment"] == start)
            removed[job, stop] = list(region["text"])
        elif row[1] == "stub":
            _, _, job, stub, _evicted, store_id, _size = row
            text, refs = store.get(store_id, (removed[job, stub], 0))
            store[store_id] = (text, refs + 1)
    return store


def test_the_dump_is_the_store(run: tuple[Kernel, list[Event], Path]) -> None:
    kernel, _, path = run
    rows = read_store(path)
    spans = kernel.store.spans()
    assert [r["store_id"] for r in rows] == [s.store_id for s in spans]
    for row, span in zip(rows, spans, strict=True):
        assert [t["text"] for t in row["tokens"]] == [t.text for t in span.tokens]
        assert row["refs"] == kernel.store.refs(span.store_id)
        assert row["origin"] == int(span.origin)
    assert sum(r["refs"] for r in rows) == kernel.store.total_puts


def test_a_span_evicted_twice_has_one_entry_with_two_refs(
    run: tuple[Kernel, list[Event], Path],
) -> None:
    kernel, events, path = run
    rows = read_store(path)
    assert len(rows) < len(of(events, SegmentEvicted)), "nothing was evicted twice"
    assert any(r["refs"] == 2 for r in rows)
    assert len(rows) == kernel.store.unique_spans


def test_the_journal_rebuilds_the_same_store(run: tuple[Kernel, list[Event], Path]) -> None:
    _, events, path = run
    rebuilt = rebuilt_from_journal(events)
    dumped = {
        r["store_id"]: ([t["text"] for t in r["tokens"]], r["refs"]) for r in read_store(path)
    }
    assert rebuilt == dumped


def test_the_payload_keys_each_entry_to_the_frame_that_first_archived_it(
    run: tuple[Kernel, list[Event], Path],
) -> None:
    _, events, path = run
    journal = Journal()
    journal.extend(events)
    built = frames(journal.records, store=read_store(path))
    assert frames(journal.records)["store"] is None
    first_stub: dict[str, int] = {}
    for frame, op, *rest in built["contexts"]:
        if op == "stub":
            first_stub.setdefault(rest[3], frame)
    rows: list[list[Any]] = built["store"]
    assert rows
    for frame, store_id, tokens, refs, *_ in rows:
        assert frame == first_stub[store_id]
        assert tokens and refs >= 1


def test_the_page_has_a_store_pane() -> None:
    assert 'id="store"' in (STATIC / "index.html").read_text()
