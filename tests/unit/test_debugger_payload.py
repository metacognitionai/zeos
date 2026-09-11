# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The debugger's projection: does it say what the case and the journal say?

The delta round-trip is the load-bearing test here. The page reconstructs every
frame by shallow-merging deltas, and the proof that the merge loses nothing is this
test rather than anything in the browser -- which is the same division the repository
keeps everywhere else: the arithmetic lives where pytest can reach it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from zeos.core import serde
from zeos.core.clock import Clock
from zeos.core.events import (
    Decoded,
    Event,
    Forked,
    Injected,
    PermsChanged,
    PipeReadEvent,
    PipeWritten,
    SegmentClosed,
    SegmentEvicted,
    Spliced,
)
from zeos.core.ids import (
    DescriptorName,
    Integrity,
    JobId,
    Perm,
    PipeName,
    Principal,
    Ring,
    SegmentId,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.vectors import VectorTable
from zeos.debugger.payload import (
    CONTEXT_OPS,
    apply_delta,
    build_payload,
    frames,
    replay_context,
    structure,
)
from zeos.descriptor.lint import lint
from zeos.descriptor.loader import CaseBundle, load_case
from zeos.descriptor.schema import Descriptor
from zeos.driver import Driver, build_kernel, load_schedule
from zeos.journal.writer import Journal, JournalRecord
from zeos.machine.scripted import PAD_TOKEN, Script, ScriptedMachine
from zeos.monitor.state import MARKED_KINDS, fold
from zeos.trace import RawTrace, replay_trace
from zeos.world.store import WorldStore

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


@pytest.fixture(scope="module")
def bundle() -> CaseBundle:
    return load_case(SMOKE)


@pytest.fixture(scope="module")
def kernel(bundle: CaseBundle) -> Kernel:
    kernel, _transport = build_kernel(bundle, config=KernelConfig(case=bundle.name))
    driver = Driver(kernel, reap=False)
    driver.boot(bundle.boot)
    driver.run(load_schedule(SMOKE / "events.jsonl"))
    return kernel


@pytest.fixture(scope="module")
def records(kernel: Kernel) -> Sequence[JournalRecord]:
    journal = Journal()
    journal.extend(kernel.events)
    return journal.records


# --- structure --------------------------------------------------------------


def test_structure_carries_the_declared_wiring(bundle: CaseBundle) -> None:
    view = structure(bundle)

    names = [d["name"] for d in view["descriptors"]]
    assert names == ["supervision", "threshold-alarm"]
    assert [p["name"] for p in view["pipes"]] == [
        "actuators.unit_a",
        "ops.report",
        "sensors.threshold",
    ]
    (vector,) = view["vectors"]
    assert (vector["source"], vector["handler"], vector["priority"]) == (
        "sensors.threshold",
        "threshold-alarm",
        5,
    )


def test_edges_follow_the_declared_bindings(bundle: CaseBundle) -> None:
    """Every edge the diagram draws is a binding somebody wrote down.

    The direction is the alias convention, which is the approximation
    ``payload.ALIAS_KIND`` documents -- so it is asserted here rather than assumed,
    and a schema that one day declares direction will fail this test loudly.
    """
    edges = {(e.get("descriptor"), e["pipe"], e["kind"]) for e in structure(bundle)["edges"]}

    assert ("supervision", "ops.report", "write") in edges  # stdout
    assert ("threshold-alarm", "actuators.unit_a", "duplex") in edges  # tools
    assert ("threshold-alarm", "sensors.threshold", "interrupt") in edges  # the vector

    actuates = [e for e in structure(bundle)["edges"] if e["kind"] == "actuates"]
    assert actuates == [{"kind": "actuates", "pipe": "actuators.unit_a", "object": "plant.unit_a"}]


def test_a_mapped_object_is_an_edge_of_its_own() -> None:
    """A status region is an information path no pipe binding describes.

    The kernel rewrites the region in place when the object changes, so the job reads
    the new value without performing a read -- and a diagram drawn from bindings alone
    shows a job built on ``maps:`` as having no way to learn anything.
    """
    watcher = Descriptor.from_frontmatter(
        {
            "name": "watcher",
            "priority": 50,
            "maps": [{"object": "plant.unit_a", "mode": "ro", "region": "status"}],
        },
        body="watch the unit",
        source="<test>",
    )
    made = CaseBundle(name="maps", descriptors={watcher.name: watcher}, scripts={})
    edges = [e for e in structure(made)["edges"] if e["kind"] == "maps"]
    assert edges == [
        {
            "kind": "maps",
            "descriptor": "watcher",
            "object": "plant.unit_a",
            "mode": "ro",
            "region": "status",
        }
    ]


def test_a_descriptor_carries_its_own_prompt(bundle: CaseBundle) -> None:
    """Clicking a box in the diagram opens the behaviour's file, so the file has to be
    in the payload. The frontmatter is projected field by field already; this is the
    body, and it must be the real one rather than a summary of it."""
    view = structure(bundle)
    bodies = {d["name"]: d["body"] for d in view["descriptors"]}
    for name, descriptor in bundle.descriptors.items():
        assert bodies[str(name)] == descriptor.body
        assert bodies[str(name)].strip(), f"{name} projected an empty prompt"


def test_the_prompt_travels_without_reading_the_case_again(bundle: CaseBundle) -> None:
    """``payload`` performs no I/O, and an exported page opens on a machine that never
    had the case. So the body must come from the loaded bundle, not from ``source`` --
    which this checks by projecting a bundle whose source paths do not exist."""
    doctored = {
        name: replace(d, source="/nowhere/that/exists.md") for name, d in bundle.descriptors.items()
    }
    view = structure(replace(bundle, descriptors=doctored))
    for d in view["descriptors"]:
        assert d["source"] == "/nowhere/that/exists.md"
        assert d["body"].strip(), "the prompt did not survive a source path that is not real"


def test_lint_findings_travel_with_the_structure(bundle: CaseBundle) -> None:
    """A broken tree is exactly when the debugger is wanted, so findings are data
    on the page rather than a reason to refuse to draw it."""
    findings = lint(
        bundle.descriptors,
        pipes=bundle.pipes,
        vectors=bundle.vectors,
        resources=bundle.resources,
    )
    view = structure(bundle, findings)
    assert len(view["lint"]) == len(findings)
    assert {f["severity"] for f in view["lint"]} <= {"error", "warning"}


def test_a_descriptor_projects_its_whole_frontmatter(bundle: CaseBundle) -> None:
    alarm = next(d for d in structure(bundle)["descriptors"] if d["name"] == "threshold-alarm")
    assert alarm["pinned"] is True
    assert alarm["preemptible"] is False  # masked; the lint caps its budget
    assert alarm["budget"]["tokens"] == 64
    assert alarm["writes"] == ["plant.unit_a"]
    assert alarm["pipes"] == {"tools": "actuators.unit_a"}


# --- frames -----------------------------------------------------------------


def test_the_deltas_reproduce_every_frame_exactly(records: Sequence[JournalRecord]) -> None:
    """The specification of the page's merge, checked against the fold it mirrors."""
    expected = fold(r.event for r in records).frames
    payload = frames(records)

    assert payload["count"] == len(expected)
    rebuilt = payload["base"]
    for index, view in enumerate(expected):
        if index:
            rebuilt = apply_delta(rebuilt, payload["deltas"][index - 1])
        want = serde.encode(view)
        want["headline"] = view.headline()
        assert rebuilt == want, f"frame {index} did not survive the delta round trip"


def test_deltas_are_smaller_than_the_frames_they_replace(
    records: Sequence[JournalRecord],
) -> None:
    """Not a performance test -- a statement that the encoding is doing its job. A
    delta stream no smaller than the frames means the keying broke."""
    import json

    payload = frames(records)
    full = sum(len(json.dumps(serde.encode(f))) for f in fold(r.event for r in records).frames)
    delta = len(json.dumps(payload["base"])) + sum(len(json.dumps(d)) for d in payload["deltas"])
    assert delta < full / 2


def test_marks_name_moments_the_specs_make_claims_about(
    records: Sequence[JournalRecord],
) -> None:
    payload = frames(records)
    labels = {label for _index, label in payload["marks"]}
    assert {"preemption", "resume", "demotion"} <= labels
    for index, _label in payload["marks"]:
        assert 0 <= index < payload["count"]


def test_lanes_cover_every_frame_once_per_job(records: Sequence[JournalRecord]) -> None:
    payload = frames(records)
    assert payload["lanes"], "a run with jobs must produce lanes"
    for lane in payload["lanes"]:
        runs = lane["runs"]
        assert runs[-1][1] == payload["count"], "a lane must run to the end of the journal"
        for before, after in zip(runs, runs[1:], strict=False):
            assert before[1] == after[0], "lane segments must abut"
            assert before[2] != after[2], "abutting segments must differ, or the run-length broke"


def test_decimation_keeps_fewer_frames(records: Sequence[JournalRecord]) -> None:
    assert frames(records, every=8)["count"] < frames(records)["count"]


def test_an_empty_journal_is_not_an_error() -> None:
    """``fold`` always snapshots at least once, so an empty journal is one frame of an
    empty system rather than nothing. The scrubber then has a single position, which
    is the honest picture of a run that recorded nothing."""
    payload = frames([])
    assert payload["count"] == 1
    assert payload["deltas"] == []
    assert payload["marks"] == []
    assert payload["lanes"] == []
    assert payload["base"]["jobs"] == []


# --- the whole payload ------------------------------------------------------


def test_structure_only_is_a_first_class_mode(bundle: CaseBundle) -> None:
    payload = build_payload(bundle)
    assert "frames" not in payload
    assert payload["structure"]["descriptors"]


# --- the token log ------------------------------------------------------------


def test_the_token_log_carries_every_token_bearing_event(
    records: Sequence[JournalRecord],
) -> None:
    """One entry per movement, and no movement missed. The pane can only show what
    the log carries, so a dropped kind is a silently emptier stream."""
    log = frames(records)["tokens"]
    expected = [
        list(r.event.text)
        for r in records
        if isinstance(r.event, Decoded | Injected | PipeWritten | PipeReadEvent)
    ]
    assert len(log) == len(expected)
    assert [entry[4] for entry in log] == expected


def test_the_log_is_in_journal_order_and_lands_inside_the_timeline(
    records: Sequence[JournalRecord],
) -> None:
    """The page finds "up to here" by taking the prefix at or before the current
    frame, which is only a prefix if the log is sorted and in range."""
    built = frames(records)
    marks = [entry[0] for entry in built["tokens"]]
    assert marks == sorted(marks)
    assert all(0 <= m < built["count"] for m in marks)


def test_decimation_keeps_every_token(records: Sequence[JournalRecord]) -> None:
    """``--every`` drops frames, not tokens: the log is a history and decimating it
    would lose what was said rather than merely where you can stop."""
    whole = frames(records)["tokens"]
    coarse = frames(records, every=8)
    assert [entry[4] for entry in coarse["tokens"]] == [entry[4] for entry in whole]
    assert all(0 <= entry[0] < coarse["count"] for entry in coarse["tokens"])


def test_a_movement_names_who_moved_it(records: Sequence[JournalRecord]) -> None:
    """A decode belongs to no pipe and a device adapter's write to no job. Both are
    facts the pane draws, so both have to survive the projection."""
    log = frames(records)["tokens"]
    decodes = [e for e in log if e[1] == "decode"]
    assert decodes and all(e[2] is not None and e[3] == "" for e in decodes)
    adapter = [e for e in log if e[1] == "write" and e[2] is None]
    assert adapter, "the fixture's alarm arrives through a device adapter"
    assert all(e[3] for e in adapter)


def test_a_pipes_contents_ride_in_the_frame(records: Sequence[JournalRecord]) -> None:
    """Contents are state, not history: they belong to the frame and must therefore
    survive the delta encoding like every other row."""
    built = frames(records)
    frame = built["base"]
    for delta in built["deltas"]:
        frame = apply_delta(frame, delta)
    contents = {p["name"]: p["contents"] for p in frame["pipes"]}
    assert contents["ops.report"] == ["shift", "schedule", "confirmed"]
    assert contents["sensors.threshold"] == []


def test_an_empty_journal_still_offers_a_token_log() -> None:
    """The page reads ``F.tokens`` unconditionally; an absent key would throw."""
    assert frames([])["tokens"] == []


def test_the_page_is_told_the_mark_vocabulary(bundle: CaseBundle) -> None:
    """Shipped as data so the renderer holds no event-kind list of its own."""
    assert build_payload(bundle)["kinds"] == dict(MARKED_KINDS)


def test_frames_carry_the_driver_tick(records: Sequence[JournalRecord]) -> None:
    """A driver moves virtual time between boundaries and never during one, so the
    tick counter must step exactly when the journal's virtual clock does -- that is
    what lets the page label a frame with the ``tN`` a driver's printout counts."""
    views = fold(r.event for r in records).frames
    instants: set[int] = set()
    for record, view in zip(records, views, strict=True):
        instants.add(record.event.clock.virtual_ns)
        assert view.tick == len(instants) - 1


def test_tick_starts_open_each_virtual_instant(records: Sequence[JournalRecord]) -> None:
    """``ticks`` names the first frame of every driver tick, so the transport can
    step boundary-by-boundary by selection alone. Each listed frame must open a new
    tick, and together they must cover every tick the fold counted."""
    built = frames(records)
    views = fold(r.event for r in records).frames
    starts: list[int] = built["ticks"]
    assert starts[0] == 0
    assert starts == sorted(set(starts))
    opened = [views[i].tick for i in starts]
    assert opened == list(range(views[-1].tick + 1))


# --- the context log ----------------------------------------------------------


def _held(window: dict[str, Any]) -> list[str]:
    """A window flattened to the token texts the machine would hold, pads included."""
    out: list[str] = []
    for region in window["regions"]:
        out.extend(region["text"] if region["text"] else [PAD_TOKEN.text] * region["count"])
    return out


def test_the_replayed_window_is_the_machines_transcript(
    kernel: Kernel, records: Sequence[JournalRecord]
) -> None:
    """The load-bearing test: replaying the log reproduces, token for token, what the
    machine held at the end of the run. The page performs the same steps, so this is
    the proof it borrows."""
    windows = replay_context(frames(records)["contexts"])
    assert windows, "a run with jobs must leave windows to compare"
    for job, window in windows.items():
        transcript = [t.text for t in kernel.machine.transcript(JobId(job))]
        assert _held(window) == transcript, f"job {job}'s window differs from its transcript"
        assert window["tokens"] == len(transcript)


def test_injected_segments_sit_where_the_kernel_closed_them(
    records: Sequence[JournalRecord],
) -> None:
    """Every ``segment.closed`` names an offset range; the replayed region must occupy
    exactly that range at that frame, which checks the pads and output segments
    before it as much as the segment itself."""
    built = frames(records)
    checked = 0
    for index, record in enumerate(records):
        if not isinstance(record.event, SegmentClosed):
            continue
        window = replay_context(built["contexts"], upto=index)[int(record.event.job)]
        offset = 0
        for region in window["regions"]:
            if region["segment"] == int(record.event.segment):
                break
            offset += region["count"]
        else:
            pytest.fail(f"segment {record.event.segment} is not in the replayed window")
        assert (offset, offset + region["count"]) == (record.event.start, record.event.end)
        checked += 1
    assert checked, "the fixture closes no segments; this test has lost its subject"


def test_an_injected_region_carries_its_provenance_and_perms(
    records: Sequence[JournalRecord],
) -> None:
    """Ring, integrity and the pipe it arrived on are what make a window readable as
    a protection story rather than a wall of text, and the perms arrive one event
    later than the text, so both halves have to land on the same region."""
    windows = replay_context(frames(records)["contexts"])
    injected = [r for w in windows.values() for r in w["regions"] if r["kind"] == "inject"]
    assert injected
    for region in injected:
        assert region["pipe"] and region["ring"] is not None and region["integrity"] is not None
        assert region["perms"] is not None
    bodies = [r for r in injected if r["ring"] == int(Ring.DESCRIPTOR)]
    assert bodies and all(r["perms"] & Perm.P.value for r in bodies), "a body is pinned"


def test_the_context_log_is_sorted_and_lands_inside_the_timeline(
    records: Sequence[JournalRecord],
) -> None:
    built = frames(records)
    marks = [row[0] for row in built["contexts"]]
    assert marks == sorted(marks)
    assert all(0 <= m < built["count"] for m in marks)
    assert {row[1] for row in built["contexts"]} <= set(CONTEXT_OPS)


def test_decimation_keeps_every_context_operation(records: Sequence[JournalRecord]) -> None:
    whole = frames(records)["contexts"]
    coarse = frames(records, every=8)
    assert [row[1:] for row in coarse["contexts"]] == [row[1:] for row in whole]
    assert all(0 <= row[0] < coarse["count"] for row in coarse["contexts"])


def test_an_empty_journal_still_offers_a_context_log() -> None:
    assert frames([])["contexts"] == []
    assert replay_context([]) == {}


# Proportioned like test_stage_d_virtual_context: spans must dwarf the stub framing.
_PATROL = {
    "name": "patrol",
    "priority": 80,
    "context": {
        "window": 300,
        "eviction": "attention-clock",
        "stub_budget": 120,
        "min_span_age": 0,
        "high_watermark": 0.6,
        "low_watermark": 0.4,
    },
}
_SWEEP = (
    "sweep {i} of the lower drive completed with all readings nominal and no "
    "exceptions logged against the transfer units or the auxiliary pumps on "
    "this pass through the section"
)


def _evicting_run() -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    steps = [{"emit": _SWEEP.format(i=i)} for i in range(24)] + [{"exit": True}]
    kernel = Kernel(
        descriptors={DescriptorName("patrol"): Descriptor.from_frontmatter(_PATROL)},
        machine=ScriptedMachine({"patrol": Script.from_spec(steps)}, block_size=4),
        pipes=PipeTable([]),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="evicting"),
    )
    kernel.start()
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()
    return kernel, events


def test_an_evicted_segment_becomes_a_stub_of_the_journalled_size() -> None:
    """The journal carries a stub's size but never its text -- ``render_stub`` runs in
    the kernel and is not journalled -- so a stub region is a placeholder that says
    which segment it stands for, how many tokens it costs, and where the content
    went. The window's total must still match the machine's."""
    kernel, events = _evicting_run()
    journal = Journal()
    journal.extend(events)
    evictions = [e for e in events if isinstance(e, SegmentEvicted)]
    assert evictions, "the run evicted nothing; the fixture is mis-proportioned"

    windows = replay_context(frames(journal.records)["contexts"])
    (window,) = windows.values()
    (job,) = windows
    assert window["tokens"] == kernel.machine.stats(JobId(job)).resident_tokens
    stubs = {r["segment"]: r for r in window["regions"] if r["kind"] == "stub"}
    live = {r["segment"] for r in window["regions"]}
    evicted = {int(e.segment) for e in evictions}
    assert not live & evicted, "an evicted segment must leave the window"
    # A stub is itself a segment and can be evicted in its turn.
    standing = [e for e in evictions if int(e.stub) not in evicted]
    assert standing
    for eviction in standing:
        stub = stubs[int(eviction.stub)]
        assert stub["count"] == eviction.stub_tokens
        assert stub["stub_of"] == int(eviction.segment)
        assert stub["store"] == eviction.store
        assert stub["text"] == []


def _records(*events: Event) -> list[JournalRecord]:
    return [JournalRecord(seq=i, event=e) for i, e in enumerate(events)]


def _inject(job: int, segment: int, *text: str) -> Injected:
    return Injected(
        clock=Clock(),
        job=JobId(job),
        segment=SegmentId(segment),
        pipe=PipeName("kernel"),
        principal=Principal.KERNEL,
        ring=Ring.KERNEL,
        integrity=Integrity(0),
        tokens=len(text),
        text=text,
    )


def test_a_removal_splice_takes_the_region_out() -> None:
    """A status-region retraction is a splice with nothing in: ``start == end`` and
    ``tokens_in == 0``, and the window must simply lose the region."""
    records = _records(
        _inject(1, 1, "a", "b"),
        _inject(1, 2, "<STATUS", "x>", "0", "</STATUS>"),
        Decoded(clock=Clock(), job=JobId(1), segment=SegmentId(3), tokens=1, text=("say",)),
        Spliced(
            clock=Clock(),
            job=JobId(1),
            start_segment=SegmentId(2),
            end_segment=SegmentId(2),
            tokens_in=0,
            tokens_out=4,
            invalidated_downstream_tokens=1,
        ),
    )
    before = replay_context(frames(records)["contexts"], upto=2)[1]
    after = replay_context(frames(records)["contexts"])[1]
    assert [r["segment"] for r in before["regions"]] == [1, 2, 3]
    assert [r["segment"] for r in after["regions"]] == [1, 3]
    assert (before["tokens"], after["tokens"]) == (7, 3)


def test_a_fork_copies_the_parents_window_before_the_childs_perms_apply() -> None:
    """The kernel revokes R on a compartment child's copied segments before it
    journals the fork, so the log has to put the fork first or the perms would name
    segments the child does not yet hold."""
    records = _records(
        _inject(1, 1, "secret"),
        _inject(1, 2, "granted"),
        PermsChanged(
            clock=Clock(),
            job=JobId(2),
            segment=SegmentId(1),
            from_perms=Perm.R,
            to_perms=Perm.NONE,
        ),
        Forked(clock=Clock(), parent=JobId(1), child=JobId(2), shared_segments=2),
    )
    log = frames(records)["contexts"]
    assert [row[1] for row in log] == ["inject", "inject", "fork", "perms"]
    windows = replay_context(log)
    assert [r["text"] for r in windows[2]["regions"]] == [["secret"], ["granted"]]
    assert windows[2]["regions"][0]["perms"] == Perm.NONE.value
    assert windows[1]["regions"][0]["perms"] is None, "the parent's own copy is untouched"


# --- the machine trace ---------------------------------------------------------


def test_the_trace_is_absent_unless_one_was_written(records: Sequence[JournalRecord]) -> None:
    """``None`` and ``[]`` mean different things to the page: no machine view to
    offer, and a machine that had nothing to say."""
    assert frames(records)["trace"] is None
    assert frames(records, trace=[])["trace"] == []
    assert frames([], trace=[])["trace"] == []


def _sampled(kernel: Kernel) -> RawTrace:
    trace = RawTrace()
    trace.sample(kernel.machine, kernel.events, 0)  # pyright: ignore[reportArgumentType]
    return trace


def test_the_trace_is_re_keyed_to_frames_and_otherwise_carried_whole(
    kernel: Kernel, records: Sequence[JournalRecord]
) -> None:
    """Replaying the re-keyed rows gives the same account as replaying the file."""
    rows = _sampled(kernel).rows
    built = frames(records, trace=rows)["trace"]
    assert [row[0] for row in built] == sorted(row[0] for row in built)
    assert all(0 <= row[0] < frames(records)["count"] for row in built)
    keys = ("seq", "job", "kv_resident", "from_word", "words", "trailing")
    assert replay_trace([dict(zip(keys, row, strict=True)) for row in built]) == replay_trace(rows)


def test_decimation_keeps_every_trace_row(kernel: Kernel, records: Sequence[JournalRecord]) -> None:
    rows = _sampled(kernel).rows
    coarse = frames(records, every=8, trace=rows)
    assert len(coarse["trace"]) == len(rows)
    assert all(0 <= row[0] < coarse["count"] for row in coarse["trace"])
