# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The projection: a case directory and a journal become one JSON object.

Pure, and that is the whole discipline of this module. No I/O, no clock, no
argument that is not already loaded -- ``server.py`` does the reading and the
writing, and the page does the drawing. The arithmetic lives here, where pytest can
reach it, for the same reason the fold lives in ``monitor/state.py`` rather than in
a dashboard.

Two halves, because the two questions have two different sources:

**Structure comes from the case.** The journal cannot answer "which job reads which
pipe": ``DescriptorLoaded`` carries a priority and three booleans, ``PipeCreated``
carries a pipe but not who binds it, and the vector table appears in the journal
only at the moment a vector fires. Bindings, read/write sets, capabilities and maps
are declared and never journalled, so the wiring diagram is a projection of
``CaseBundle`` and nothing else.

**Runtime comes from the journal**, through the existing fold. Frames are
``SystemView`` exactly as ``monitor.fold`` produced them; nothing here recomputes a
kernel fact, and anything this module reported that the journal did not carry would
be a fact nobody could verify after the run.

**Tokens are both.** What is sitting in a pipe right now is state, so it rides in
the frame as ``PipeView.contents``. What a job has said is history, so it rides
beside the frames as a flat log -- see ``_token_log`` for why a history inside a
delta-encoded frame is quadratic.

Frames are delta-encoded because a run is long and a frame is not small. The
encoding is mechanical -- changed keys, and changed rows of the keyed lists -- and
``apply_delta`` is its documented inverse, tested to reproduce every frame exactly.
That test is what makes the page's merge safe: the browser performs the same
shallow merge, but the proof that the merge is lossless is in Python.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from zeos.core import serde
from zeos.core.events import Decoded, Injected, PipeReadEvent, PipeWritten
from zeos.core.gates import GateTable
from zeos.core.principals import PrincipalTable
from zeos.descriptor.lint import Finding
from zeos.descriptor.loader import CaseBundle
from zeos.descriptor.schema import Descriptor
from zeos.journal.writer import JournalRecord
from zeos.monitor.state import MARKED_KINDS, SystemView, fold

__all__ = [
    "ALIAS_KIND",
    "KEYED",
    "MOVEMENTS",
    "apply_delta",
    "build_payload",
    "frames",
    "structure",
]

#: Which way tokens travel on a binding, by the alias's conventional role.
#:
#: **A declared approximation.** ``PipeBindings`` records a name per alias and no
#: direction, so this is convention rather than fact: ``stdin`` is the blocking read
#: source, ``stdout`` the output sink, and ``tools`` is a write followed by a read of
#: the result (core §4.3). Any other binding -- ``reset-count`` reaching a second
#: actuator through ``peer:`` -- could be either, so it is drawn as both. The journal
#: is ground truth once there is one: observed reads and writes are what the animated
#: view lights up, and a declared edge nothing ever travels is itself worth seeing.
ALIAS_KIND: Mapping[str, str] = {"stdin": "read", "stdout": "write", "tools": "duplex"}

#: Frame sections that are lists of rows with a stable key, so a delta can carry one
#: row instead of the list. Everything else in a frame is replaced whole.
KEYED: Mapping[str, str] = {
    "jobs": "job",
    "pipes": "name",
    "vectors": "name",
    "resources": "name",
}


# --- structure --------------------------------------------------------------


def _descriptor(d: Descriptor) -> dict[str, Any]:
    """One behaviour, projected for drawing.

    Hand-written rather than ``serde.encode``: ``Descriptor`` carries lists and a
    ``Mapping[str, Any]`` of unparsed frontmatter, which ``serde`` refuses by design.
    A drawing projection is not a round-trip anyway -- it may drop what no pane shows,
    and it must not fail the whole page because a case used a key from a later stage.
    """
    return {
        "name": str(d.name),
        "priority": int(d.priority),
        "pinned": d.pinned,
        "preemptible": d.preemptible,
        "placement": d.placement.value,
        "model": d.model,
        "ring": int(d.ring),
        "integrity": {
            "start": int(d.integrity.start),
            "dynamics": d.integrity.dynamics,
        },
        "budget": {"tokens": d.budget.tokens, "deadline_ns": d.budget.deadline_ns},
        "reads": sorted(str(p) for p in d.reads.patterns),
        "writes": sorted(str(p) for p in d.writes.patterns),
        "pipes": {alias: str(name) for alias, name in _bindings(d)},
        "maps": [
            {"object": str(m.obj), "mode": m.mode, "region": m.region, "on_demand": m.on_demand}
            for m in d.maps
        ],
        "capabilities": [
            {
                "pipe": str(c.pipe),
                "min_integrity": int(c.min_integrity),
                "schema": None if c.schema is None else c.schema.name,
                "schema_bits": None if c.schema is None else round(c.schema.capacity_bits(), 1),
                "rate": None if c.rate is None else f"{c.rate.max_events}/{c.rate.window_ns}ns",
            }
            for c in d.capabilities
        ],
        "requires": {
            "tooling": sorted(d.requires.tooling),
            "locomotion": d.requires.locomotion,
            "sensors": sorted(d.requires.sensors),
            "battery_min": d.requires.battery_min,
        },
        "utterances": [{"pattern": p.pattern, "confirm": p.confirm} for p in d.utterances],
        "children": [str(c) for c in d.children],
        "compartments": [
            {"name": c.name, "descriptor": str(c.descriptor), "integrity": int(c.integrity)}
            for c in d.compartments
        ],
        "resources": [str(r) for r in d.resources],
        "endorsers": [str(e) for e in d.endorsers],
        "on_fault": d.on_fault.kind.value,
        "on_complete": d.on_complete.kind.value,
        "context": {
            "window": d.context.window,
            "eviction": d.context.eviction.value,
            "stub_budget": d.context.stub_budget,
        },
        "source": d.source,
        #: The prompt, verbatim. The frontmatter above it is already projected field by
        #: field, so a descriptor's whole file is reachable from the page: this is the
        #: half no other pane shows, and the half that explains what a job is doing.
        #: Carried from ``Descriptor.body`` rather than re-read from ``source``, which
        #: would put I/O in a module that has none and would break an exported page
        #: opened on a machine that never had the case.
        "body": d.body,
    }


def _bindings(d: Descriptor) -> list[tuple[str, str]]:
    """Alias/pipe pairs in a stable order: the three conventional ones, then the rest."""
    named = [(a, d.pipes.resolve(a)) for a in ("stdin", "stdout", "tools")]
    pairs = [(a, str(p)) for a, p in named if p is not None]
    pairs.extend((a, str(d.pipes.extra[a])) for a in sorted(d.pipes.extra))
    return pairs


def _edges(bundle: CaseBundle) -> list[dict[str, Any]]:
    """Every declared connection, derived once here so the renderer draws rather than
    reconstructs."""
    edges: list[dict[str, Any]] = []
    for name in sorted(bundle.descriptors):
        for alias, pipe in _bindings(bundle.descriptors[name]):
            edges.append(
                {
                    "kind": ALIAS_KIND.get(alias, "duplex"),
                    "descriptor": str(name),
                    "pipe": pipe,
                    "alias": alias,
                }
            )
    for vector in bundle.vectors:
        edges.append(
            {
                "kind": "interrupt",
                "descriptor": str(vector.handler),
                "pipe": str(vector.source),
                "vector": str(vector.name),
                "priority": int(vector.priority),
            }
        )
    for pipe in bundle.pipes:
        if pipe.world_object:
            edges.append({"kind": "actuates", "pipe": str(pipe.name), "object": pipe.world_object})
    # A mapped object is an information path into a job that no pipe binding shows:
    # the kernel rewrites the status region in place when the object changes, so the
    # job reads the new value without ever performing a read. Leaving it undrawn made
    # the wiring of a case built on status regions look like it had no way of learning
    # anything, which is the opposite of true.
    for name in sorted(bundle.descriptors):
        for spec in bundle.descriptors[name].maps:
            edges.append(
                {
                    "kind": "maps",
                    "descriptor": str(name),
                    "object": str(spec.obj),
                    "mode": spec.mode,
                    "region": spec.region,
                }
            )
    return edges


def _principals(table: PrincipalTable | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    return [
        {
            "id": str(envelope.id),
            "label": envelope.label,
            "ceiling": int(envelope.ceiling),
            "ring": int(envelope.ring),
            "integrity": int(envelope.integrity),
            "capabilities": sorted(str(p) for p in envelope.capabilities),
            "unrestricted": envelope.unrestricted,
        }
        for _, envelope in sorted(table.principals.items())
    ]


def _gates(table: GateTable | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    return [
        {
            "pipe": str(spec.pipe),
            "descriptor": str(spec.descriptor),
            "requests": str(spec.requests),
            "verdicts": str(spec.verdicts),
        }
        for _, spec in sorted(table.gates.items())
    ]


def structure(bundle: CaseBundle, findings: Sequence[Finding] = ()) -> dict[str, Any]:
    """What the files declare: the wiring diagram's whole input."""
    return {
        "case": bundle.name,
        "root": None if bundle.root is None else str(bundle.root),
        "descriptors": [_descriptor(bundle.descriptors[n]) for n in sorted(bundle.descriptors)],
        "pipes": [
            {
                "name": str(p.name),
                "ring": int(p.ring),
                "principal": p.principal.value,
                "capacity_tokens": p.capacity_tokens,
                "transport": p.transport,
                "device": p.device,
                "world_object": p.world_object,
            }
            for p in sorted(bundle.pipes, key=lambda p: p.name)
        ],
        "vectors": [
            {
                "name": str(v.name),
                "source": str(v.source),
                "handler": str(v.handler),
                "priority": int(v.priority),
                "policy": v.policy.value,
                "min_interval_ns": v.min_interval_ns,
                "deadline_ns": v.deadline_ns,
            }
            for v in sorted(bundle.vectors, key=lambda v: v.name)
        ],
        "resources": [
            {
                "name": str(r.name),
                "kind": r.kind.value,
                "capacity": r.capacity,
                "authority": r.authority,
                "description": r.description,
            }
            for r in sorted(bundle.resources, key=lambda r: r.name)
        ],
        "platforms": [
            {
                "name": p.name,
                "locomotion": p.locomotion,
                "tooling": sorted(p.tooling),
                "sensors": sorted(p.sensors),
            }
            for p in sorted(bundle.platforms, key=lambda p: p.name)
        ],
        "principals": _principals(bundle.principals),
        "gates": _gates(bundle.gates),
        "world": {str(k): v for k, v in sorted(bundle.world.items())},
        "boot": [str(b) for b in bundle.boot],
        "edges": _edges(bundle),
        "lint": [
            {
                "severity": f.severity.value,
                "rule": f.rule,
                "descriptor": None if f.descriptor is None else str(f.descriptor),
                "detail": f.detail,
            }
            for f in findings
        ],
    }


# --- frames -----------------------------------------------------------------


def _encode(view: SystemView) -> dict[str, Any]:
    """One frame. ``headline`` is folded in here rather than left to the page, so the
    rule that the page performs no arithmetic holds without an asterisk."""
    encoded = serde.encode(view)
    encoded["headline"] = view.headline()
    return encoded


def _delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """What changed between two frames.

    A keyed section carries only its changed rows; everything else is carried whole.
    Row *removal* is not expressible and does not need to be: the fold never drops a
    job, a pipe, a vector or a resource once it has seen one.
    """
    out: dict[str, Any] = {}
    for key, value in current.items():
        field = KEYED.get(key)
        if field is None:
            if previous.get(key) != value:
                out[key] = value
            continue
        before = {row[field]: row for row in previous.get(key, ())}
        changed = [row for row in value if before.get(row[field]) != row]
        if changed:
            out[key] = changed
    return out


def apply_delta(frame: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of ``_delta``, and the specification the page's merge implements.

    Rows are re-sorted by key on the way out because the fold emits every section
    sorted (``Monitor.snapshot``), so reconstructing in key order reproduces the
    frame exactly rather than approximately.
    """
    merged = dict(frame)
    for key, value in delta.items():
        field = KEYED.get(key)
        if field is None:
            merged[key] = value
            continue
        rows = {row[field]: row for row in merged.get(key, ())}
        for row in value:
            rows[row[field]] = row
        merged[key] = [rows[k] for k in sorted(rows)]
    return merged


def _lanes(views: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-job state runs over the whole timeline, for the scheduling strip.

    Run-length encoded here rather than scanned in the browser: a lane is a fact
    about the run, and every frame already holds the state it needs. Each lane
    carries its descriptor name too, so labelling a row costs the page no search.
    """
    runs: dict[int, list[list[Any]]] = {}
    names: dict[int, str] = {}
    for index, view in enumerate(views):
        for row in view.get("jobs", ()):
            job = int(row["job"])
            state = str(row["state"])
            names.setdefault(job, str(row["descriptor"]))
            lane = runs.setdefault(job, [])
            if lane and lane[-1][2] == state:
                lane[-1][1] = index + 1
            else:
                lane.append([index, index + 1, state])
    return [{"job": job, "descriptor": names[job], "runs": runs[job]} for job in sorted(runs)]


#: The token log's four movements, and what each one means. Read and inject are
#: separate rows on purpose: a read takes tokens out of a pipe, an inject puts them
#: into a context, and they are not always the same tokens -- a status-region
#: refresh injects with no read behind it.
MOVEMENTS: Mapping[str, str] = {
    "decode": "the model produced these",
    "inject": "these entered a context",
    "write": "these entered a pipe",
    "read": "these left a pipe",
}


def _token_log(records: Sequence[JournalRecord], *, every: int, count: int) -> list[list[Any]]:
    """Every token that moved, in journal order, tagged with the frame it lands on.

    A *history*, and deliberately not part of a frame. A frame is delta-encoded and
    a job's row is re-sent on every decode, so carrying the growing stream inside it
    would make the payload quadratic in the run's length. Here it costs one entry per
    token-bearing event, and the page shows the prefix at or before the frame being
    read -- a slice of a sorted list, which is selection rather than arithmetic.

    Rows are ``[frame, movement, job, pipe, tokens]``. ``job`` is null for a device
    adapter's write; ``pipe`` is empty for a decode, which belongs to no pipe.
    """
    log: list[list[Any]] = []
    for index, record in enumerate(records):
        # ``fold`` snapshots after applying event ``k * every``, so the first frame
        # that reflects event ``index`` is the one at ``ceil(index / every)``.
        frame = min(-(-index // every), count - 1)
        match record.event:
            case Decoded() as event:
                log.append([frame, "decode", int(event.job), "", list(event.text)])
            case Injected() as event:
                log.append([frame, "inject", int(event.job), str(event.pipe), list(event.text)])
            case PipeWritten() as event:
                job = None if event.job is None else int(event.job)
                log.append([frame, "write", job, str(event.pipe), list(event.text)])
            case PipeReadEvent() as event:
                log.append([frame, "read", int(event.job), str(event.pipe), list(event.text)])
            case _:
                continue
    return log


def _tick_starts(views: Sequence[Mapping[str, Any]]) -> list[int]:
    """The first frame of each driver tick, for the transport's tick stepping.

    Computed here rather than scanned in the browser for the usual reason: the page
    selects from lists the payload has already worked out, it does not derive them.
    """
    starts: list[int] = []
    last: Any = None
    for index, view in enumerate(views):
        if view["tick"] != last:
            starts.append(index)
            last = view["tick"]
    return starts


def frames(records: Iterable[JournalRecord], *, every: int = 1) -> dict[str, Any]:
    """Fold a journal into a scrubbable, delta-encoded timeline."""
    records = list(records)
    timeline = fold((r.event for r in records), every=every)
    views = [_encode(f) for f in timeline.frames]
    if not views:
        return {
            "base": {},
            "deltas": [],
            "marks": [],
            "lanes": [],
            "tokens": [],
            "ticks": [],
            "count": 0,
        }
    deltas = [_delta(a, b) for a, b in zip(views, views[1:], strict=False)]
    return {
        "base": views[0],
        "deltas": deltas,
        "marks": [[index, label] for index, label in timeline.marks],
        "lanes": _lanes(views),
        "tokens": _token_log(records, every=every, count=len(views)),
        "ticks": _tick_starts(views),
        "count": len(views),
    }


def build_payload(
    bundle: CaseBundle,
    *,
    records: Sequence[JournalRecord] | None = None,
    findings: Sequence[Finding] = (),
    every: int = 1,
) -> dict[str, Any]:
    """The page's entire input.

    ``frames`` is absent when there is no journal, and a structure-only page is a
    first-class mode rather than a degraded one: reading the wiring of a tree you
    have not run yet is half of what this is for.

    ``kinds`` ships ``MARKED_KINDS`` as data so the page holds no event-kind list of
    its own. Renaming an event then breaks a contract test instead of silently
    emptying a dropdown.
    """
    payload: dict[str, Any] = {
        "structure": structure(bundle, findings),
        "kinds": dict(MARKED_KINDS),
    }
    if records is not None:
        payload["frames"] = frames(records, every=every)
    return payload
