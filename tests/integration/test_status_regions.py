# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Status regions: a live view of world state, retracted and rewritten rather than piled up.

``maps: [{object: …, region: status}]`` is the durable half of a job's memory. Everything
else in a context is the pager's to take -- spans are evicted to stubs under pressure and
the job is expected to fault them back -- but a status region carries ``Perm.W`` and
``plan_eviction`` skips it unconditionally. That makes it the one place a long-running job
can keep something it must not lose.

Which is why a refresh has to *remove* what it supersedes, and why removal cannot be
delegated to the pager. Eviction exists for spans that might be wanted again, and pays
``STUB_FRAMING_TOKENS`` for a handle to fault them back with; ``stub_size`` therefore
exceeds the span for anything under about eleven tokens, so a status region handed to
``plan_eviction`` yields ``freed <= 0``, is marked net-negative, and is declined at every
boundary for the life of the job. Merely dropping ``Perm.W`` closes the leak on paper and
leaves it open in fact: the copies stay resident, and only the line of code refusing to
reclaim them changes.

A superseded region needs no handle. Its content is ``world.get(obj)`` -- free to render
again, and the current value is already in the window -- so the kernel retracts it: the
span leaves the sequence, no stub, no store, no record. What that costs is downstream KV,
and position makes it cheap: the region was appended at the previous refresh, so the
invalidated extent is whatever the job generated *since*. For a job descheduled on a pipe
read while world updates arrive -- the ordinary life of a status region -- the extent is
zero and the replacement lands on the same block boundary the old one occupied. The region
really is rewritten in place (Appendix C); the tail is merely where the place is.

``retract_recompute_ratio`` bounds the trade, because invalidating a long tail of KV to
reclaim a dozen tokens is the worse bargain. Above it the old demote-to-history behaviour
stands, and says so in the journal.

Nothing exercised any of this before: no fixture in the repository declared ``maps:``.
"""

from __future__ import annotations

from zeos.core.events import Event, Note, Spliced, StatusRegionRetracted
from zeos.core.ids import DescriptorName, ObjectName, Perm, PipeName, Residency
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pcb import Job
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.residency import plan_eviction
from zeos.core.resources import ResourceTable
from zeos.core.segments import SegmentRecord, map_tag
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import ContextPolicy, Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

NOTES = ObjectName("job.notes")
SENSOR = PipeName("sensors.notes")
TAG = map_tag(NOTES)
WATCHER = DescriptorName("watcher")

#: Equal-length values, so a refresh replaces a region with one of exactly its size and
#: any growth in the context is growth in the *number* of copies.
VALUES = ("one", "two", "six", "ten", "abc", "xyz", "def", "ghi")

PRESSURE = ContextPolicy(window=16, high_watermark=0.1, low_watermark=0.05, min_span_age=0)


def _descriptor(name: str, *, ratio: float | None = None) -> Descriptor:
    context: dict[str, object] = {"window": 4096}
    if ratio is not None:
        context["retract_recompute_ratio"] = ratio
    return Descriptor.from_frontmatter(
        {
            "name": name,
            "priority": 50,
            "reads": [str(NOTES)],
            "pipes": {"stdin": "user.cmd"},
            "maps": [{"object": str(NOTES), "mode": "ro", "region": "status"}],
            "context": context,
        },
        body="watch the notes",
    )


def _kernel(descriptor: Descriptor, script: Script) -> tuple[Kernel, ScriptedMachine, list[Event]]:
    machine = ScriptedMachine({descriptor.name: script}, block_size=8)
    sink: list[Event] = []
    world = WorldStore()
    kernel = Kernel(
        descriptors={DescriptorName(descriptor.name): descriptor},
        machine=machine,
        pipes=PipeTable(
            [
                PipeSpec(name=SENSOR, device=True, world_object=str(NOTES)),
                PipeSpec(name=PipeName("user.cmd")),
            ]
        ),
        vectors=VectorTable(),
        world=world,
        resources=ResourceTable(),
        config=KernelConfig(case="status-regions"),
        journal_sink=sink,
    )
    world.set(NOTES, "seeded", at=kernel.clock)
    kernel.start()
    kernel.spawn(DescriptorName(descriptor.name))
    return kernel, machine, sink


def _build() -> tuple[Kernel, ScriptedMachine, list[Event]]:
    """A job parked on a blocking read -- the ordinary life of a status region.

    Every refresh below lands on a descheduled job, which is the case that matters and
    the easy one to get wrong. It is also the case where the region is still the tail, so
    retraction is free.
    """
    kernel, machine, sink = _kernel(
        _descriptor("watcher"),
        Script.from_spec([{"emit": "ready"}, {"read": "stdin"}, {"exit": True}]),
    )
    for _ in range(8):
        if not kernel.tick():
            break
    return kernel, machine, sink


def _generating(ratio: float) -> tuple[Kernel, ScriptedMachine, list[Event]]:
    """A job that keeps writing, so each refresh has real output downstream of the region
    it supersedes -- the case where retraction is no longer free."""
    kernel, machine, sink = _kernel(
        _descriptor("watcher", ratio=ratio),
        Script.from_spec(
            [
                *({"emit": f"thinking about it at length step {i}"} for i in range(10)),
                {"exit": True},
            ]
        ),
    )
    for value in VALUES:
        kernel.tick()
        kernel.deliver(SENSOR, value)
    return kernel, machine, sink


def _regions(kernel: Kernel) -> tuple[Job, tuple[SegmentRecord, ...]]:
    job = next(j for j in kernel.sched.jobs() if j.name == "watcher")
    return job, job.segments.by_tag(TAG)


def _check_table_describes_the_sequence(job: Job, machine: ScriptedMachine) -> None:
    """Retraction renumbers everything downstream, so this is the invariant most at risk."""
    length = len(machine.transcript(job.job_id))
    records = job.segments.all()
    for record in records:
        assert record.start <= record.end <= length, f"out of range: {record.describe()}"
    for earlier, later in zip(records, records[1:], strict=False):
        assert earlier.end <= later.start, f"overlap: {earlier.describe()} / {later.describe()}"


def test_a_region_is_seeded_at_start_and_lands_after_the_body() -> None:
    """A job begins able to see the state it declared it depends on.

    Seeding matters for ordering as much as for content. A world write landing between
    ``spawn`` and first dispatch used to create the region ahead of the descriptor
    body, so the job read the state before the instructions explaining it -- and a job
    whose object simply never changed again saw nothing at all.
    """
    kernel, machine, _ = _build()
    job, regions = _regions(kernel)

    assert len(regions) == 1
    assert Perm.W in regions[0].perms
    body = next(r for r in job.segments.all() if r.provenance.tag == "descriptor")
    assert body.end <= regions[0].start, "the code comes before the state it describes"
    assert "seeded" in " ".join(t.text for t in machine.transcript(job.job_id))


def test_a_blocked_job_still_receives_its_refresh() -> None:
    """The region is a view maintained by the kernel, not something the job fetches.

    A job blocked on a pipe runs no forward passes and cannot ask for anything, so if
    the refresh did not reach it the view would be stale exactly when the job is least
    able to notice.
    """
    kernel, machine, _ = _build()
    job, _ = _regions(kernel)
    assert job.state.value == "blocked"

    kernel.deliver(SENSOR, "one")

    _, regions = _regions(kernel)
    assert [r for r in regions if Perm.W in r.perms]
    assert "one" in " ".join(t.text for t in machine.transcript(job.job_id))


def test_a_refresh_leaves_exactly_one_region() -> None:
    """The superseded span leaves the sequence; it is not demoted and kept."""
    kernel, machine, _ = _build()
    for value in VALUES:
        kernel.deliver(SENSOR, value)

    job, regions = _regions(kernel)
    assert len(regions) == 1, "a retracted region leaves no record behind"
    assert Perm.W in regions[0].perms
    live = machine.transcript(job.job_id)[regions[0].start : regions[0].end]
    assert " ".join(t.text for t in live) == f"<STATUS {NOTES}> {VALUES[-1]} </STATUS>"


def test_refreshing_does_not_grow_the_context() -> None:
    """The leak, closed for real -- not handed to a pager that declines it.

    Under the previous fix every superseded copy stayed resident: ``stub_size(4)`` is
    11, so ``freed`` was -7, ``net_positive`` was false, and ``plan_eviction`` put the
    copy in ``skipped`` at every boundary forever. The window cost grew without bound
    while the leak looked closed.
    """
    kernel, machine, _ = _build()
    job, _ = _regions(kernel)

    kernel.deliver(SENSOR, VALUES[0])
    after_one = machine.stats(job.job_id).resident_tokens

    for value in VALUES[1:]:
        kernel.deliver(SENSOR, value)

    assert machine.stats(job.job_id).resident_tokens == after_one, (
        "seven further refreshes of an equal-length value must cost nothing"
    )
    _check_table_describes_the_sequence(job, machine)


def test_a_tail_region_is_rewritten_in_its_own_place() -> None:
    """Appendix C's "rewritten in place", and the reason retraction is free here.

    The region was injected just after a pad, so it starts on a block boundary.
    Retracting it returns the context to that boundary, and the replacement is laid
    down at the same offset -- no downstream KV invalidated, no alignment disturbed.
    """
    kernel, machine, sink = _build()
    # The seeded region still has the job's own "ready" output after it. One refresh
    # settles the view at the tail, which is where it stays for a descheduled job.
    kernel.deliver(SENSOR, "one")
    _, before = _regions(kernel)
    where = before[0].start

    kernel.deliver(SENSOR, "two")

    job, after = _regions(kernel)
    assert after[0].start == where, "the replacement occupies the offset the old view had"

    retracted = [e for e in sink if isinstance(e, StatusRegionRetracted)]
    assert len(retracted) == 2, "both superseded regions were retracted, not demoted"
    assert retracted[-1].invalidated_downstream_tokens == 0, "a tail retraction is free"
    assert retracted[-1].freed_tokens == before[0].tokens
    _check_table_describes_the_sequence(job, machine)


def test_the_splice_accounting_folds_from_the_journal_alone() -> None:
    """``Spliced.tokens_out`` is what lets replay reconstruct context length."""
    kernel, _, sink = _build()
    for value in VALUES:
        kernel.deliver(SENSOR, value)

    splices = [e for e in sink if isinstance(e, Spliced)]
    assert len(splices) == len(VALUES), "one removal per refresh"
    for splice in splices:
        assert splice.tokens_in == 0, "a retraction puts nothing back"
        assert splice.tokens_out > 0
        assert splice.start_segment == splice.end_segment, "a removal has no successor"


def test_retraction_renumbers_correctly_under_downstream_output() -> None:
    """When the job has generated since the last refresh, the cost is real but bounded,
    and every offset downstream of the hole has to move."""
    kernel, machine, sink = _generating(ratio=32.0)
    job, regions = _regions(kernel)

    retracted = [e for e in sink if isinstance(e, StatusRegionRetracted)]
    assert len(regions) == 1, "still exactly one view"
    assert len(retracted) == len(VALUES)
    assert any(e.invalidated_downstream_tokens > 0 for e in retracted), (
        "the fixture must actually put output after the region, or it tests the free case"
    )
    assert all(e.invalidated_downstream_tokens < 32 for e in retracted), (
        "the invalidated extent is what the job wrote since the last refresh, not the context"
    )
    _check_table_describes_the_sequence(job, machine)


def test_a_costly_retraction_is_declined_and_recorded() -> None:
    """The bound. Below the ratio the kernel retracts; above it, the old behaviour --
    demote to history -- stands, and the journal says why rather than staying silent."""
    kernel, machine, sink = _generating(ratio=0.5)
    job, regions = _regions(kernel)

    declined = [e for e in sink if isinstance(e, Note) and "retract-declined" in e.tags]
    assert declined, "a tight ratio must decline"
    assert "demoted to history" in declined[0].text
    assert len(regions) > 1, "a declined retraction leaves the copy in place"
    assert sum(1 for r in regions if Perm.W in r.perms) == 1, "only one live view"
    _check_table_describes_the_sequence(job, machine)


def test_a_status_region_is_never_an_eviction_candidate() -> None:
    """The property the durability of a long-running job rests on."""
    kernel, _, _ = _build()
    kernel.deliver(SENSOR, "keep me")
    job, regions = _regions(kernel)

    resident = sum(r.tokens for r in job.segments.resident())
    plan = plan_eviction(
        job.segments, PRESSURE, resident_tokens=resident, current_token_clock=10_000
    )
    live = next(r for r in regions if Perm.W in r.perms)
    considered = {c.record.id for c in plan.candidates} | {c.record.id for c in plan.skipped}
    assert considered, "the fixture must actually be under eviction pressure"
    assert live.residency is Residency.RESIDENT
    assert live.id not in considered, (
        "a status region is filtered out before eligibility is even scored, so it "
        "appears in neither the plan nor its skipped list"
    )


def test_a_duplicated_map_declaration_yields_one_region() -> None:
    """``maps:`` is not validated for uniqueness, and a spec loop would refresh once per
    entry -- appending a redundant region per world write."""
    descriptor = Descriptor.from_frontmatter(
        {
            "name": "watcher",
            "priority": 50,
            "reads": [str(NOTES)],
            "pipes": {"stdin": "user.cmd"},
            "maps": [
                {"object": str(NOTES), "mode": "ro", "region": "status"},
                {"object": str(NOTES), "mode": "ro", "region": "status"},
            ],
        },
        body="watch the notes",
    )
    kernel, _, _ = _kernel(
        descriptor, Script.from_spec([{"emit": "ready"}, {"read": "stdin"}, {"exit": True}])
    )
    for _ in range(8):
        if not kernel.tick():
            break
    kernel.deliver(SENSOR, "one")

    _, regions = _regions(kernel)
    assert len(regions) == 1
