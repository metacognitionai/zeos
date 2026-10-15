# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Segments are block-aligned by construction, so that the per-block mask can hide one
segment without touching its neighbour. Output used to open where the previous
injection ended, mid-block (only ``Kernel._inject`` padded), so the output a job wrote
after a read shared a block with the read's text.

This is ``test_masked_segments_are_excluded_from_the_bitmap`` from the stage C suite
with one step added: the parent says something after its read and before the spawn.
The child inherits that output with R revoked; it must share no block with the granted
page, or ``allowed_blocks``, a union over readable segments, would allow the block and
the revoked tokens with it.
"""

from __future__ import annotations

from zeos.core.ids import DescriptorName, PipeName, Principal, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

PIPES = [
    PipeSpec(PipeName("web.fetch"), ring=Ring.EXTERNAL, principal=Principal.TOOL, device=True),
]
PARENT = {
    "name": "supervisor",
    "pipes": {"stdin": "web.fetch"},
    "priority": 50,
    "integrity": {"start": 1},
    "capabilities": [{"pipe": "web.fetch", "min_integrity": 3}],
    "compartments": [
        {"name": "reader", "descriptor": "web-reader", "integrity": 3, "grants": ["web.fetch"]}
    ],
}
CHILD = {
    "name": "web-reader",
    "priority": 60,
    "integrity": {"start": 3},
    "pipes": {"stdin": "web.fetch"},
}
SCRIPTS = {
    "supervisor": [
        {"emit": "secret material"},
        {"read": "web.fetch"},
        {"emit": "a second secret"},
        {"spawn": "reader"},
        {"exit": True},
    ],
    "web-reader": [{"emit": "summarising", "attend": ["web.fetch"]}, {"exit": True}],
}


def test_a_revoked_segment_is_in_no_block_the_child_may_attend() -> None:
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in (PARENT, CHILD)
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in SCRIPTS.items()}, block_size=8),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=[],
        config=KernelConfig(case="segments-share-kv-blocks"),
    )
    kernel.start()
    parent = kernel.spawn(DescriptorName("supervisor"))
    kernel.deliver(PipeName("web.fetch"), "public market data")
    kernel.run_until_quiescent()

    child = next(j for j in kernel.sched.jobs() if j.compartment_of == parent.job_id)
    inherited = {s.id for s in parent.segments.all()}
    allowed = child.segments.allowed_blocks()
    for record in child.segments.all():
        if record.id in inherited and not record.readable and record.tokens > 0:
            assert not (child.segments.blocks_for(record) & allowed), (
                f"segment {record.id} [{record.start},{record.end}) is unreadable, "
                f"yet its blocks {sorted(child.segments.blocks_for(record))} are allowed"
            )
