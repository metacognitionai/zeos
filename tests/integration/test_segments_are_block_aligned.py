# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Every record in every job's segment table starts on a block edge, so no two records
share a KV block and the per-block mask can hide one without touching its neighbour.

Padding belongs to no record. The three runs cover the three ways a record used to land
mid-block: output opening where an injection ended, a fork copying the parent's open
output, and a stub shorter than the span it replaced shifting everything after it.
"""

from __future__ import annotations

from pathlib import Path

from tests.integration.test_segments_share_kv_blocks import CHILD, PARENT, PIPES, SCRIPTS
from tests.integration.test_stage_d_virtual_context import (
    LONG_JOB,
    build,
    chatter,
    first_evicted,
)
from zeos.core.events import Event, Forked, PagedIn, SegmentEvicted
from zeos.core.ids import DescriptorName, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.loader import load_case
from zeos.descriptor.schema import Descriptor
from zeos.driver import Driver, build_kernel
from zeos.journal.writer import Journal
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

COUNTER = Path(__file__).resolve().parents[2] / "demo" / "counter" / "simple_count"


def check_block_aligned(kernel: Kernel) -> None:
    block_size = kernel.machine.block_size
    for job in kernel.sched.jobs():
        table = job.segments
        owners: dict[int, int] = {}
        for record in table.all():
            assert record.start % block_size == 0, (
                f"job {job.job_id}: {record.describe()} starts mid-block"
            )
            for block in table.blocks_for(record):
                assert block not in owners, (
                    f"job {job.job_id}: block {block} is shared by segments "
                    f"{owners[block]} and {record.id}"
                )
                owners[block] = int(record.id)


def test_the_counter_case_keeps_every_segment_on_a_block_edge() -> None:
    bundle = load_case(COUNTER)
    kernel, transport = build_kernel(
        bundle, journal_sink=[], config=KernelConfig(case=bundle.name, max_ticks=16)
    )
    driver = Driver(kernel, transport=transport, journal=Journal(None))
    driver.boot(bundle.boot)
    driver.run()

    assert any(j.segments.by_tag("self") for j in kernel.sched.jobs()), "nothing was decoded"
    check_block_aligned(kernel)


def test_a_compartment_inherits_only_aligned_records() -> None:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in (PARENT, CHILD)
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in SCRIPTS.items()}, block_size=8),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="segments-block-aligned"),
    )
    kernel.start()
    kernel.spawn(DescriptorName("supervisor"))
    kernel.deliver(PipeName("web.fetch"), "public market data")
    kernel.run_until_quiescent()

    assert any(isinstance(e, Forked) for e in events), "the parent never forked"
    check_block_aligned(kernel)


def test_eviction_and_page_in_keep_every_segment_on_a_block_edge() -> None:
    target = first_evicted(12)
    script = chatter(12) + [{"fault": target}] + chatter(4) + [{"exit": True}]
    kernel, events = build([LONG_JOB], {"patrol": script})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    assert any(isinstance(e, SegmentEvicted) for e in events)
    assert any(isinstance(e, PagedIn) for e in events), "the fault paged nothing in"
    check_block_aligned(kernel)
