# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The allowed-block mask is installed at a block boundary and not refreshed until the
next one, so it must already allow the block the job writes next: the output segment
that opens there has no token, and no record, until the first decode lands. A job with
nothing unreadable in its context is never denied attention to its own output.
"""

from __future__ import annotations

from zeos.core.events import AttentionDenied, Event
from zeos.core.ids import DescriptorName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

TALKER = {"name": "talker", "priority": 50, "body": "say the words"}
#: One word per decode, as a model seat decodes.
SCRIPTS = {"talker": [{"emit": f"w{i}", "attend": ["self"]} for i in range(36)] + [{"exit": True}]}


def test_a_job_with_nothing_unreadable_is_never_denied_attention() -> None:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("talker"): Descriptor.from_frontmatter(TALKER)},
        machine=ScriptedMachine(
            {n: Script.from_spec(s) for n, s in SCRIPTS.items()}, block_size=16
        ),
        pipes=PipeTable(()),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="mask-misses-the-open-block"),
    )
    kernel.start()
    job = kernel.spawn(DescriptorName("talker"))
    kernel.run_until_quiescent()

    assert all(s.readable for s in job.segments.all() if s.tokens > 0)
    denied = [e.segment for e in events if isinstance(e, AttentionDenied)]
    assert not denied, f"attention denied on the job's own readable output: segments {denied}"
