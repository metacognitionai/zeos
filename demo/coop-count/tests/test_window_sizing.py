# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Every descriptor's window must leave room to work in after the parts that stay put."""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.descriptor.loader import load_case
from zeos.machine.base import tokens_from_text

CASES = sorted((Path(__file__).resolve().parent.parent / "cases").iterdir())

#: Roughly the number of tokens one turn costs.
TURN = 70
TURNS_OF_HEADROOM = 6


#: Checked for every descriptor of every case, not just the counters.
@pytest.mark.parametrize(
    ("case", "name"),
    [(case, name) for case in CASES for name in sorted(load_case(case).descriptors)],
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_the_window_clears_what_cannot_be_evicted(case: Path, name: str) -> None:
    descriptor = load_case(case).descriptors[name]
    policy = descriptor.context
    body = len(tokens_from_text(descriptor.body))
    # One status region per mapped object, four tokens each, none of them evictable.
    regions = 4 * sum(1 for spec in descriptor.maps if spec.is_status_region)
    floor = body + regions + policy.stub_budget
    workable = policy.window - floor

    assert workable > 0, (
        f"{name}: body {body} + regions {regions} + stub budget {policy.stub_budget} "
        f"= {floor} exceeds the window {policy.window}; the pager can never reach its "
        f"low mark and the job evicts from its first turn"
    )
    assert policy.window * policy.high_watermark > floor, (
        f"{name}: high watermark {int(policy.window * policy.high_watermark)} sits below "
        f"the unevictable floor {floor}; eviction is triggered before the job speaks"
    )
    assert workable >= TURN * TURNS_OF_HEADROOM, (
        f"{name}: only {workable} tokens to work in, under {TURN * TURNS_OF_HEADROOM} "
        f"for {TURNS_OF_HEADROOM} turns; the body is {body} tokens"
    )
