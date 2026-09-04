# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A descriptor body must not contain anything that looks like kernel framing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CASES = Path(__file__).resolve().parent.parent / "cases"

#: The markers the kernel writes, which a descriptor body must never show.
FRAMING = re.compile(r"<STATUS\b|</STATUS>|<STUB\b|<RESUME>|</RESUME>")


@pytest.mark.parametrize("body", sorted(CASES.rglob("*.md")), ids=lambda p: p.name)
def test_no_descriptor_exhibits_kernel_framing(body: Path) -> None:
    found = FRAMING.findall(body.read_text())
    assert not found, (
        f"{body.relative_to(CASES)} contains kernel framing the model cannot "
        f"distinguish from the real thing: {found}. Describe the line instead of "
        f"writing one."
    )
