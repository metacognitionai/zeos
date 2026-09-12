# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The concurrent-write lint names the objects the two descriptors share, and only those."""

from __future__ import annotations

from zeos.core.ids import DescriptorName
from zeos.descriptor.lint import lint
from zeos.descriptor.schema import Descriptor


def _findings():
    descriptors = {
        DescriptorName(name): Descriptor.from_frontmatter(
            {"name": name, "priority": 50, "writes": [own, "shared"]}, body="b"
        )
        for name, own in (("a", "x.a"), ("b", "x.b"))
    }
    return [f for f in lint(descriptors) if f.rule == "concurrent-write"]


def test_the_shared_object_is_flagged() -> None:
    findings = _findings()
    assert len(findings) == 1
    assert "shared" in findings[0].detail


def test_only_the_shared_object_is_named() -> None:
    assert "x.b" not in _findings()[0].detail
