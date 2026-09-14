# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A script's ``spawn:`` must name something the descriptor may spawn, checked at load.

The kernel refuses a spawn outside ``children:`` or a declared compartment as a
capability fault mid-run. A script's targets are written by the case author, so the
refusal is decidable before the tree runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.ids import DescriptorName
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script

SPAWN_THEN_EXIT: list[dict[str, Any]] = [{"spawn": "child"}, {"exit": True}]


def _findings(
    frontmatter: Mapping[str, Any], steps: Sequence[Mapping[str, Any]] | None
) -> list[str]:
    name = str(frontmatter["name"])
    descriptors = {
        DescriptorName(name): Descriptor.from_frontmatter(dict(frontmatter), body="b"),
        DescriptorName("child"): Descriptor.from_frontmatter(
            {"name": "child", "priority": 50}, body="b"
        ),
    }
    scripts = {} if steps is None else {name: Script.from_spec(steps)}
    return [
        f.detail
        for f in lint(descriptors, scripts=scripts)
        if f.rule == "undeclared-spawn" and f.severity is Severity.ERROR
    ]


def test_a_spawn_outside_children_is_an_error() -> None:
    details = _findings({"name": "p", "priority": 50}, SPAWN_THEN_EXIT)
    assert len(details) == 1
    assert "'child'" in details[0]


def test_a_declared_child_is_accepted() -> None:
    assert _findings({"name": "p", "priority": 50, "children": ["child"]}, SPAWN_THEN_EXIT) == []


def test_a_compartment_is_accepted_by_its_name() -> None:
    frontmatter = {
        "name": "p",
        "priority": 50,
        "compartments": [{"name": "child", "descriptor": "child"}],
    }
    assert _findings(frontmatter, SPAWN_THEN_EXIT) == []


def test_a_compartment_is_accepted_by_its_descriptor() -> None:
    frontmatter = {
        "name": "p",
        "priority": 50,
        "compartments": [{"name": "reader", "descriptor": "child"}],
    }
    assert _findings(frontmatter, SPAWN_THEN_EXIT) == []


def test_a_descriptor_with_no_script_is_not_flagged() -> None:
    assert _findings({"name": "p", "priority": 50}, None) == []


def test_a_script_that_never_spawns_is_not_flagged() -> None:
    assert _findings({"name": "p", "priority": 50}, [{"emit": "one"}, {"exit": True}]) == []


def test_the_same_bad_target_is_reported_once() -> None:
    steps: list[dict[str, Any]] = [{"spawn": "child"}, {"spawn": "child"}, {"exit": True}]
    assert len(_findings({"name": "p", "priority": 50}, steps)) == 1
