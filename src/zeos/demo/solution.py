# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A solution: a descriptor tree, and nothing else.

A solution directory looks like a case directory minus the environment. It has
``goals/``, ``handlers/``, ``services/``, ``guards/`` and a ``system/`` holding
``vectors.yaml`` and ``boot.yaml`` -- but no ``pipes.yaml`` and no
``world-state.yaml``, because those belong to the problem.

The omission is load-bearing rather than tidy. A solution that could declare its own
pipes could give itself the actuator it wished existed, or quietly re-ring a hostile
feed from 3 to 2 and make the injection problem disappear. ``declares_environment``
exists so the harness can refuse that, and refuse it with an explanation.

Loading reuses ``zeos.descriptor.loader`` rather than reimplementing it, so a solution
and a standalone case parse through exactly the same path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from zeos.core.ids import DescriptorName
from zeos.core.vectors import VectorSpec
from zeos.descriptor.loader import load_case
from zeos.descriptor.schema import Descriptor, DescriptorError
from zeos.machine.scripted import Script

__all__ = ["Solution"]

_ENVIRONMENT_FILES = ("pipes.yaml", "world-state.yaml")


@dataclass(frozen=True, slots=True)
class Solution:
    name: str
    descriptors: Mapping[DescriptorName, Descriptor]
    scripts: Mapping[str, Script]
    vectors: tuple[VectorSpec, ...]
    boot: tuple[DescriptorName, ...]
    #: True when the directory contains environment files it has no business owning.
    declares_environment: bool = False
    root: Path | None = None

    @staticmethod
    def load(root: Path) -> Solution:
        if not root.is_dir():
            raise DescriptorError(f"{root}: not a directory")
        bundle = load_case(root)
        declares = any((root / "system" / f).is_file() for f in _ENVIRONMENT_FILES)
        return Solution(
            name=root.name,
            descriptors=bundle.descriptors,
            scripts=bundle.scripts,
            vectors=bundle.vectors,
            boot=bundle.boot,
            declares_environment=declares,
            root=root,
        )

    def describe(self) -> str:
        handlers = {spec.handler for spec in self.vectors}
        goals = [n for n in sorted(self.descriptors) if n not in handlers]
        return (
            f"{self.name}: {len(self.descriptors)} descriptors "
            f"({len(goals)} unbound, {len(handlers)} handler(s)), "
            f"{len(self.vectors)} vector(s)"
        )
