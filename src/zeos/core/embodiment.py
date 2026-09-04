# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Platform profiles and embodiment requirements -- the job/body separation.

ZEOS-Fleet's central move is that **a robot is not where a job lives, it is what a
job wears**:

> A task is not "robot 7's job"; it is a job currently *embodied* in robot 7.
> …
> **A robot is a body a job wears; changing bodies is a resume.**

That follows from a rule the core design has carried from the start -- the
transcript, not the KV cache and not the machine, is the source of truth -- so jobs
are portable across bodies for free.

This module holds the two halves of the interface:

* a **platform profile** is the driver: what a body *is*. Joining the fleet
  is device hotplug.
* an **embodiment requirement** is what a behaviour needs a body to *be*.
  ``requires`` is to bodies what ``reads``/``writes`` is to state: a declared
  interface, checked at allocation, with the same declared-versus-observed
  trajectory -- a job that uses tooling it did not declare is descriptor drift, and
  the journal can say so.

Matching is deliberately split into **feasibility** and **cost**. Fleet marks
``near:`` as "soft: feeds allocation cost, not feasibility", and the distinction
matters: a body that cannot do the job is not a worse choice, it is not a choice.
Collapsing the two would let a sufficiently attractive cost outvote a missing
gripper.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import DescriptorName, PipeName, ResourceName

__all__ = [
    "PlatformProfile",
    "EmbodimentRequirements",
    "MatchResult",
    "lease_name",
    "LEASE_PREFIX",
]

#: Leases are ordinary resources, so they need names that cannot collide with a
#: doorway or a crane. One prefix, reserved.
LEASE_PREFIX = "body:"


def lease_name(platform: str) -> ResourceName:
    return ResourceName(f"{LEASE_PREFIX}{platform}")


def _percent(value: object) -> float | None:
    """Parse ``30%`` / ``30`` / ``0.3`` into a fraction. Returns None if unparseable."""
    if value is None:
        return None
    text = str(value).strip()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*%?$", text)
    if match is None:
        return None
    number = float(match.group(1))
    return number / 100.0 if text.endswith("%") or number > 1.0 else number


def _as_set(value: object) -> frozenset[str]:
    """One item or many, from hand-written YAML. ``tooling: gripper`` and
    ``tooling: [gripper]`` mean the same thing to whoever wrote the descriptor."""
    if value is None:
        return frozenset()
    if isinstance(value, list | tuple | set | frozenset):
        items: list[object] = list(value)  # pyright: ignore[reportUnknownArgumentType]
        return frozenset(str(item) for item in items)
    return frozenset({str(value)})


@dataclass(frozen=True, slots=True)
class PlatformProfile:
    """What a body is. Presented when a robot joins; validated at load."""

    name: str
    locomotion: str = ""
    tooling: frozenset[str] = frozenset()
    sensors: frozenset[str] = frozenset()
    model_class: str = "small-local"
    hbm_pin_budget: int = 0
    #: Physically co-located peers reachable by a direct platform-to-platform link.
    #: Declared here, unused until F2 -- the mesh has no transport yet.
    mesh: frozenset[str] = frozenset()
    resident_handlers: tuple[DescriptorName, ...] = ()
    pipes: Mapping[str, PipeName] = field(default_factory=dict[str, PipeName])
    #: Live state that feeds allocation. In a real fleet these are world-state
    #: objects fed by telemetry; here they are the profile's declared starting
    #: point, and the kernel updates ``self.*`` from them on embodiment.
    battery: float = 1.0
    location: str = ""

    @property
    def lease(self) -> ResourceName:
        return lease_name(self.name)

    def describe(self) -> str:
        return (
            f"{self.name}: {self.locomotion or 'no locomotion'}, "
            f"tooling={sorted(self.tooling) or '[]'}, battery={self.battery:.0%}"
        )


@dataclass(frozen=True, slots=True)
class EmbodimentRequirements:
    """What a behaviour needs a body to be."""

    tooling: frozenset[str] = frozenset()
    locomotion: str = ""
    sensors: frozenset[str] = frozenset()
    battery_min: float = 0.0
    #: Soft. Feeds allocation *cost*, never feasibility.
    near: str = ""

    def __bool__(self) -> bool:
        return bool(
            self.tooling or self.locomotion or self.sensors or self.battery_min or self.near
        )

    @staticmethod
    def parse(raw: Mapping[str, object]) -> EmbodimentRequirements:
        return EmbodimentRequirements(
            tooling=_as_set(raw.get("tooling")),
            locomotion=str(raw.get("locomotion", "") or ""),
            sensors=_as_set(raw.get("sensors")),
            battery_min=_percent(raw.get("battery_min")) or 0.0,
            near=str(raw.get("near", "") or ""),
        )

    def render(self) -> str:
        parts: list[str] = []
        if self.locomotion:
            parts.append(f"locomotion={self.locomotion}")
        if self.tooling:
            parts.append(f"tooling={sorted(self.tooling)}")
        if self.sensors:
            parts.append(f"sensors={sorted(self.sensors)}")
        if self.battery_min:
            parts.append(f"battery>={self.battery_min:.0%}")
        if self.near:
            parts.append(f"near={self.near}")
        return ", ".join(parts) or "(any body)"


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Whether a body can serve, and how good a choice it is."""

    platform: str
    feasible: bool
    #: Lower is better. Only meaningful when feasible.
    cost: float = 0.0
    reasons: tuple[str, ...] = ()

    def render(self) -> str:
        if self.feasible:
            return f"{self.platform} (cost {self.cost:.2f})"
        return f"{self.platform}: {'; '.join(self.reasons)}"


def match(requirements: EmbodimentRequirements, profile: PlatformProfile) -> MatchResult:
    """Check one body against one requirement.

    Feasibility is a conjunction of hard facts about the body. ``near`` never
    appears here -- a distant body that *can* do the job remains a legal answer, just
    a worse one, and Fleet is explicit that proximity feeds cost rather than
    feasibility.
    """
    reasons: list[str] = []
    missing_tooling = sorted(requirements.tooling - profile.tooling)
    if missing_tooling:
        reasons.append(f"lacks tooling {missing_tooling}")
    missing_sensors = sorted(requirements.sensors - profile.sensors)
    if missing_sensors:
        reasons.append(f"lacks sensors {missing_sensors}")
    if requirements.locomotion and requirements.locomotion != profile.locomotion:
        reasons.append(
            f"locomotion is {profile.locomotion or 'none'}, needs {requirements.locomotion}"
        )
    if requirements.battery_min and profile.battery < requirements.battery_min:
        reasons.append(f"battery {profile.battery:.0%} below {requirements.battery_min:.0%}")

    if reasons:
        return MatchResult(platform=profile.name, feasible=False, reasons=tuple(reasons))

    # Cost: prefer a body that is already where the work is, then one with more
    # battery in reserve. Deliberately simple -- the design is explicit that the
    # allocator is a policy module and ZEOS-Fleet "claims no new allocation
    # algorithm", so this is a default to be replaced, not a contribution.
    cost = 0.0
    if requirements.near and profile.location != requirements.near:
        cost += 1.0
    cost += 1.0 - profile.battery
    return MatchResult(platform=profile.name, feasible=True, cost=cost)


def feasible_platforms(
    requirements: EmbodimentRequirements, profiles: Sequence[PlatformProfile]
) -> tuple[MatchResult, ...]:
    """Every body that could serve, best first. Ties break by name, for determinism."""
    results = [match(requirements, p) for p in profiles]
    usable = [r for r in results if r.feasible]
    return tuple(sorted(usable, key=lambda r: (r.cost, r.platform)))


def unsatisfiable(
    requirements: EmbodimentRequirements, profiles: Sequence[PlatformProfile]
) -> tuple[str, ...]:
    """Why *no* body in the fleet can serve -- for the load-time check.

    Fleet is specific that this is rejected "at load, not discovered at allocation":
    a descriptor asking for a gripper no robot has is a design error, and finding it
    when the mission is dispatched is finding it too late.
    """
    if not profiles:
        return ("no platform profiles are declared",)
    if feasible_platforms(requirements, profiles):
        return ()
    return tuple(
        f"{r.platform}: {'; '.join(r.reasons)}" for r in (match(requirements, p) for p in profiles)
    )
