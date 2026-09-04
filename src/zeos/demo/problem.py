# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Demonstration problems, and the contract that separates them from solutions.

A **problem** is a world: what exists, what the sensors report, what the actuators
accept, what happens and when, and what counts as success. It contains no
behaviours. A **solution** is a descriptor tree that tries to handle it. The two are
loaded from separate directories and are independently versionable, so the same
problem can be attempted by many solutions and a solution can be re-run against a
harder scenario.

The interface between them is the **pipe contract**, and that is not an invention of
this module -- it is what the design already says a pipe is::

    Pipes are the function signature: what this behaviour consumes and produces. A
    descriptor that reads ``door.sensor`` and writes ``user.notifications`` composes
    with anything speaking those pipes -- the author of either side needs to know the
    pipe's schema, not the other author.
                                                    -- the ZEOS Programming Model

So the split falls out of the architecture rather than being imposed on it:

* the **problem** owns ``pipes`` and ``world-state`` -- the environment;
* the **solution** owns ``descriptors``, ``vectors``, and ``boot`` -- the behaviour;
* a solution that ships its own ``pipes.yaml`` is rejected, because it would be
  defining the world it is supposed to be tested against.

That last rule is the one that keeps the suite honest. Without it a solution can
quietly widen its own environment -- declare the actuator it wishes existed, or
downgrade a ring-3 feed to ring 2 -- and pass by changing the question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import yaml

from zeos.core.ids import ObjectName, PipeName, Principal, ResourceKind, ResourceName, Ring
from zeos.core.pipes import DEFAULT_CAPACITY, PipeSpec
from zeos.core.resources import ResourceSpec
from zeos.descriptor.lint import Finding, Severity
from zeos.descriptor.schema import DescriptorError

__all__ = [
    "PipeContract",
    "WorldEntry",
    "Contract",
    "Problem",
    "ProblemError",
]


class ProblemError(ValueError):
    """A malformed problem definition."""


@dataclass(frozen=True, slots=True)
class PipeContract:
    """One pipe the world offers, with the role it plays.

    ``role`` is documentation for a solution author *and* a check: a solution that
    writes to a sensor pipe, or reads an actuator as if it were a data source, has
    misunderstood the problem, and the harness can say so before the run.
    """

    name: PipeName
    role: str  # "sensor" | "actuator" | "report"
    ring: Ring = Ring.TRUSTED
    principal: Principal = Principal.DEVICE
    capacity_tokens: int = DEFAULT_CAPACITY
    world_object: ObjectName | None = None
    accepts: tuple[str, ...] = ()
    description: str = ""

    @property
    def is_writable_by_solution(self) -> bool:
        return self.role in ("actuator", "report")

    @property
    def is_readable_by_solution(self) -> bool:
        return self.role in ("sensor", "report")

    def to_spec(self) -> PipeSpec:
        return PipeSpec(
            name=self.name,
            ring=self.ring,
            principal=self.principal,
            capacity_tokens=self.capacity_tokens,
            device=self.role == "sensor",
            world_object=None if self.world_object is None else str(self.world_object),
        )


@dataclass(frozen=True, slots=True)
class WorldEntry:
    obj: ObjectName
    initial: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class Contract:
    """Everything a solution author needs to know, and nothing more."""

    pipes: tuple[PipeContract, ...] = ()
    world: tuple[WorldEntry, ...] = ()
    #: Sensor pipes a solution *must* bind a handler to. A problem that fires an
    #: alarm nobody is listening for is not testing anything.
    required_vectors: tuple[PipeName, ...] = ()
    #: Holdable resources in this world -- doorways, cranes, docks. Environment, so
    #: they belong to the problem, exactly as pipes and world state do.
    resources: tuple[ResourceSpec, ...] = ()

    def pipe(self, name: PipeName) -> PipeContract | None:
        return next((p for p in self.pipes if p.name == name), None)

    def resource_specs(self) -> tuple[ResourceSpec, ...]:
        return self.resources

    def pipe_specs(self) -> tuple[PipeSpec, ...]:
        return tuple(p.to_spec() for p in self.pipes)

    def initial_world(self) -> Mapping[ObjectName, str]:
        return {entry.obj: entry.initial for entry in self.world}

    def render(self) -> str:
        """The contract as a solution author sees it."""
        lines = ["pipes:"]
        for pipe in self.pipes:
            detail = f"    {pipe.name}  [{pipe.role}] ring={pipe.ring.name}"
            if pipe.accepts:
                detail += f" accepts={list(pipe.accepts)}"
            if pipe.world_object:
                detail += f" -> {pipe.world_object}"
            lines.append(detail)
            if pipe.description:
                lines.append(f"        {pipe.description}")
        lines.append("world state:")
        for entry in self.world:
            lines.append(f"    {entry.obj} = {entry.initial!r}  {entry.description}")
        if self.resources:
            lines.append("resources:")
            for resource in self.resources:
                lines.append(
                    f"    {resource.name}  [{resource.kind.value}] capacity={resource.capacity}"
                )
        if self.required_vectors:
            lines.append(f"must handle: {[str(v) for v in self.required_vectors]}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Problem:
    """A world plus a scenario plus success criteria. Contains no behaviours."""

    name: str
    title: str
    description: str
    contract: Contract
    #: Raw criteria specs; parsed by ``zeos.demo.criteria`` to avoid a circular
    #: import, since a criterion needs to know about journal events.
    criteria: tuple[Mapping[str, Any], ...] = ()
    scenario_path: Path | None = None
    root: Path | None = None
    #: Simulated wall-clock cost of one token boundary for this problem. Part of the
    #: problem rather than the solution: it is a property of the machine the world
    #: runs on, and letting a solution choose it would let it buy its own latency.
    ns_per_tick: int = 1_000_000
    block_size: int = 16

    @staticmethod
    def load(root: Path) -> Problem:
        if not root.is_dir():
            raise ProblemError(f"{root}: not a directory")
        spec_path = root / "problem.yaml"
        if not spec_path.is_file():
            raise ProblemError(f"{root}: missing problem.yaml")

        raw = cast("Mapping[str, Any]", yaml.safe_load(spec_path.read_text("utf-8")) or {})
        contract_raw = cast("Mapping[str, Any]", raw.get("contract") or {})

        pipes: list[PipeContract] = []
        for role in ("sensors", "actuators", "reports"):
            for entry in cast("Sequence[Any]", contract_raw.get(role) or ()):
                if not isinstance(entry, Mapping):
                    raise ProblemError(f"{spec_path}: every {role} entry must be a mapping")
                pipes.append(_pipe_contract(cast("Mapping[str, Any]", entry), role[:-1], spec_path))

        world: list[WorldEntry] = []
        for obj, entry in sorted(
            cast("Mapping[str, Any]", contract_raw.get("world") or {}).items()
        ):
            if isinstance(entry, Mapping):
                typed = cast("Mapping[str, Any]", entry)
                world.append(
                    WorldEntry(
                        obj=ObjectName(str(obj)),
                        initial=str(typed.get("initial", "")),
                        description=str(typed.get("description", "")),
                    )
                )
            else:
                world.append(WorldEntry(obj=ObjectName(str(obj)), initial=str(entry)))

        resources: list[ResourceSpec] = []
        for entry in cast("Sequence[Any]", contract_raw.get("resources") or ()):
            if not isinstance(entry, Mapping):
                raise ProblemError(f"{spec_path}: every resource entry must be a mapping")
            rspec = cast("Mapping[str, Any]", entry)
            rname = rspec.get("name")
            if not rname:
                raise ProblemError(f"{spec_path}: resource entry missing 'name'")
            capacity = int(rspec.get("capacity", 1))
            kind_raw = rspec.get("kind")
            kind = (
                ResourceKind(str(kind_raw))
                if kind_raw is not None
                else (ResourceKind.SEMAPHORE if capacity > 1 else ResourceKind.MUTEX)
            )
            resources.append(
                ResourceSpec(
                    name=ResourceName(str(rname)),
                    capacity=capacity,
                    kind=kind,
                    authority=str(rspec.get("authority", "local")),
                    description=str(rspec.get("description", "")).strip(),
                )
            )

        criteria_path = root / "criteria.yaml"
        criteria: tuple[Mapping[str, Any], ...] = ()
        if criteria_path.is_file():
            loaded: Any = yaml.safe_load(criteria_path.read_text("utf-8")) or []
            if not isinstance(loaded, Sequence) or isinstance(loaded, str):
                raise ProblemError(f"{criteria_path}: expected a list of criteria")
            criteria = tuple(
                cast("Mapping[str, Any]", c)
                for c in cast("Sequence[Any]", loaded)
                if isinstance(c, Mapping)
            )

        scenario = root / "scenario.jsonl"
        return Problem(
            name=str(raw.get("name", root.name)),
            title=str(raw.get("title", root.name)),
            description=str(raw.get("description", "")).strip(),
            contract=Contract(
                pipes=tuple(pipes),
                world=tuple(world),
                required_vectors=tuple(
                    PipeName(str(v))
                    for v in cast("Sequence[Any]", raw.get("required_vectors") or ())
                ),
                resources=tuple(resources),
            ),
            criteria=criteria,
            scenario_path=scenario if scenario.is_file() else None,
            root=root,
            ns_per_tick=int(raw.get("ns_per_tick", 1_000_000)),
            block_size=int(raw.get("block_size", 16)),
        )

    # -- contract conformance -------------------------------------------------

    def validate(self, solution: SolutionLike) -> tuple[Finding, ...]:
        """Check a solution against this problem's contract.

        Runs *before* the scenario, so a solution that cannot possibly work fails
        with an explanation rather than a mysterious zero score.
        """
        findings: list[Finding] = []

        if solution.declares_environment:
            findings.append(
                Finding(
                    rule="solution-redefines-environment",
                    severity=Severity.ERROR,
                    detail=(
                        "solution ships its own pipes.yaml or world-state.yaml; the "
                        "environment belongs to the problem, and redefining it lets a "
                        "solution pass by changing the question"
                    ),
                )
            )

        known = {p.name for p in self.contract.pipes}
        for descriptor in solution.descriptors.values():
            for pipe in descriptor.pipes.all_names():
                contract = self.contract.pipe(pipe)
                if contract is None:
                    findings.append(
                        Finding(
                            rule="pipe-not-in-contract",
                            severity=Severity.ERROR,
                            detail=f"binds {str(pipe)!r}, which this problem does not offer",
                            descriptor=descriptor.name,
                        )
                    )
            for capability in descriptor.capabilities:
                contract = self.contract.pipe(capability.pipe)
                if contract is None:
                    findings.append(
                        Finding(
                            rule="capability-not-in-contract",
                            severity=Severity.ERROR,
                            detail=(
                                f"holds a capability for {str(capability.pipe)!r}, "
                                "which this problem does not offer"
                            ),
                            descriptor=descriptor.name,
                        )
                    )
                elif not contract.is_writable_by_solution:
                    findings.append(
                        Finding(
                            rule="write-to-sensor",
                            severity=Severity.ERROR,
                            detail=(
                                f"holds a write capability for {str(capability.pipe)!r}, "
                                f"which is a {contract.role} -- solutions read sensors, "
                                "they do not write them"
                            ),
                            descriptor=descriptor.name,
                        )
                    )

        known_resources = {r.name for r in self.contract.resources}
        for descriptor in solution.descriptors.values():
            for resource in getattr(descriptor, "resources", ()):
                if resource not in known_resources:
                    findings.append(
                        Finding(
                            rule="resource-not-in-contract",
                            severity=Severity.ERROR,
                            detail=(
                                f"declares resource {str(resource)!r}, which this "
                                "problem does not offer"
                            ),
                            descriptor=descriptor.name,
                        )
                    )

        bound = {spec.source for spec in solution.vectors}
        for required in self.contract.required_vectors:
            if required not in bound:
                findings.append(
                    Finding(
                        rule="unhandled-required-source",
                        severity=Severity.ERROR,
                        detail=(
                            f"no vector binds {str(required)!r}; this problem fires "
                            "events there and an unhandled event tests nothing"
                        ),
                    )
                )
        for spec in solution.vectors:
            if spec.source not in known:
                findings.append(
                    Finding(
                        rule="vector-source-not-in-contract",
                        severity=Severity.ERROR,
                        detail=(
                            f"vector {str(spec.name)!r} sources from "
                            f"{str(spec.source)!r}, which this problem does not offer"
                        ),
                    )
                )
        return tuple(findings)


@runtime_checkable
class SolutionLike(Protocol):
    """Structural type for what ``validate`` needs from a solution.

    A Protocol rather than an import, so that ``problem`` does not depend on
    ``solution``. The dependency runs the other way, mirroring the real
    relationship: a problem is meaningful with no solution in existence, while a
    solution without a problem is not.
    """

    @property
    def descriptors(self) -> Mapping[Any, Any]: ...

    @property
    def vectors(self) -> Sequence[Any]: ...

    @property
    def declares_environment(self) -> bool: ...


def _pipe_contract(entry: Mapping[str, Any], role: str, source: Path) -> PipeContract:
    name = entry.get("name")
    if not name:
        raise ProblemError(f"{source}: {role} entry missing 'name'")

    ring_raw = entry.get("ring")
    if ring_raw is None:
        # Sensible defaults by role: the physical world is untrusted, reports are
        # operator-facing. A problem may override, but should not have to state the
        # obvious -- and defaulting a sensor to ring 3 is the safe direction to be
        # wrong in.
        ring = Ring.EXTERNAL if role == "sensor" else Ring.TRUSTED
    elif isinstance(ring_raw, int) and not isinstance(ring_raw, bool):
        ring = Ring(ring_raw)
    else:
        member = Ring.__members__.get(str(ring_raw).strip().upper())
        if member is None:
            raise ProblemError(f"{source}: unknown ring {ring_raw!r}")
        ring = member

    principal_raw = str(entry.get("principal", "device")).strip().lower()
    try:
        principal = Principal(principal_raw)
    except ValueError as exc:
        raise ProblemError(f"{source}: unknown principal {principal_raw!r}") from exc

    accepts_raw = entry.get("accepts") or ()
    accepts = (
        tuple(str(a) for a in cast("Sequence[Any]", accepts_raw))
        if isinstance(accepts_raw, list | tuple)
        else (str(accepts_raw),)
    )

    try:
        return PipeContract(
            name=PipeName(str(name)),
            role=role,
            ring=ring,
            principal=principal,
            capacity_tokens=int(entry.get("capacity", DEFAULT_CAPACITY)),
            world_object=(
                None
                if entry.get("world_object") is None
                else ObjectName(str(entry["world_object"]))
            ),
            accepts=accepts,
            description=str(entry.get("description", "")).strip(),
        )
    except DescriptorError as exc:  # pragma: no cover - defensive
        raise ProblemError(f"{source}: {exc}") from exc


_ = field  # keep the import meaningful for future contract fields
