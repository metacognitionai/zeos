# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Loading a descriptor tree from disk.

A system is a directory listing: behaviours in markdown
files with YAML frontmatter, plus a small amount of system configuration that binds
them together. Loading is the "compile and link" step -- parse each behaviour,
resolve its pipes and vectors, and hand the kernel something already checked.

Scripts live in the frontmatter under ``script:`` and are pulled out here rather
than being part of ``Descriptor``. That placement is deliberate: a script is M0
scaffolding standing in for the model's behaviour, not part of the descriptor
format. When a real model arrives the key simply goes unused, and nothing in the
descriptor schema has to change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from zeos.core.embodiment import PlatformProfile
from zeos.core.gates import ALLOW, VETO, GateSpec, GateTable
from zeos.core.ids import (
    DescriptorName,
    Integrity,
    ObjectName,
    PipeName,
    Principal,
    PrincipalId,
    Priority,
    ResourceKind,
    ResourceName,
    Ring,
    VectorName,
    VectorPolicy,
)
from zeos.core.pipes import DEFAULT_CAPACITY, PipeSpec
from zeos.core.principals import CompilationTarget, PrincipalEnvelope, PrincipalTable
from zeos.core.resources import ResourceSpec
from zeos.core.vectors import VectorSpec
from zeos.descriptor.schema import Descriptor, DescriptorError, parse_duration_ns
from zeos.machine.scripted import Script

__all__ = ["CaseBundle", "load_case", "parse_descriptor_file", "split_frontmatter"]

_FENCE = "---"
_DESCRIPTOR_DIRS = ("goals", "handlers", "services", "guards")


@dataclass(frozen=True, slots=True)
class CaseBundle:
    """Everything needed to run one case."""

    name: str
    descriptors: Mapping[DescriptorName, Descriptor]
    scripts: Mapping[str, Script]
    pipes: tuple[PipeSpec, ...] = ()
    vectors: tuple[VectorSpec, ...] = ()
    resources: tuple[ResourceSpec, ...] = ()
    platforms: tuple[PlatformProfile, ...] = ()
    #: Who may ask for what. Site configuration, like the resources:
    #: which humans exist and what they may cause is not a behaviour's business.
    principals: PrincipalTable | None = None
    #: Semantic guards on actuator paths.
    gates: GateTable | None = None
    world: Mapping[ObjectName, str] = field(default_factory=dict[ObjectName, str])
    #: Descriptors spawned at start. Handlers are not booted -- they are dispatched
    #: by their vectors -- so this is normally just the goal jobs.
    boot: tuple[DescriptorName, ...] = ()
    root: Path | None = None


def split_frontmatter(text: str, *, source: str) -> tuple[Mapping[str, Any], str]:
    """Split ``---`` YAML frontmatter from the markdown body.

    The body is the prompt and the frontmatter is the contract; a file with no
    frontmatter is an error rather than a body-only descriptor, because a behaviour
    with no declared contract cannot be scheduled, protected, or linked.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        raise DescriptorError(f"{source}: expected YAML frontmatter opening '---'")
    for index in range(1, len(lines)):
        if lines[index].strip() == _FENCE:
            loaded: Any = yaml.safe_load("\n".join(lines[1:index])) or {}
            if not isinstance(loaded, Mapping):
                raise DescriptorError(f"{source}: frontmatter must be a mapping")
            body = "\n".join(lines[index + 1 :]).strip()
            return cast("Mapping[str, Any]", loaded), body
    raise DescriptorError(f"{source}: unterminated frontmatter (no closing '---')")


def parse_descriptor_file(path: Path) -> tuple[Descriptor, Script | None]:
    raw, body = split_frontmatter(path.read_text(encoding="utf-8"), source=str(path))
    descriptor = Descriptor.from_frontmatter(raw, body=body, source=str(path))
    script_spec = descriptor.extra.get("script")
    script: Script | None = None
    if script_spec is not None:
        if not isinstance(script_spec, Sequence) or isinstance(script_spec, str):
            raise DescriptorError(f"{path}: 'script' must be a list of steps")
        steps: list[Mapping[str, Any]] = []
        for step in cast("Sequence[Any]", script_spec):
            if not isinstance(step, Mapping):
                raise DescriptorError(f"{path}: every script step must be a mapping")
            steps.append(cast("Mapping[str, Any]", step))
        script = Script.from_spec(steps)
    return descriptor, script


def load_case(root: Path) -> CaseBundle:
    """Load a case directory into a runnable bundle."""
    if not root.is_dir():
        raise DescriptorError(f"{root}: not a directory")

    descriptors: dict[DescriptorName, Descriptor] = {}
    scripts: dict[str, Script] = {}
    for path in sorted(_descriptor_paths(root)):
        descriptor, script = parse_descriptor_file(path)
        if descriptor.name in descriptors:
            previous = descriptors[descriptor.name].source
            raise DescriptorError(
                f"duplicate descriptor {descriptor.name!r}: {previous} and {path}"
            )
        descriptors[descriptor.name] = descriptor
        if script is not None:
            scripts[str(descriptor.name)] = script

    system = root / "system"
    vectors = _load_vectors(system / "vectors.yaml")
    return CaseBundle(
        name=root.name,
        descriptors=descriptors,
        scripts=scripts,
        pipes=_load_pipes(system / "pipes.yaml"),
        vectors=vectors,
        resources=_load_resources(system / "resources.yaml"),
        platforms=_load_platforms(root / "platforms"),
        principals=_load_principals(system / "principals.yaml"),
        gates=_load_gates(system / "gates.yaml"),
        world=_load_world(system / "world-state.yaml"),
        boot=_load_boot(system / "boot.yaml", root, descriptors, vectors),
        root=root,
    )


def _load_boot(
    path: Path,
    root: Path,
    descriptors: Mapping[DescriptorName, Descriptor],
    vectors: Sequence[VectorSpec],
) -> tuple[DescriptorName, ...]:
    """Which descriptors start at boot.

    Explicit if ``system/boot.yaml`` says so. Otherwise inferred as: everything in
    ``goals/`` that is neither a child of another descriptor nor bound to a vector.
    The inference exists so a small case does not need the ceremony, and it is
    conservative -- anything reachable another way is left for that way to start it.
    """
    raw = _load_yaml(path)
    if raw is not None:
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            raise DescriptorError(f"{path}: expected a list of descriptor names")
        names = tuple(DescriptorName(str(n)) for n in cast("Sequence[Any]", raw))
        for name in names:
            if name not in descriptors:
                raise DescriptorError(f"{path}: boot lists unknown descriptor {name!r}")
        return names

    handlers = {spec.handler for spec in vectors}
    children = {child for d in descriptors.values() for child in d.children}
    goals_dir = root / "goals"
    return tuple(
        name
        for name in sorted(descriptors)
        if name not in handlers
        and name not in children
        and (not goals_dir.is_dir() or Path(descriptors[name].source).parent.name == "goals")
    )


def _descriptor_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for subdir in _DESCRIPTOR_DIRS:
        directory = root / subdir
        if directory.is_dir():
            paths.extend(p for p in directory.glob("*.md"))
    # Also accept descriptors sitting directly in the case root, so a two-file
    # experiment does not need the full directory ceremony.
    paths.extend(p for p in root.glob("*.md"))
    return paths


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_pipes(path: Path) -> tuple[PipeSpec, ...]:
    raw = _load_yaml(path)
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise DescriptorError(f"{path}: expected a list of pipe declarations")
    specs: list[PipeSpec] = []
    for entry in cast("Sequence[Any]", raw):
        if not isinstance(entry, Mapping):
            raise DescriptorError(f"{path}: every pipe entry must be a mapping")
        spec = cast("Mapping[str, Any]", entry)
        name = spec.get("name") or spec.get("pipe")
        if not name:
            raise DescriptorError(f"{path}: pipe entry missing 'name'")
        specs.append(
            PipeSpec(
                name=PipeName(str(name)),
                ring=_ring(spec.get("ring", Ring.TRUSTED.value), source=str(path)),
                principal=_principal(spec.get("principal", "peer_job"), source=str(path)),
                capacity_tokens=int(spec.get("capacity", DEFAULT_CAPACITY)),
                transport=str(spec.get("transport", "local")),
                device=bool(spec.get("device", False)),
                world_object=(
                    None if spec.get("world_object") is None else str(spec["world_object"])
                ),
            )
        )
    return tuple(specs)


def _load_vectors(path: Path) -> tuple[VectorSpec, ...]:
    raw = _load_yaml(path)
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise DescriptorError(f"{path}: expected a list of vector bindings")
    specs: list[VectorSpec] = []
    for entry in cast("Sequence[Any]", raw):
        if not isinstance(entry, Mapping):
            raise DescriptorError(f"{path}: every vector entry must be a mapping")
        spec = cast("Mapping[str, Any]", entry)
        for required in ("vector", "source", "handler", "priority"):
            if required not in spec:
                raise DescriptorError(f"{path}: vector entry missing {required!r}")
        policy_raw = str(spec.get("policy", VectorPolicy.COALESCE.value))
        try:
            policy = VectorPolicy(policy_raw)
        except ValueError as exc:
            raise DescriptorError(f"{path}: unknown vector policy {policy_raw!r}") from exc
        specs.append(
            VectorSpec(
                name=VectorName(str(spec["vector"])),
                source=PipeName(str(spec["source"])),
                handler=DescriptorName(str(spec["handler"])),
                priority=Priority(int(spec["priority"])),
                policy=policy,
                min_interval_ns=parse_duration_ns(
                    spec.get("min_interval"), field_name=f"{path}: min_interval"
                ),
                deadline_ns=parse_duration_ns(spec.get("deadline"), field_name=f"{path}: deadline"),
            )
        )
    return tuple(specs)


def _load_resources(path: Path) -> tuple[ResourceSpec, ...]:
    """Load ``system/resources.yaml`` -- doorways, cranes, docks, work cells.

    These belong to the *environment*, not to any behaviour, which is why they sit
    beside pipes and world state rather than in a descriptor.
    """
    raw = _load_yaml(path)
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise DescriptorError(f"{path}: expected a list of resource declarations")
    specs: list[ResourceSpec] = []
    for entry in cast("Sequence[Any]", raw):
        if not isinstance(entry, Mapping):
            raise DescriptorError(f"{path}: every resource entry must be a mapping")
        spec = cast("Mapping[str, Any]", entry)
        name = spec.get("name") or spec.get("resource")
        if not name:
            raise DescriptorError(f"{path}: resource entry missing 'name'")
        capacity = int(spec.get("capacity", 1))
        kind_raw = spec.get("kind")
        if kind_raw is None:
            kind = ResourceKind.SEMAPHORE if capacity > 1 else ResourceKind.MUTEX
        else:
            try:
                kind = ResourceKind(str(kind_raw))
            except ValueError as exc:
                raise DescriptorError(f"{path}: unknown resource kind {kind_raw!r}") from exc
        specs.append(
            ResourceSpec(
                name=ResourceName(str(name)),
                capacity=capacity,
                kind=kind,
                authority=str(spec.get("authority", "local")),
                description=str(spec.get("description", "")).strip(),
            )
        )
    return tuple(specs)


def _load_principals(path: Path) -> PrincipalTable | None:
    """Load ``system/principals.yaml`` -- the identities and their envelopes.

    Returns None when the file is absent, which is different from an empty table: a
    case with no principals file is a case with no natural-language front door at
    all, and its jobs are all kernel-owned. An empty *table* would still contain the
    kernel principal and would advertise an NLI surface the site never configured.
    """
    raw = _load_yaml(path)
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise DescriptorError(f"{path}: expected a list of principal declarations")
    envelopes: list[PrincipalEnvelope] = []
    for entry in cast("Sequence[Any]", raw):
        if not isinstance(entry, Mapping):
            raise DescriptorError(f"{path}: every principal entry must be a mapping")
        spec = cast("Mapping[str, Any]", entry)
        name = spec.get("id") or spec.get("principal") or spec.get("name")
        if not name:
            raise DescriptorError(f"{path}: principal entry missing 'id'")
        target = str(spec.get("max_target", CompilationTarget.INVOCATION))
        if target not in CompilationTarget.ALL:
            raise DescriptorError(
                f"{path}: {name}: unknown max_target {target!r}; expected one of "
                f"{list(CompilationTarget.ALL)}"
            )
        ceiling_raw = spec.get("ceiling")
        if ceiling_raw is None:
            raise DescriptorError(
                f"{path}: {name}: 'ceiling' is required -- a principal with no "
                f"declared ceiling could ask for any priority"
            )
        envelopes.append(
            PrincipalEnvelope(
                id=PrincipalId(str(name)),
                ceiling=Priority(int(ceiling_raw)),
                capabilities=frozenset(PipeName(str(c)) for c in _as_seq(spec.get("capabilities"))),
                ring=_ring(spec.get("ring", int(Ring.EXTERNAL)), source=f"{path}: {name}"),
                integrity=Integrity(int(spec.get("integrity", 3))),
                max_target=target,
                invocable=frozenset(str(d) for d in _as_seq(spec.get("invocable"))),
                may_elevate=bool(spec.get("may_elevate", False)),
                unrestricted=bool(spec.get("unrestricted", False)),
                label=str(spec.get("label", "")).strip(),
            )
        )
    return PrincipalTable(envelopes)


def _load_gates(path: Path) -> GateTable | None:
    """Load ``system/gates.yaml`` -- which actuator paths have a semantic guard."""
    raw = _load_yaml(path)
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise DescriptorError(f"{path}: expected a list of gate declarations")
    specs: list[GateSpec] = []
    for entry in cast("Sequence[Any]", raw):
        if not isinstance(entry, Mapping):
            raise DescriptorError(f"{path}: every gate entry must be a mapping")
        spec = cast("Mapping[str, Any]", entry)
        pipe = spec.get("pipe") or spec.get("guards")
        descriptor = spec.get("descriptor") or spec.get("gate")
        if not pipe or not descriptor:
            raise DescriptorError(f"{path}: gate entry needs 'pipe' and 'descriptor'")
        failure = str(spec.get("on_gate_failure", VETO))
        if failure not in (ALLOW, VETO):
            raise DescriptorError(
                f"{path}: {descriptor}: on_gate_failure must be {ALLOW!r} or {VETO!r}"
            )
        specs.append(
            GateSpec(
                pipe=PipeName(str(pipe)),
                descriptor=DescriptorName(str(descriptor)),
                requests=PipeName(str(spec.get("requests", f"gates.{descriptor}.requests"))),
                verdicts=PipeName(str(spec.get("verdicts", f"gates.{descriptor}.verdicts"))),
                on_gate_failure=failure,
                timeout_ticks=int(spec.get("timeout_ticks", 32)),
            )
        )
    try:
        return GateTable(specs)
    except ValueError as exc:
        raise DescriptorError(f"{path}: {exc}") from exc


def _load_platforms(directory: Path) -> tuple[PlatformProfile, ...]:
    """Load ``platforms/*.yaml`` -- one profile per body.

    A robot joining the fleet is device hotplug, and this is the driver it presents.
    Profiles live beside the descriptor tree rather than inside it because a body is
    environment: which robots exist is not a behaviour's business.
    """
    if not directory.is_dir():
        return ()
    profiles: list[PlatformProfile] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = _load_yaml(path)
        if not isinstance(raw, Mapping):
            raise DescriptorError(f"{path}: expected a platform profile mapping")
        spec = cast("Mapping[str, Any]", raw)
        name = spec.get("platform") or spec.get("name")
        if not name:
            raise DescriptorError(f"{path}: platform profile missing 'platform'")
        body = cast("Mapping[str, Any]", spec.get("body") or {})
        node = cast("Mapping[str, Any]", spec.get("node") or {})
        locomotion_raw = body.get("locomotion")
        locomotion = (
            str(cast("Mapping[str, Any]", locomotion_raw).get("type", ""))
            if isinstance(locomotion_raw, Mapping)
            else str(locomotion_raw or "")
        )
        battery_raw = body.get("battery")
        battery = 1.0
        if isinstance(battery_raw, Mapping):
            level = cast("Mapping[str, Any]", battery_raw).get("level")
            if level is not None:
                battery = _fraction(level)
        elif battery_raw is not None:
            battery = _fraction(battery_raw)
        profiles.append(
            PlatformProfile(
                name=str(name),
                locomotion=locomotion,
                tooling=frozenset(str(t) for t in _as_seq(body.get("tooling"))),
                sensors=frozenset(str(t) for t in _as_seq(body.get("sensors"))),
                model_class=str(node.get("model_class", "small-local")),
                hbm_pin_budget=int(node.get("hbm_pin_budget", 0)),
                mesh=frozenset(str(m) for m in _as_seq(spec.get("mesh"))),
                resident_handlers=tuple(
                    DescriptorName(str(h)) for h in _as_seq(spec.get("resident_handlers"))
                ),
                pipes={
                    str(k): PipeName(str(v))
                    for k, v in cast("Mapping[str, Any]", spec.get("pipes") or {}).items()
                },
                battery=battery,
                location=str(spec.get("location", "") or ""),
            )
        )
    return tuple(profiles)


def _as_seq(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return cast("Sequence[Any]", value)
    return (value,)


def _fraction(value: Any) -> float:
    text = str(value).strip()
    number = float(text.rstrip("%"))
    return number / 100.0 if text.endswith("%") or number > 1.0 else number


def _load_world(path: Path) -> Mapping[ObjectName, str]:
    raw = _load_yaml(path)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise DescriptorError(f"{path}: expected a mapping of object -> initial value")
    initial = cast("Mapping[str, Any]", raw)
    # ``namespaces:`` documents the vocabulary; ``initial:`` holds starting values.
    values: Any = initial.get("initial", initial)
    if not isinstance(values, Mapping):
        raise DescriptorError(f"{path}: 'initial' must be a mapping")
    typed = cast("Mapping[str, Any]", values)
    return {ObjectName(str(k)): str(v) for k, v in typed.items()}


def _ring(value: Any, *, source: str) -> Ring:
    if isinstance(value, int) and not isinstance(value, bool):
        return Ring(value)
    text = str(value).strip().upper()
    if text.isdigit():
        return Ring(int(text))
    member = Ring.__members__.get(text)
    if member is None:
        raise DescriptorError(
            f"{source}: unknown ring {value!r}; expected 0-3 or one of {sorted(Ring.__members__)}"
        )
    return member


def _principal(value: Any, *, source: str) -> Principal:
    try:
        return Principal(str(value).strip().lower())
    except ValueError as exc:
        raise DescriptorError(
            f"{source}: unknown principal {value!r}; expected one of {[p.value for p in Principal]}"
        ) from exc
