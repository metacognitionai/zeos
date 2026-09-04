# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The task descriptor: one behaviour, one file.

The frontmatter is not configuration -- it is the behaviour's *interface*, and the
kernel is its type-checker and linker. Everything here is
therefore parsed strictly and rejected loudly: a descriptor that does not typecheck
should fail at load, before the robot moves, rather than surprising anyone at
runtime.

The body is the prompt. The kernel never reads it; only the model does.

Field groups arrive per stage -- core scheduling fields here, MP (rings,
capabilities, endorsers) and VM (window, eviction, maps) as those stages land.
Unrecognised keys are preserved in ``extra`` rather than dropped, so a descriptor
written against a later stage still round-trips through an earlier kernel instead
of silently losing its protection or paging configuration.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

from zeos.core.allocator import GangSpec, ReleasePolicy
from zeos.core.capabilities import Capability, Schema, capabilities_from_spec
from zeos.core.embodiment import EmbodimentRequirements
from zeos.core.ids import (
    DescriptorName,
    EvictionPolicy,
    Integrity,
    ObjectName,
    OnComplete,
    OnFault,
    PipeName,
    Placement,
    PrincipalId,
    Priority,
    ResourceName,
    Ring,
)
from zeos.core.residency import ContextPolicy
from zeos.nli.envelope import Phrasing
from zeos.world.store import ObjectSet

__all__ = [
    "Descriptor",
    "Budget",
    "PipeBindings",
    "CompletionPolicy",
    "FaultPolicy",
    "CompartmentSpec",
    "IntegritySpec",
    "MapSpec",
    "DescriptorError",
    "parse_duration_ns",
]

#: Priority is an arbitrary integer, lower = more urgent. These bounds exist only
#: to catch transposed or defaulted values, not to impose tiers.
MIN_PRIORITY: Final = 0
MAX_PRIORITY: Final = 999

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ns|us|ms|s|m|h)\s*$", re.IGNORECASE)
_UNIT_NS: Final[dict[str, int]] = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "h": 3_600_000_000_000,
}


class DescriptorError(ValueError):
    """A descriptor that cannot be loaded. Always names the descriptor and field."""


def parse_duration_ns(value: Any, *, field_name: str = "duration") -> int | None:
    """Parse ``"40ms"`` / ``"2s"`` / ``none`` into nanoseconds.

    Durations are written with units because a bare number in a latency budget is
    an invitation to a unit error, and unit errors in a latency budget are exactly
    the class of mistake that produces a missed deadline nobody can explain.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "", "null"):
        return None
    if isinstance(value, bool):
        raise DescriptorError(f"{field_name}: expected a duration, got a boolean")
    if isinstance(value, int | float):
        raise DescriptorError(
            f"{field_name}: durations need units (got {value!r}; write e.g. '40ms')"
        )
    match = _DURATION.match(str(value))
    if match is None:
        raise DescriptorError(f"{field_name}: cannot parse duration {value!r}")
    magnitude, unit = match.groups()
    return int(float(magnitude) * _UNIT_NS[unit.lower()])


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard generation budget. Exceeding either raises a fault (core §3)."""

    tokens: int | None = None
    deadline_ns: int | None = None

    @staticmethod
    def parse(raw: Any, *, owner: str) -> Budget:
        if raw is None:
            return Budget()
        if not isinstance(raw, Mapping):
            raise DescriptorError(f"{owner}: 'budget' must be a mapping")
        spec = cast("Mapping[str, Any]", raw)
        tokens = spec.get("tokens")
        if tokens is not None and (not isinstance(tokens, int) or isinstance(tokens, bool)):
            raise DescriptorError(f"{owner}: budget.tokens must be an integer")
        if isinstance(tokens, int) and tokens <= 0:
            raise DescriptorError(f"{owner}: budget.tokens must be positive")
        return Budget(
            tokens=tokens,
            deadline_ns=parse_duration_ns(
                spec.get("deadline"), field_name=f"{owner}: budget.deadline"
            ),
        )


@dataclass(frozen=True, slots=True)
class PipeBindings:
    """A descriptor's pipes -- the function signature of the behaviour.

    ``stdin``/``stdout``/``tools`` are conventional names from the specs; ``extra``
    holds any other named bindings so a descriptor can speak to more than three
    pipes without the schema growing a field per use case.
    """

    stdin: PipeName | None = None
    stdout: PipeName | None = None
    tools: PipeName | None = None
    extra: Mapping[str, PipeName] = field(default_factory=dict[str, PipeName])

    def all_names(self) -> tuple[PipeName, ...]:
        named = [p for p in (self.stdin, self.stdout, self.tools) if p is not None]
        return tuple(named) + tuple(self.extra.values())

    def resolve(self, alias: str) -> PipeName | None:
        match alias:
            case "stdin":
                return self.stdin
            case "stdout":
                return self.stdout
            case "tools":
                return self.tools
            case _:
                return self.extra.get(alias)

    @staticmethod
    def parse(raw: Any, *, owner: str) -> PipeBindings:
        if raw is None:
            return PipeBindings()
        if not isinstance(raw, Mapping):
            raise DescriptorError(f"{owner}: 'pipes' must be a mapping")
        spec = cast("Mapping[str, Any]", raw)
        reserved = {"stdin", "stdout", "tools"}
        extra = {str(k): PipeName(str(v)) for k, v in spec.items() if str(k) not in reserved}

        def get(key: str) -> PipeName | None:
            value = spec.get(key)
            return None if value is None else PipeName(str(value))

        return PipeBindings(
            stdin=get("stdin"), stdout=get("stdout"), tools=get("tools"), extra=extra
        )


@dataclass(frozen=True, slots=True)
class CompletionPolicy:
    """What a handler does to the stack beneath it when it finishes (core §6.3)."""

    kind: OnComplete = OnComplete.RETURN
    depth: int = 0  # for cancel-below
    replacement: DescriptorName | None = None  # for replace-with

    @staticmethod
    def parse(raw: Any, *, owner: str) -> CompletionPolicy:
        if raw is None:
            return CompletionPolicy()
        text = str(raw).strip()
        if text == OnComplete.RETURN.value:
            return CompletionPolicy()
        head, _, arg = text.partition(":")
        head, arg = head.strip(), arg.strip()
        if head == OnComplete.CANCEL_BELOW.value:
            if not arg.isdigit() or int(arg) < 1:
                raise DescriptorError(
                    f"{owner}: on_complete 'cancel-below' needs a positive depth, "
                    f"e.g. 'cancel-below:2' (got {text!r})"
                )
            return CompletionPolicy(kind=OnComplete.CANCEL_BELOW, depth=int(arg))
        if head == OnComplete.REPLACE_WITH.value:
            if not arg:
                raise DescriptorError(
                    f"{owner}: on_complete 'replace-with' needs a descriptor name"
                )
            return CompletionPolicy(kind=OnComplete.REPLACE_WITH, replacement=DescriptorName(arg))
        raise DescriptorError(f"{owner}: unknown on_complete policy {text!r}")


@dataclass(frozen=True, slots=True)
class FaultPolicy:
    kind: OnFault = OnFault.ESCALATE
    handler: DescriptorName | None = None

    @staticmethod
    def parse(raw: Any, *, owner: str) -> FaultPolicy:
        if raw is None:
            return FaultPolicy()
        text = str(raw).strip()
        head, _, arg = text.partition(":")
        head, arg = head.strip(), arg.strip()
        if head == OnFault.HANDLER.value:
            if not arg:
                raise DescriptorError(f"{owner}: on_fault 'handler' needs a descriptor name")
            return FaultPolicy(kind=OnFault.HANDLER, handler=DescriptorName(arg))
        try:
            return FaultPolicy(kind=OnFault(head))
        except ValueError as exc:
            raise DescriptorError(f"{owner}: unknown on_fault policy {text!r}") from exc


@dataclass(frozen=True, slots=True)
class CompartmentSpec:
    """A low-integrity child for dirty reads.

    The cheapest escape hatch from monotone taint decay, and the one the lint
    nudges authors toward: the parent grants the child R on just the dirty
    segments, the child returns a result over a pipe, and the parent's watermark
    never moves.
    """

    name: str
    descriptor: DescriptorName
    integrity: Integrity = Integrity(3)
    #: Source tags the child may attend. Everything else of the parent is
    #: unreadable to it -- not by convention but by attention mask, which is the
    #: whole point: secrets stay unreadable rather than "please don't repeat this".
    grants: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntegritySpec:
    start: Integrity = Integrity(2)
    #: ``low-watermark`` demotes on meaningful reads; ``static`` pins the level, and
    #: is only defensible for jobs with no ring-3 inputs at all.
    dynamics: str = "low-watermark"
    on_demotion: str = "continue"

    @property
    def is_dynamic(self) -> bool:
        return self.dynamics == "low-watermark"

    @staticmethod
    def parse(raw: Any, *, owner: str) -> IntegritySpec:
        if raw is None:
            return IntegritySpec()
        if isinstance(raw, int) and not isinstance(raw, bool):
            return IntegritySpec(start=Integrity(raw))
        if not isinstance(raw, Mapping):
            raise DescriptorError(f"{owner}: 'integrity' must be an integer or a mapping")
        spec = cast("Mapping[str, Any]", raw)
        dynamics = str(spec.get("dynamics", "low-watermark"))
        if dynamics not in ("low-watermark", "static"):
            raise DescriptorError(
                f"{owner}: integrity.dynamics must be 'low-watermark' or 'static', got {dynamics!r}"
            )
        return IntegritySpec(
            start=Integrity(int(spec.get("start", 2))),
            dynamics=dynamics,
            on_demotion=str(spec.get("on_demotion", "continue")),
        )


@dataclass(frozen=True, slots=True)
class MapSpec:
    """A mapped world-state or knowledge object.

    ``region: status`` gives a kernel-maintained live view at the hot tail, refreshed
    when another job's write-set touches the object. A refresh *retracts* the copy it
    supersedes rather than demoting it (``Kernel._retire_status_region``): a stale view
    is regenerable from the world store, so eviction -- which buys a fault handle
    nobody needs, at a stub larger than the region itself -- would never reclaim it.
    For a descheduled job the region is still the tail, so the retraction is free and
    the replacement lands on the same block boundary: rewritten in place, as
    Transformer-OS Appendix C describes it. ``paging: on-demand`` makes the object a
    pre-registered NEED target -- pager sugar, resident only when faulted in.

    The design position (OQ-6) is that descriptors should migrate read-set entries
    into maps over time, shrinking the RESUME diff to the irreducible core of beliefs
    the model *derived* rather than read.
    """

    obj: ObjectName
    mode: str = "ro"
    region: str | None = None  # "status" for an in-place refreshed region
    on_demand: bool = False
    min_refresh_blocks: int = 1

    @property
    def is_status_region(self) -> bool:
        return self.region == "status"

    @staticmethod
    def parse(raw: Any, *, owner: str) -> MapSpec:
        if not isinstance(raw, Mapping):
            raise DescriptorError(f"{owner}: every map entry must be a mapping")
        spec = cast("Mapping[str, Any]", raw)
        obj = spec.get("object")
        if not obj:
            raise DescriptorError(f"{owner}: map entry missing 'object'")
        mode = str(spec.get("mode", "ro"))
        if mode not in ("ro", "rw"):
            raise DescriptorError(f"{owner}: map mode must be 'ro' or 'rw', got {mode!r}")
        return MapSpec(
            obj=ObjectName(str(obj)),
            mode=mode,
            region=None if spec.get("region") is None else str(spec["region"]),
            on_demand=str(spec.get("paging", "")) == "on-demand",
            min_refresh_blocks=int(spec.get("min_refresh_blocks", 1)),
        )


@dataclass(frozen=True, slots=True)
class Descriptor:
    """One behaviour. Immutable "code"; the transcript is the state."""

    name: DescriptorName
    priority: Priority
    body: str = ""
    model: str = "default"
    preemptible: bool = True
    pinned: bool = False
    placement: Placement = Placement.ANY
    budget: Budget = field(default_factory=Budget)
    reads: ObjectSet = field(default_factory=ObjectSet)
    writes: ObjectSet = field(default_factory=ObjectSet)
    pipes: PipeBindings = field(default_factory=PipeBindings)
    children: tuple[DescriptorName, ...] = ()
    on_fault: FaultPolicy = field(default_factory=FaultPolicy)
    on_complete: CompletionPolicy = field(default_factory=CompletionPolicy)
    # -- Protected Mode ------------------------------------------------------
    #: The code ring of the descriptor body. Ring 1 and nothing else, ever (MP §4);
    #: it is a field only so that a case can be explicit about it.
    ring: Ring = Ring.DESCRIPTOR
    integrity: IntegritySpec = field(default_factory=IntegritySpec)
    capabilities: tuple[Capability, ...] = ()
    endorsers: tuple[DescriptorName, ...] = ()
    compartments: tuple[CompartmentSpec, ...] = ()
    # -- Virtual Context ------------------------------------------------------
    context: ContextPolicy = field(default_factory=ContextPolicy)
    #: Source tags whose segments are pinned -- exempt from eviction. The
    #: descriptor body is always pinned; this names anything else.
    pins: tuple[str, ...] = ()
    maps: tuple[MapSpec, ...] = ()
    pager: str = "default"
    # -- Resources (R0) --------------------------------------------------------
    # -- Embodiment (F0) -------------------------------------------------------
    #: What kind of body this behaviour needs. Declared, not named:
    #: a task is not "robot 7's job", it is a job currently embodied in robot 7.
    requires: EmbodimentRequirements = field(default_factory=EmbodimentRequirements)
    #: How cleanly the job can be evicted from a body -- the preemptibility
    #: declaration, applied to flesh instead of tokens.
    release_policy: str = ReleasePolicy.ACTION_BOUNDARY
    #: Coupled manoeuvre this descriptor belongs to, if any.
    gang: GangSpec | None = None
    #: Resources this descriptor may acquire, **in the order it acquires them**.
    #: The order is the declaration: a consistent global order across descriptors
    #: makes deadlock impossible, and the lint checks for disagreement.
    resources: tuple[ResourceName, ...] = ()
    # -- Natural language (N0) -------------------------------------------------
    #: Ways a human may ask for this behaviour.
    #:
    #: **Addressability is opt-in.** A descriptor is reachable by voice only if it
    #: declares a phrasing here, which is the structural form of addressability: "the most
    #: safety-critical layer is not protected from natural language; it is *deaf* to
    #: it." A blacklist would have to anticipate every way of asking; there is nothing
    #: to anticipate when the only reachable descriptors are the ones that opted in.
    utterances: tuple[Phrasing, ...] = ()
    #: Principals permitted to invoke this behaviour by name. Empty means "any that
    #: the site's principal table allows" -- authority still narrows what the
    #: invocation can *do*, so this is for tight deployments rather than the
    #: load-bearing check.
    principals: tuple[PrincipalId, ...] = ()
    #: Staleness tolerance for replica-backed reads. Recorded
    #: and journalled in phase 1; enforced when a real transport exists.
    max_staleness_ns: int | None = None
    #: Frontmatter keys this stage does not yet interpret, preserved rather than
    #: dropped so a descriptor written for MP/VM survives an earlier kernel.
    extra: Mapping[str, Any] = field(default_factory=dict[str, Any])
    source: str = "<memory>"

    @staticmethod
    def from_frontmatter(
        raw: Mapping[str, Any],
        *,
        body: str = "",
        source: str = "<memory>",
        schemas: Mapping[str, Schema] | None = None,
    ) -> Descriptor:
        name_raw = raw.get("name")
        if not name_raw:
            raise DescriptorError(f"{source}: descriptor has no 'name'")
        name = DescriptorName(str(name_raw))

        if "priority" not in raw:
            raise DescriptorError(f"{name}: 'priority' is required (lower = more urgent)")
        priority_raw = raw["priority"]
        if not isinstance(priority_raw, int) or isinstance(priority_raw, bool):
            raise DescriptorError(f"{name}: 'priority' must be an integer")
        if not MIN_PRIORITY <= priority_raw <= MAX_PRIORITY:
            raise DescriptorError(
                f"{name}: priority {priority_raw} outside [{MIN_PRIORITY}, {MAX_PRIORITY}]"
            )

        placement_raw = str(raw.get("placement", Placement.ANY.value))
        try:
            placement = Placement(placement_raw)
        except ValueError as exc:
            raise DescriptorError(
                f"{name}: unknown placement {placement_raw!r}; "
                f"expected one of {[p.value for p in Placement]}"
            ) from exc

        known = {
            "name",
            "priority",
            "model",
            "preemptible",
            "pinned",
            "placement",
            "budget",
            "reads",
            "writes",
            "pipes",
            "children",
            "on_fault",
            "on_complete",
            "max_staleness",
            "ring",
            "integrity",
            "capabilities",
            "endorsers",
            "compartments",
            "context",
            "pins",
            "maps",
            "working_set",
            "pager",
            "resources",
            "requires",
            "release_policy",
            "gang",
            "utterances",
            "principals",
        }

        try:
            capabilities = tuple(
                capabilities_from_spec(
                    [
                        cast("Mapping[str, object]", e)
                        for e in _as_list(raw.get("capabilities"), owner=str(name))
                        if isinstance(e, Mapping)
                    ],
                    schemas or {},
                )
            )
        except ValueError as exc:
            raise DescriptorError(f"{name}: capabilities: {exc}") from exc

        compartments: list[CompartmentSpec] = []
        for entry in _as_list(raw.get("compartments"), owner=str(name)):
            if not isinstance(entry, Mapping):
                raise DescriptorError(f"{name}: every compartment must be a mapping")
            spec = cast("Mapping[str, Any]", entry)
            target = spec.get("descriptor")
            if not target:
                raise DescriptorError(f"{name}: compartment missing 'descriptor'")
            grants_raw: Any = spec.get("grants", ())
            grants = (
                tuple(str(g) for g in cast("Sequence[Any]", grants_raw))
                if isinstance(grants_raw, list | tuple)
                else (str(grants_raw),)
            )
            compartments.append(
                CompartmentSpec(
                    name=str(spec.get("name", target)),
                    descriptor=DescriptorName(str(target)),
                    integrity=Integrity(int(spec.get("integrity", 3))),
                    grants=grants,
                )
            )

        return Descriptor(
            name=name,
            priority=Priority(priority_raw),
            body=body,
            model=str(raw.get("model", "default")),
            preemptible=bool(raw.get("preemptible", True)),
            pinned=bool(raw.get("pinned", False)),
            placement=placement,
            budget=Budget.parse(raw.get("budget"), owner=str(name)),
            reads=_object_set(raw.get("reads"), owner=str(name), key="reads"),
            writes=_object_set(raw.get("writes"), owner=str(name), key="writes"),
            pipes=PipeBindings.parse(raw.get("pipes"), owner=str(name)),
            children=tuple(
                DescriptorName(str(c)) for c in _as_list(raw.get("children"), owner=str(name))
            ),
            on_fault=FaultPolicy.parse(raw.get("on_fault"), owner=str(name)),
            on_complete=CompletionPolicy.parse(raw.get("on_complete"), owner=str(name)),
            max_staleness_ns=parse_duration_ns(
                raw.get("max_staleness"), field_name=f"{name}: max_staleness"
            ),
            ring=Ring(int(raw.get("ring", Ring.DESCRIPTOR))),
            integrity=IntegritySpec.parse(raw.get("integrity"), owner=str(name)),
            capabilities=capabilities,
            endorsers=tuple(
                DescriptorName(str(e)) for e in _as_list(raw.get("endorsers"), owner=str(name))
            ),
            compartments=tuple(compartments),
            context=_context_policy(raw, owner=str(name)),
            pins=tuple(str(p) for p in _as_list(raw.get("pins"), owner=str(name))),
            maps=tuple(
                MapSpec.parse(m, owner=str(name))
                for m in _as_list(raw.get("maps"), owner=str(name))
            ),
            pager=str(raw.get("pager", "default")),
            resources=tuple(
                ResourceName(str(r)) for r in _as_list(raw.get("resources"), owner=str(name))
            ),
            requires=_requires(raw.get("requires"), owner=str(name)),
            release_policy=_release_policy(raw.get("release_policy"), owner=str(name)),
            gang=_gang(raw.get("gang"), owner=str(name)),
            utterances=_utterances(raw.get("utterances"), owner=name),
            principals=tuple(
                PrincipalId(str(p)) for p in _as_list(raw.get("principals"), owner=str(name))
            ),
            extra={k: v for k, v in raw.items() if k not in known},
            source=source,
        )


def _utterances(raw: Any, *, owner: DescriptorName) -> tuple[Phrasing, ...]:
    """Parse ``utterances:`` -- a bare string, or a mapping with a priority and a
    confirmation flag.

    ``confirm: true`` marks a consequential mission that must be echoed back before
    dispatch. It is per-phrasing rather than per-descriptor because the same
    behaviour can be routine one way and consequential another.
    """
    out: list[Phrasing] = []
    for entry in _as_list(raw, owner=str(owner)):
        if isinstance(entry, str):
            out.append(Phrasing(descriptor=owner, pattern=entry))
            continue
        if not isinstance(entry, Mapping):
            raise DescriptorError(f"{owner}: every utterance must be a string or a mapping")
        spec = cast("Mapping[str, Any]", entry)
        pattern = spec.get("say") or spec.get("pattern")
        if not pattern:
            raise DescriptorError(f"{owner}: utterance missing 'say'")
        priority_raw = spec.get("priority")
        if priority_raw is not None and (
            not isinstance(priority_raw, int) or isinstance(priority_raw, bool)
        ):
            raise DescriptorError(f"{owner}: utterance priority must be an integer")
        out.append(
            Phrasing(
                descriptor=owner,
                pattern=str(pattern),
                priority=None if priority_raw is None else Priority(priority_raw),
                confirm=bool(spec.get("confirm", False)),
            )
        )
    return tuple(out)


def _as_list(raw: Any, *, owner: str) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list | tuple):
        return list(raw)  # pyright: ignore[reportUnknownArgumentType]
    raise DescriptorError(f"{owner}: expected a list, got {type(raw).__name__}")


def _object_set(raw: Any, *, owner: str, key: str) -> ObjectSet:
    try:
        return ObjectSet.of(str(item) for item in _as_list(raw, owner=owner))
    except ValueError as exc:
        raise DescriptorError(f"{owner}: {key}: {exc}") from exc


def _context_policy(raw: Mapping[str, Any], *, owner: str) -> ContextPolicy:
    """Parse the ``context:`` and ``working_set:`` blocks into one policy object.

    They are separate in the descriptor because they read as separate concerns to an
    author -- how big is my window, versus how much do I expect to be using -- but the
    kernel needs them together, since admission control compares one against the
    other.
    """
    ctx_raw = raw.get("context")
    ctx: Mapping[str, Any] = (
        cast("Mapping[str, Any]", ctx_raw) if isinstance(ctx_raw, Mapping) else {}
    )
    ws_raw = raw.get("working_set")
    ws: Mapping[str, Any] = cast("Mapping[str, Any]", ws_raw) if isinstance(ws_raw, Mapping) else {}

    policy_name = str(ctx.get("eviction", EvictionPolicy.ATTENTION_CLOCK.value))
    try:
        eviction = EvictionPolicy(policy_name)
    except ValueError as exc:
        raise DescriptorError(
            f"{owner}: unknown eviction policy {policy_name!r}; expected one of "
            f"{[p.value for p in EvictionPolicy]}"
        ) from exc

    defaults = ContextPolicy()
    return ContextPolicy(
        window=int(ctx.get("window", defaults.window)),
        eviction=eviction,
        tau_blocks=float(ctx.get("tau_blocks", defaults.tau_blocks)),
        stub_budget=int(ctx.get("stub_budget", defaults.stub_budget)),
        min_span_age=int(ctx.get("min_span_age", defaults.min_span_age)),
        high_watermark=float(ctx.get("high_watermark", defaults.high_watermark)),
        low_watermark=float(ctx.get("low_watermark", defaults.low_watermark)),
        theta_ws=float(ctx.get("theta_ws", defaults.theta_ws)),
        declared_working_set=(int(ws["declared"]) if "declared" in ws else None),
        on_thrash=str(ws.get("on_thrash", defaults.on_thrash)),
        thrash_threshold=float(ctx.get("thrash_threshold", defaults.thrash_threshold)),
        refault_window_blocks=int(ctx.get("refault_window_blocks", defaults.refault_window_blocks)),
        retract_recompute_ratio=float(
            ctx.get("retract_recompute_ratio", defaults.retract_recompute_ratio)
        ),
    )


def _requires(raw: Any, *, owner: str) -> EmbodimentRequirements:
    if raw is None:
        return EmbodimentRequirements()
    if not isinstance(raw, Mapping):
        raise DescriptorError(f"{owner}: 'requires' must be a mapping")
    return EmbodimentRequirements.parse(cast("Mapping[str, Any]", raw))


def _release_policy(raw: Any, *, owner: str) -> str:
    if raw is None:
        return ReleasePolicy.ACTION_BOUNDARY
    text = str(raw).strip()
    if text not in ReleasePolicy.ALL:
        raise DescriptorError(
            f"{owner}: unknown release_policy {text!r}; expected one of {list(ReleasePolicy.ALL)}"
        )
    return text


def _gang(raw: Any, *, owner: str) -> GangSpec | None:
    """Parse a gang declaration.

    ``sync_bound`` is parsed but only enforced once there is a mesh to carry phase
    synchronisation (F2). What F0 owns is the part that is an *allocation*
    constraint: all-or-none dispatch and all-or-none preemption.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DescriptorError(f"{owner}: 'gang' must be a mapping")
    spec = cast("Mapping[str, Any]", raw)
    members = _as_list(spec.get("members"), owner=owner)
    if not members:
        raise DescriptorError(f"{owner}: gang declares no members")
    coupling = str(spec.get("coupling", "loose"))
    if coupling not in ("rigid", "loose"):
        raise DescriptorError(
            f"{owner}: gang coupling must be 'rigid' or 'loose', got {coupling!r}"
        )
    return GangSpec(
        name=str(spec.get("name", owner)),
        members=tuple(DescriptorName(str(m)) for m in members),
        coupling=coupling,
        sync_bound_ns=parse_duration_ns(
            spec.get("sync_bound"), field_name=f"{owner}: gang.sync_bound"
        ),
        on_member_fault=str(spec.get("on_member_fault", "") or ""),
    )
