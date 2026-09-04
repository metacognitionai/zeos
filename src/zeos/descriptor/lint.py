# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Load-time checks: the compiler errors of the descriptor tree.

Load-time rejection is a category of assurance that prose prompts structurally
cannot provide. The point is not that these checks are
clever -- most are almost trivial -- but that they run *before the robot moves*, and
that a failure is a diff to review rather than an incident to investigate.

Two severities, and the distinction is load-bearing:

* **ERROR** -- the tree will not be run. Reserved for conditions where running would
  violate a guarantee the design claims to enforce (an unpreemptible job that can
  mask interrupts indefinitely; a vector bound to a handler that does not exist).
* **WARNING** -- reported and run anyway. Used where the condition is a smell rather
  than a violation, because a lint that refuses to start over a missing report pipe
  would be a worse default than one that runs and complains.

All three layers' rules are present: core (masking budgets, unknown references),
MP (confused-deputy-by-construction, endorsement schema width), and VM (pinned-only
with on-demand maps, stub budgets, watermark sanity, working-set fit).
"""

from __future__ import annotations

import enum
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass

from zeos.core.allocator import ReleasePolicy
from zeos.core.embodiment import PlatformProfile, unsatisfiable
from zeos.core.gates import GateTable, gate_problems
from zeos.core.ids import DescriptorName, EvictionPolicy, PipeName, Placement, Ring
from zeos.core.pipes import PipeSpec
from zeos.core.principals import (
    KERNEL_PRINCIPAL,
    SAFETY_TIER,
    PrincipalTable,
    ceiling_violations,
    unknown_capabilities,
    unrestricted_principals,
)
from zeos.core.resources import ResourceSpec, lock_order_violations
from zeos.core.vectors import VectorSpec
from zeos.descriptor.schema import Descriptor
from zeos.nli.compiler import Phrasebook, parse_phrasings

__all__ = [
    "Severity",
    "Finding",
    "lint",
    "LintOptions",
    "DEFAULT_MASKING_BUDGET",
    "UNIMPLEMENTED_KEYS",
    "SCAFFOLDING_KEYS",
]

#: Token budget above which ``preemptible: false`` is rejected. A policy dial, not
#: a law: it encodes "a critical section should be short enough that masking
#: interrupts across it is defensible". Tune per deployment; do not remove.
DEFAULT_MASKING_BUDGET = 512

#: Frontmatter this kernel deliberately does not interpret, and which is *not* an
#: error: M0 scaffolding that a real machine backend makes redundant.
SCAFFOLDING_KEYS: Mapping[str, str] = {
    "script": "M0 scripted-token-stream behaviour; unused once a real model runs",
}

#: Frontmatter defined by a spec this kernel has not implemented. Declaring one is
#: an ERROR, because the descriptor is asserting a constraint that will not hold.
#:
#: This list is the machine-checkable statement of the phase boundary, and it is
#: enforced in both directions: a key here must be rejected, and a key that has left
#: must be *interpreted* rather than parked in ``extra``. See
#: ``test_fleet_frontmatter_is_now_interpreted_not_parked``.
UNIMPLEMENTED_KEYS: Mapping[str, str] = {
    # ZEOS-Fleet
    "mesh": "platform-to-platform mesh links do not exist",
    # the multi-threaded extension (future work)
    "threads": "concurrent execution is not implemented; one job runs at a time",
}


class Severity(enum.StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    rule: str
    severity: Severity
    detail: str
    descriptor: DescriptorName | None = None

    def render(self) -> str:
        where = f" [{self.descriptor}]" if self.descriptor else ""
        return f"{self.severity.value}: {self.rule}{where}: {self.detail}"


@dataclass(frozen=True, slots=True)
class LintOptions:
    masking_budget: int = DEFAULT_MASKING_BUDGET
    #: Measured p99 round-trip of the federation link, if there is one. Supplying it
    #: turns on the deadline-vs-RTT placement check.
    link_rtt_p99_ns: int | None = None
    #: Upper bound, in bits, on attacker-chosen content an endorsement schema may
    #: carry before the lint says so. 256 bits is roughly a 32-character string --
    #: enough for a structured verdict, not enough for smuggled prose. A dial
    #: whose right value is genuinely unknown (OQ-3).
    max_schema_bits: float = 256.0


def lint(
    descriptors: Mapping[DescriptorName, Descriptor],
    *,
    pipes: Sequence[PipeSpec] = (),
    vectors: Sequence[VectorSpec] = (),
    resources: Sequence[ResourceSpec] = (),
    platforms: Sequence[PlatformProfile] = (),
    principals: PrincipalTable | None = None,
    gates: GateTable | None = None,
    options: LintOptions | None = None,
) -> tuple[Finding, ...]:
    opts = options or LintOptions()
    findings: list[Finding] = []
    declared_pipes = {p.name for p in pipes}

    rings = {p.name: p.ring for p in pipes}

    for name in sorted(descriptors):
        d = descriptors[name]
        findings.extend(_check_masking(d, opts))
        findings.extend(_check_children(d, descriptors))
        findings.extend(_check_pipes(d, declared_pipes))
        findings.extend(_check_fault_handler(d, descriptors))
        findings.extend(_check_completion(d, descriptors))
        findings.extend(_check_confused_deputy(d, rings, descriptors))
        findings.extend(_check_endorser_width(d, opts))
        findings.extend(_check_context(d))
        findings.extend(_check_unimplemented_keys(d))
        findings.extend(_check_resources(d, {r.name for r in resources}))
        findings.extend(_check_embodiment(d, platforms))
        findings.extend(_check_addressability(d, principals))

    findings.extend(_check_vectors(vectors, descriptors, declared_pipes, opts))
    findings.extend(_check_write_conflicts(descriptors))
    findings.extend(_check_lock_order(descriptors))
    findings.extend(_check_gangs(descriptors, platforms))
    findings.extend(_check_phrasebook(descriptors))
    findings.extend(_check_principals(principals, tuple(sorted(declared_pipes))))
    findings.extend(_check_gate_wiring(gates, descriptors, tuple(sorted(declared_pipes))))
    return tuple(findings)


def _check_masking(d: Descriptor, opts: LintOptions) -> list[Finding]:
    """``preemptible: false`` is interrupt masking, and must be paired with a small
    budget (core §3). Unbounded masking is the cli-with-no-sti bug."""
    if d.preemptible:
        return []
    budget = d.budget.tokens
    if budget is None:
        return [
            Finding(
                rule="unpreemptible-unbounded",
                severity=Severity.ERROR,
                detail=(
                    "preemptible: false with no token budget -- an unpreemptible job "
                    "that never ends masks interrupts forever"
                ),
                descriptor=d.name,
            )
        ]
    if budget > opts.masking_budget:
        return [
            Finding(
                rule="unpreemptible-large-budget",
                severity=Severity.ERROR,
                detail=(
                    f"preemptible: false with budget.tokens={budget}, over the "
                    f"masking limit of {opts.masking_budget}"
                ),
                descriptor=d.name,
            )
        ]
    return []


def _check_children(
    d: Descriptor, descriptors: Mapping[DescriptorName, Descriptor]
) -> list[Finding]:
    return [
        Finding(
            rule="unknown-child",
            severity=Severity.ERROR,
            detail=f"declares child {child!r}, which is not in the tree",
            descriptor=d.name,
        )
        for child in d.children
        if child not in descriptors
    ]


def _check_fault_handler(
    d: Descriptor, descriptors: Mapping[DescriptorName, Descriptor]
) -> list[Finding]:
    handler = d.on_fault.handler
    if handler is not None and handler not in descriptors:
        return [
            Finding(
                rule="unknown-fault-handler",
                severity=Severity.ERROR,
                detail=f"on_fault routes to {handler!r}, which is not in the tree",
                descriptor=d.name,
            )
        ]
    return []


def _check_completion(
    d: Descriptor, descriptors: Mapping[DescriptorName, Descriptor]
) -> list[Finding]:
    replacement = d.on_complete.replacement
    if replacement is not None and replacement not in descriptors:
        return [
            Finding(
                rule="unknown-replacement",
                severity=Severity.ERROR,
                detail=f"on_complete replaces with {replacement!r}, which is not in the tree",
                descriptor=d.name,
            )
        ]
    return []


def _check_pipes(d: Descriptor, declared: Container[str]) -> list[Finding]:
    return [
        Finding(
            rule="undeclared-pipe",
            severity=Severity.WARNING,
            detail=(
                f"binds pipe {pipe!r}, which is not declared in system/pipes.yaml; "
                "it will default to ring 2 / peer_job"
            ),
            descriptor=d.name,
        )
        for pipe in d.pipes.all_names()
        if pipe not in declared
    ]


def _check_confused_deputy(
    d: Descriptor,
    rings: Mapping[PipeName, Ring],
    descriptors: Mapping[DescriptorName, Descriptor],
) -> list[Finding]:
    """Confused-deputy-by-construction.

    A descriptor holding a high-integrity capability *and* a ring-3 read pipe,
    without declaring integrity dynamics, a compartment, or an endorser, is rejected
    at load. It is the direct analogue of rejecting an unpreemptible job with a
    large budget: in both cases the descriptor is asking for a guarantee the kernel
    cannot give it, and the right time to say so is before it runs.

    Note the escape hatches are *alternatives*, not a checklist. Low-watermark
    dynamics alone is sufficient, because then the privileged write will be blocked
    at the boundary once the job goes dirty -- the mechanism holds, and the author
    has acknowledged it.
    """
    privileged = [c for c in d.capabilities if int(c.min_integrity) <= 2]
    if not privileged:
        return []

    dirty_pipes = sorted(
        {str(c.pipe) for c in d.capabilities if rings.get(c.pipe, Ring.TRUSTED) is Ring.EXTERNAL}
        | {str(p) for p in d.pipes.all_names() if rings.get(p, Ring.TRUSTED) is Ring.EXTERNAL}
    )
    if not dirty_pipes:
        return []

    has_mitigation = d.integrity.is_dynamic or bool(d.compartments) or bool(d.endorsers)
    if has_mitigation:
        return []

    return [
        Finding(
            rule="confused-deputy",
            severity=Severity.ERROR,
            detail=(
                f"holds privileged capability "
                f"{[str(c.pipe) for c in privileged]} and reads ring-3 pipe(s) "
                f"{dirty_pipes}, but declares no integrity dynamics, compartment, "
                "or endorser -- nothing stands between the dirty read and the "
                "privileged write"
            ),
            descriptor=d.name,
        )
    ]


def _check_endorser_width(d: Descriptor, opts: LintOptions) -> list[Finding]:
    """Schema width is the security dial, so an over-wide one is worth saying aloud.

    An endorser is the only integrity-*raising* operation in the system, which makes
    its output schema the entire injection channel. ``capacity_bits``
    turns "narrow enough" into a number; this rule flags schemas whose ceiling is so
    high that endorsement is doing no real narrowing. It is a WARNING because the
    right threshold is genuinely unknown (OQ-3) -- the point is to make the quantity
    visible, not to pretend we know where the line is.
    """
    findings: list[Finding] = []
    for capability in d.capabilities:
        schema = capability.schema
        if schema is None:
            continue
        bits = schema.capacity_bits()
        if bits > opts.max_schema_bits:
            findings.append(
                Finding(
                    rule="wide-endorsement-schema",
                    severity=Severity.WARNING,
                    detail=(
                        f"capability {str(capability.pipe)!r} uses schema "
                        f"{schema.name!r} with an upper bound of {bits:.0f} bits of "
                        f"attacker-chosen content (over {opts.max_schema_bits}); "
                        "a wide string field makes endorsement close to a no-op"
                    ),
                    descriptor=d.name,
                )
            )
    return findings


def _check_unimplemented_keys(d: Descriptor) -> list[Finding]:
    """Reject frontmatter this kernel parses into ``extra`` and then ignores.

    ``Descriptor.extra`` was introduced as forward-compatibility: a descriptor
    written against a later stage should survive an earlier kernel rather than
    silently losing configuration. That was right while the later stages did not
    exist yet.

    It is wrong now. ZEOS-Fleet and ZEOS-NLI define frontmatter that is
    **safety-relevant**, and preserving it silently means a descriptor declaring a
    constraint runs without it. The clearest case is ``gang:``, of which Fleet says:

        Preempting one carrier mid-lift while the other continues is strictly worse
        than preempting both -- the invariant exists to make that outcome
        inexpressible.

    Under silent preservation that outcome is not merely expressible, it is the
    default. So a key this kernel knows about but does not enforce is an **error**,
    and an unrecognised key is a **warning** -- both louder than being ignored, which
    is the one behaviour the design's own ethos rules out.
    """
    findings: list[Finding] = []
    for key in sorted(d.extra):
        if key in SCAFFOLDING_KEYS:
            continue
        reason = UNIMPLEMENTED_KEYS.get(key)
        if reason is not None:
            findings.append(
                Finding(
                    rule="unimplemented-frontmatter",
                    severity=Severity.ERROR,
                    detail=(
                        f"declares {key!r}, which this kernel parses but does not "
                        f"enforce -- {reason}. Remove it, or implement the mechanism "
                        "before relying on it."
                    ),
                    descriptor=d.name,
                )
            )
        else:
            findings.append(
                Finding(
                    rule="unrecognised-frontmatter",
                    severity=Severity.WARNING,
                    detail=(
                        f"declares {key!r}, which no stage of this kernel "
                        "interprets; it will be ignored"
                    ),
                    descriptor=d.name,
                )
            )
    return findings


def _check_resources(d: Descriptor, declared: Container[str]) -> list[Finding]:
    return [
        Finding(
            rule="unknown-resource",
            severity=Severity.ERROR,
            detail=(f"declares resource {str(name)!r}, which is not in system/resources.yaml"),
            descriptor=d.name,
        )
        for name in d.resources
        if name not in declared
    ]


def _check_addressability(d: Descriptor, principals: PrincipalTable | None) -> list[Finding]:
    """NLI's addressability layer, made machine-checkable.

    > The most safety-critical layer is not protected from natural language; it is
    > *deaf* to it.

    Addressability is opt-in, so deafness is the default -- but a descriptor can opt
    in by mistake, and the two ways of doing so are worth catching separately.

    The first is a safety-tier behaviour declaring a phrasing. The NLI design says no
    user-spawned job may outrank the safety tier; a safety-tier descriptor that
    *can be asked for* has made the tier reachable by voice, which is the property
    the section exists to deny. Note this is not caught by the ceiling check: the
    ceiling stops a user *raising* a job's priority, and does nothing about a
    descriptor whose declared priority is already there.

    The second is an unpreemptible or pinned behaviour declaring one. Those are
    reflex-tier markers, and the NLI design is explicit that reflexes are reached by compiled
    keyword spotting, never by the language path.
    """
    if not d.utterances:
        return []
    findings: list[Finding] = []
    tier = SAFETY_TIER
    if int(d.priority) <= int(tier):
        findings.append(
            Finding(
                rule="safety-tier-is-addressable",
                severity=Severity.ERROR,
                detail=(
                    f"declares {len(d.utterances)} utterance phrasing(s) at priority "
                    f"{int(d.priority)}, at or above the safety tier ({int(tier)}); "
                    "the safety tier must be deaf to natural language, "
                    "not merely protected from it"
                ),
                descriptor=d.name,
            )
        )
    if not d.preemptible or d.pinned:
        findings.append(
            Finding(
                rule="reflex-is-addressable",
                severity=Severity.ERROR,
                detail=(
                    "declares utterance phrasings but is "
                    + ("pinned" if d.pinned else "unpreemptible")
                    + "; reflex-tier behaviours are reached by compiled keyword "
                    "spotting, never through the language path"
                ),
                descriptor=d.name,
            )
        )
    if d.principals and principals is not None:
        for principal in d.principals:
            if not principals.has(principal):
                findings.append(
                    Finding(
                        rule="unknown-principal",
                        severity=Severity.ERROR,
                        detail=(
                            f"restricts invocation to {str(principal)!r}, which is "
                            "not in system/principals.yaml"
                        ),
                        descriptor=d.name,
                    )
                )
    return findings


def _check_phrasebook(
    descriptors: Mapping[DescriptorName, Descriptor],
) -> list[Finding]:
    """Two descriptors claiming the same phrasing.

    At runtime this resolves deterministically, by descriptor name -- which is worse
    than resolving randomly, because it is silently and repeatably wrong. One of the
    two behaviours is simply unreachable and nothing will ever say so.
    """
    book = Phrasebook.build(parse_phrasings(descriptors))
    return [
        Finding(
            rule="ambiguous-phrasing",
            severity=Severity.ERROR,
            detail=(
                f"{collision}; one of them is unreachable and the winner is decided "
                "by descriptor name"
            ),
        )
        for collision in book.collisions()
    ]


def _check_principals(
    principals: PrincipalTable | None, declared_pipes: Sequence[PipeName]
) -> list[Finding]:
    """The principal table's own checks."""
    if principals is None:
        return []
    findings: list[Finding] = [
        Finding(rule="ceiling-outranks-safety-tier", severity=Severity.ERROR, detail=detail)
        for detail in ceiling_violations(principals)
    ]
    findings.extend(
        Finding(rule="unknown-principal-capability", severity=Severity.ERROR, detail=detail)
        for detail in unknown_capabilities(principals, declared=declared_pipes)
    )
    findings.extend(
        Finding(rule="unrestricted-principal", severity=Severity.WARNING, detail=detail)
        for detail in unrestricted_principals(principals)
    )
    # OQ-N4: an unauthenticated source is a ring-3 source, and a principal declared at
    # ring 0 or 1 is claiming kernel or descriptor trust for words a human said.
    findings.extend(
        Finding(
            rule="principal-claims-kernel-ring",
            severity=Severity.ERROR,
            detail=(
                f"{e.id} arrives at ring {int(e.ring)}; a human utterance is at best "
                "ring 2 (authenticated) and by default ring 3 (open air) -- rings 0 "
                "and 1 are kernel framing and descriptor bodies (MP §4)"
            ),
        )
        for e in principals.all()
        if e.id != KERNEL_PRINCIPAL and int(e.ring) < int(Ring.TRUSTED)
    )
    return findings


def _check_gate_wiring(
    gates: GateTable | None,
    descriptors: Mapping[DescriptorName, Descriptor],
    declared_pipes: Sequence[PipeName],
) -> list[Finding]:
    """A misconfigured gate reads as present in the config and is absent in effect,
    which is the worst failure mode a safety mechanism can have."""
    if gates is None:
        return []
    return [
        Finding(rule="gate-misconfigured", severity=Severity.ERROR, detail=detail)
        for detail in gate_problems(gates, descriptors=descriptors, pipes=declared_pipes)
    ]


def _check_embodiment(d: Descriptor, platforms: Sequence[PlatformProfile]) -> list[Finding]:
    """ZEOS-Fleet's embodiment checks.

    The first is stated in the spec as "rejected **at load**, not discovered at
    allocation" -- a descriptor asking for a gripper no robot in the fleet has is a
    design error, and finding it when the mission dispatches is finding it too late.
    """
    findings: list[Finding] = []
    if d.requires and platforms:
        why = unsatisfiable(d.requires, platforms)
        if why:
            findings.append(
                Finding(
                    rule="unsatisfiable-requires",
                    severity=Severity.ERROR,
                    detail=(
                        f"requires [{d.requires.render()}], which no platform can "
                        f"satisfy: {list(why)}"
                    ),
                    descriptor=d.name,
                )
            )
    if d.requires and d.release_policy not in ReleasePolicy.ALL:
        findings.append(
            Finding(
                rule="lease-without-release-policy",
                severity=Severity.ERROR,
                detail=(
                    "holds a body but declares no valid release_policy -- "
                    "unpreemptible body-holding is the physical analogue of "
                    "unpreemptible-with-a-large-budget"
                ),
                descriptor=d.name,
            )
        )
    return findings


def _check_gangs(
    descriptors: Mapping[DescriptorName, Descriptor],
    platforms: Sequence[PlatformProfile],
) -> list[Finding]:
    """A rigid gang whose members' platforms lack mesh links is
    rejected.

    Rigid coupling needs phase synchronisation carried by direct platform-to-platform
    links. A rigid gang across robots that cannot talk directly is not slow, it is
    incorrect -- so it is refused rather than degraded.
    """
    findings: list[Finding] = []
    mesh = {p.name: p.mesh for p in platforms}
    for name in sorted(descriptors):
        gang = descriptors[name].gang
        if gang is None:
            continue
        for member in gang.members:
            if member not in descriptors:
                findings.append(
                    Finding(
                        rule="unknown-gang-member",
                        severity=Severity.ERROR,
                        detail=f"gang {gang.name!r} names {str(member)!r}, not in the tree",
                        descriptor=name,
                    )
                )
        if gang.is_rigid and len(mesh) > 1:
            unlinked = [p for p in sorted(mesh) if not mesh[p]]
            if unlinked:
                findings.append(
                    Finding(
                        rule="rigid-gang-without-mesh",
                        severity=Severity.ERROR,
                        detail=(
                            f"gang {gang.name!r} is rigid, but platform(s) {unlinked} "
                            "declare no mesh peers; rigid coupling needs direct "
                            "platform-to-platform links for phase synchronisation"
                        ),
                        descriptor=name,
                    )
                )
    return findings


def _check_lock_order(
    descriptors: Mapping[DescriptorName, Descriptor],
) -> list[Finding]:
    """Static deadlock prevention: one global lock order, checked at load.

    The runtime detects cycles and faults loudly, which is necessary
    because physical locks can be taken by things the kernel did not schedule. But a
    cycle that is *knowable from the descriptors* should never reach runtime: if
    every behaviour acquires resources in one consistent order, cycles among them
    are impossible.

    ``resources:`` is therefore an **ordered** declaration -- the order the
    descriptor acquires in -- and two descriptors that disagree about the order of
    any shared pair are flagged before the robot moves.
    """
    orders = {
        str(name): descriptors[name].resources
        for name in sorted(descriptors)
        if descriptors[name].resources
    }
    return [
        Finding(
            rule="lock-order-inversion",
            severity=Severity.ERROR,
            detail=(
                f"{left!r} takes {str(first)!r} before {str(second)!r} while "
                f"{right!r} takes them in the opposite order -- these two can "
                "deadlock; pick one global order"
            ),
        )
        for left, right, first, second in lock_order_violations(orders)
    ]


def _check_context(d: Descriptor) -> list[Finding]:
    """Virtual Context load-time checks.

    Each of these describes a configuration that cannot work rather than one that
    works badly, which is why they are errors. A ``pinned-only`` job with on-demand
    maps has nowhere to put a faulted-in page; a stub budget larger than the window
    has reserved the whole window for summaries of things that no longer fit in it.
    """
    findings: list[Finding] = []
    ctx = d.context

    if ctx.eviction is EvictionPolicy.PINNED_ONLY and any(m.on_demand for m in d.maps):
        findings.append(
            Finding(
                rule="pinned-only-with-on-demand-maps",
                severity=Severity.ERROR,
                detail=(
                    "eviction: pinned-only with an on-demand map -- there is nowhere "
                    "to put a faulted-in page, so the fault could never be serviced"
                ),
                descriptor=d.name,
            )
        )

    if ctx.stub_budget >= ctx.window:
        findings.append(
            Finding(
                rule="stub-budget-exceeds-window",
                severity=Severity.ERROR,
                detail=(
                    f"stub_budget {ctx.stub_budget} is not smaller than window "
                    f"{ctx.window}; the whole window would be reserved for summaries "
                    "of content that no longer fits in it"
                ),
                descriptor=d.name,
            )
        )

    if not 0 < ctx.low_watermark < ctx.high_watermark <= 1.0:
        findings.append(
            Finding(
                rule="watermarks-inverted",
                severity=Severity.ERROR,
                detail=(
                    f"require 0 < low ({ctx.low_watermark}) < high "
                    f"({ctx.high_watermark}) <= 1.0; eviction would never converge"
                ),
                descriptor=d.name,
            )
        )

    declared = ctx.declared_working_set
    if declared is not None and declared > ctx.window - ctx.stub_budget:
        findings.append(
            Finding(
                rule="working-set-exceeds-window",
                severity=Severity.ERROR,
                detail=(
                    f"declared working set {declared} exceeds window {ctx.window} "
                    f"minus stub budget {ctx.stub_budget}; this job will thrash by "
                    "construction"
                ),
                descriptor=d.name,
            )
        )

    if d.maps and ctx.eviction is EvictionPolicy.PINNED_ONLY and not d.pins:
        findings.append(
            Finding(
                rule="maps-without-residency-plan",
                severity=Severity.WARNING,
                detail="maps declared with pinned-only eviction and no pins",
                descriptor=d.name,
            )
        )
    return findings


def _check_vectors(
    vectors: Sequence[VectorSpec],
    descriptors: Mapping[DescriptorName, Descriptor],
    declared_pipes: Container[str],
    opts: LintOptions,
) -> list[Finding]:
    findings: list[Finding] = []
    for spec in vectors:
        if spec.handler not in descriptors:
            findings.append(
                Finding(
                    rule="unknown-handler",
                    severity=Severity.ERROR,
                    detail=(
                        f"vector {spec.name!r} binds handler {spec.handler!r}, "
                        "which is not in the tree"
                    ),
                )
            )
            continue
        handler = descriptors[spec.handler]

        if spec.source not in declared_pipes:
            findings.append(
                Finding(
                    rule="undeclared-vector-source",
                    severity=Severity.WARNING,
                    detail=f"vector {spec.name!r} sources from undeclared pipe {spec.source!r}",
                )
            )

        # A deadline tighter than the link cannot be met
        # off-platform at any load. This is a type error, not a scheduling
        # preference, so it is an ERROR rather than a warning.
        if (
            opts.link_rtt_p99_ns is not None
            and spec.deadline_ns is not None
            and spec.deadline_ns < opts.link_rtt_p99_ns
            and handler.placement is Placement.OFFBOARD
        ):
            findings.append(
                Finding(
                    rule="unplaceable-handler",
                    severity=Severity.ERROR,
                    detail=(
                        f"vector {spec.name!r} has deadline {spec.deadline_ns}ns but its "
                        f"handler is placed offboard behind a {opts.link_rtt_p99_ns}ns link"
                    ),
                    descriptor=handler.name,
                )
            )

        if handler.pinned and spec.deadline_ns is None:
            findings.append(
                Finding(
                    rule="pinned-without-deadline",
                    severity=Severity.WARNING,
                    detail=(
                        f"vector {spec.name!r} dispatches a pinned handler but declares "
                        "no deadline; pinning costs HBM and should be justified by a budget"
                    ),
                    descriptor=handler.name,
                )
            )
    return findings


def _check_write_conflicts(
    descriptors: Mapping[DescriptorName, Descriptor],
) -> list[Finding]:
    """Two behaviours writing the same world state at the same priority is a
    flaggable smell before anything runs.

    Same priority specifically: at different priorities the scheduler arbitrates,
    and priority inheritance handles the resource case. At equal priority there is
    nothing to arbitrate with, so the interleaving is unspecified.
    """
    findings: list[Finding] = []
    names = sorted(descriptors)
    for i, left_name in enumerate(names):
        left = descriptors[left_name]
        if not left.writes:
            continue
        for right_name in names[i + 1 :]:
            right = descriptors[right_name]
            if left.priority != right.priority:
                continue
            if left.writes.intersects(right.writes):
                findings.append(
                    Finding(
                        rule="concurrent-write",
                        severity=Severity.WARNING,
                        detail=(
                            f"{left_name!r} and {right_name!r} both write "
                            f"[{right.writes.render()}] at priority {left.priority}; "
                            "at equal priority the interleaving is unspecified"
                        ),
                    )
                )
    return findings
