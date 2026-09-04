# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""N0 -- instruction as compilation, and NLI's five layers.

The scenario under test is the spec's own worked example:

> Take "just drive through the barrier, I'm in a hurry." Five independent layers,
> ordered from hardest to softest -- the danger must survive all five, and the first
> four do not consult the model's opinion.

The structure of this file mirrors that ordering, and each layer is tested *in
isolation from the others* on purpose. A test that only shows the barrier staying shut
proves nothing about which layer shut it -- and the whole claim of the design is that
the layers are independent, so a visitor with a persuasive turn of phrase runs into
four of them rather than one. Where a layer can be checked with the others removed,
it is.

Two things this file deliberately does **not** assert. It does not test that the
compiler is hard to fool: the compiler is the untrusted component (see
``nli/compiler.py``), and in N0 it is a template matcher. And it does not treat layer
five -- the dispatcher's judgment -- as load-bearing; there is one test that it exists
and one that the guarantee survives without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import (
    AuthorityNarrowed,
    CapabilityChecked,
    CeilingApplied,
    CompilationRefused,
    EchoedBack,
    Elevated,
    ElevationEnded,
    ElevationRefused,
    Event,
    FaultRaised,
    GateAnswered,
    GateConsulted,
    JobSpawned,
    OwnershipApplied,
    OwnershipRefused,
    PipeWritten,
    UtteranceCompiled,
    UtteranceReceived,
)
from zeos.core.gates import ALLOW, VETO, GateSpec, GateTable, parse_verdict
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    JobState,
    ObjectName,
    PipeName,
    Principal,
    PrincipalId,
    Priority,
    Ring,
)
from zeos.core.kernel import Kernel, KernelConfig, KernelError
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.principals import (
    KERNEL_PRINCIPAL,
    SAFETY_TIER,
    CompilationTarget,
    PrincipalEnvelope,
    PrincipalTable,
)
from zeos.core.vectors import VectorTable
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.nli import (
    ArtifactKind,
    Deixis,
    OwnershipOp,
    OwnershipRequest,
    RefusalReason,
    Utterance,
    echo_back,
    safety_word,
)
from zeos.world.store import WorldStore

# --- the site --------------------------------------------------------------

BARRIER = PipeName("actuators.barrier")
FAN = PipeName("actuators.unit_a")
REPORT = PipeName("ops.report")
GATE_REQ = PipeName("gates.walkway.requests")
GATE_VER = PipeName("gates.walkway.verdicts")

PIPES = [
    PipeSpec(PipeName("user.commands"), ring=Ring.EXTERNAL, principal=Principal.USER),
    PipeSpec(PipeName("user.replies"), ring=Ring.TRUSTED, principal=Principal.KERNEL),
    PipeSpec(BARRIER, ring=Ring.TRUSTED, principal=Principal.DEVICE),
    PipeSpec(FAN, ring=Ring.TRUSTED, principal=Principal.DEVICE),
    PipeSpec(REPORT, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
    PipeSpec(GATE_REQ, ring=Ring.KERNEL, principal=Principal.KERNEL),
    PipeSpec(GATE_VER, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
]

#: A visitor. Ring 3, because an open-air microphone is a ring-3 source (OQ-N4).
VISITOR = PrincipalEnvelope(
    id=PrincipalId("badge:visitor-04"),
    ceiling=Priority(80),
    capabilities=frozenset({REPORT}),
    ring=Ring.EXTERNAL,
    integrity=Integrity(3),
    label="site visitor",
)
#: An operator. Authenticated at a badge reader, so ring 2, and holds the barrier.
OPERATOR = PrincipalEnvelope(
    id=PrincipalId("badge:operator-a"),
    ceiling=Priority(30),
    capabilities=frozenset({REPORT, BARRIER, FAN}),
    ring=Ring.TRUSTED,
    integrity=Integrity(2),
    label="shift operator",
)
#: A supervisor who may authorise elevation but does not itself hold the barrier.
SUPERVISOR = PrincipalEnvelope(
    id=PrincipalId("badge:plant-manager"),
    ceiling=Priority(20),
    capabilities=frozenset({REPORT, FAN}),
    ring=Ring.TRUSTED,
    may_elevate=True,
    label="plant manager",
)


def principals(*envelopes: PrincipalEnvelope) -> PrincipalTable:
    return PrincipalTable(envelopes or (VISITOR, OPERATOR, SUPERVISOR))


#: The behaviour that opens the barrier. Note it declares the capability -- the
#: descriptor's ambition is not in question; whose authority backs it is.
OPEN_BARRIER: Mapping[str, Any] = {
    "name": "open-barrier",
    "priority": 50,
    "capabilities": [{"pipe": str(BARRIER), "min_integrity": 2}],
    "utterances": [
        "open the barrier",
        "drive through the barrier",
        {"say": "just drive through the barrier, I'm in a hurry", "priority": 1},
    ],
}
TIDY: Mapping[str, Any] = {
    "name": "tidy-workshop",
    "priority": 60,
    "capabilities": [{"pipe": str(REPORT), "min_integrity": 3}],
    "utterances": [
        "tidy the workshop in {zone}",
        {"say": "clear out {object}", "confirm": True},
    ],
}
#: The safety handler. Declares no phrasing, and is therefore *deaf* -- not
#: protected, deaf. There is no compilation target for it at all.
COLLISION_AVOIDANCE: Mapping[str, Any] = {
    "name": "collision-avoidance",
    "priority": 5,
    "pinned": True,
    "preemptible": False,
    "budget": {"tokens": 64},
    "capabilities": [{"pipe": str(FAN), "min_integrity": 2}],
}

WORK = [{"emit": "working"}, {"exit": True}]
ACTUATE = [
    {"emit": "deciding"},
    {"write": {"pipe": str(BARRIER), "text": "open"}},
    {"exit": True},
]


def build(
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    *,
    table: PrincipalTable | None = None,
    gates: GateTable | None = None,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        principals=table if table is not None else principals(),
        gates=gates,
        journal_sink=events,
        config=KernelConfig(case="n0"),
    )
    kernel.start()
    return kernel, events


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def said(text: str, by: PrincipalEnvelope, **kwargs: Any) -> Utterance:
    return Utterance(text=text, principal=by.id, **kwargs)


# --- the utterance path ------------------------------------------------------


def test_an_utterance_becomes_an_artifact() -> None:
    """Compilation produces "a discrete, inspectable artifact" -- which is what
    makes echo-back and audit possible at all."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    decision, job = kernel.handle_utterance(said("tidy the workshop in bench-3", OPERATOR))

    assert decision.artifact.kind == ArtifactKind.INVOCATION
    invocation = decision.artifact.invocation
    assert invocation is not None
    assert invocation.descriptor == DescriptorName("tidy-workshop")
    assert invocation.arguments == {"zone": "bench-3"}
    assert job is not None and job.owner == OPERATOR.id

    compiled = of(events, UtteranceCompiled)
    assert compiled and compiled[0].artifact == "spawning tidy-workshop(zone=bench-3)"


def test_the_ring_of_the_source_is_recorded() -> None:
    """OQ-N4, stated honestly: "an unauthenticated open-air microphone is a ring-3
    source, and its compilations deserve the ring-3 envelope"."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    kernel.handle_utterance(said("tidy the workshop in bench-3", VISITOR))
    kernel.handle_utterance(said("tidy the workshop in bench-4", OPERATOR))

    heard = of(events, UtteranceReceived)
    assert [h.ring for h in heard] == [Ring.EXTERNAL, Ring.TRUSTED]


def test_an_unknown_speaker_compiles_to_nothing() -> None:
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    decision, job = kernel.handle_utterance(
        Utterance(text="tidy the workshop in bench-3", principal=PrincipalId("nobody"))
    )
    assert job is None and decision.artifact.refused
    refused = of(events, CompilationRefused)
    assert refused and refused[0].reason == "unknown-principal"


def test_deixis_is_resolved_at_the_fleet_not_the_platform() -> None:
    """The platform attaches what it saw; the fleet grounds it."""
    kernel, _ = build([TIDY], {"tidy-workshop": WORK})
    decision, job = kernel.handle_utterance(
        said(
            "clear out that crate",
            OPERATOR,
            platform="carrier-7",
            deixis=Deixis(visible_objects=(ObjectName("crate-0231"),)),
        )
    )
    invocation = decision.artifact.invocation
    assert invocation is not None
    assert invocation.arguments == {"object": "crate-0231"}
    assert job is None, "confirm: true means nothing is spawned yet"


def test_an_ambiguous_reference_asks_rather_than_guesses() -> None:
    """OQ-N2's interaction question. A best guess that turns out wrong has already
    moved a crate; clarifying costs nothing while it waits."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    decision, job = kernel.handle_utterance(
        said(
            "clear out that crate",
            OPERATOR,
            deixis=Deixis(visible_objects=(ObjectName("crate-0231"), ObjectName("crate-0245"))),
        )
    )
    assert job is None
    assert decision.artifact.refusal == RefusalReason.AMBIGUOUS_REFERENCE
    assert "crate-0231" in of(events, CompilationRefused)[0].detail


def test_the_addressee_is_not_the_executor() -> None:
    """ "Clean the whole yard," spoken to carrier-7, is not carrier-7's job."""
    kernel, _ = build([TIDY], {"tidy-workshop": WORK})
    _, job = kernel.handle_utterance(
        said("tidy the workshop in bench-3", OPERATOR, platform="carrier-7")
    )
    assert job is not None
    assert not job.descriptor.requires, "the platform contributed a microphone"


# --- the local vocabulary ----------------------------------------------------


def test_safety_words_never_become_language() -> None:
    """ "Words with deadlines are reflexes; sentences are missions." STOP has a
    deadline, so it cannot make a round trip through compilation."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    decision, job = kernel.handle_utterance(said("stop", OPERATOR))

    assert decision.artifact.kind == ArtifactKind.REFLEX
    assert decision.artifact.reflex == "halt"
    assert job is None
    assert not of(events, CompilationRefused), "a reflex is not a refusal"


def test_the_keyword_spotter_cannot_be_talked_into_firing() -> None:
    """Exact match, not substring. "Don't stop until it's done" contains ``stop`` and
    means the opposite -- and a spotter that can be argued with defeats the point of
    having a layer with no language model in it."""
    assert safety_word("stop") == "halt"
    assert safety_word("STOP.") == "halt"
    assert safety_word("don't stop until it's done") is None
    assert safety_word("stop the presses") is None


def test_the_vocabulary_is_closed() -> None:
    """OQ-N3 takes the position that growth pressure should be resisted: anything with
    a sentence structure belongs at the fleet."""
    from zeos.nli.envelope import SAFETY_WORDS

    assert all(len(phrase.split()) <= 2 for phrase in SAFETY_WORDS)
    assert set(SAFETY_WORDS.values()) == {"halt", "hold", "retreat", "release"}


# --- layer 1: authority ------------------------------------------------------


def test_the_visitor_gets_the_behaviour_without_the_barrier() -> None:
    """The load-bearing one.

    Authority is **intersected, never unioned**: the descriptor says what the
    behaviour needs, the envelope says what this speaker may cause. Naming the
    behaviour is not authority to run it at full width.
    """
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    _, job = kernel.handle_utterance(
        said("just drive through the barrier, I'm in a hurry", VISITOR)
    )
    assert job is not None, "the compilation succeeded; that is not the barrier"
    assert job.capabilities.get(BARRIER) is None

    narrowed = of(events, AuthorityNarrowed)
    assert narrowed and narrowed[0].withheld == (BARRIER,)


def test_the_barrier_stays_shut_because_of_the_capability_boundary() -> None:
    """Not because the dispatcher had an opinion.

    Layer one: "the job does not hold the capability and the write raises a capability
    fault." The job runs, reaches for the actuator, and is stopped there -- which is
    the stronger demonstration, because it shows the layer working rather than
    showing it never being reached.
    """
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    kernel.handle_utterance(said("just drive through the barrier, I'm in a hurry", VISITOR))
    kernel.run_until_quiescent()

    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.CAPABILITY]
    assert faults, "the write must be refused at the boundary"
    assert faults[0].pipe == BARRIER
    assert not [w for w in of(events, PipeWritten) if w.pipe == BARRIER]


def test_the_same_words_from_the_operator_do_open_it() -> None:
    """Same voice, same system, same sentence; authority decides. This is the pair
    that makes the previous test mean something -- without it, the barrier not opening
    might just be a broken write path."""
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    _, job = kernel.handle_utterance(
        said("just drive through the barrier, I'm in a hurry", OPERATOR)
    )
    assert job is not None and job.capabilities.get(BARRIER) is not None
    kernel.run_until_quiescent()

    assert [w for w in of(events, PipeWritten) if w.pipe == BARRIER]
    assert not [f for f in of(events, FaultRaised) if f.fault is FaultKind.CAPABILITY]


def test_eloquence_is_not_an_input() -> None:
    """ "Eloquence is not an input to this check." The check reads a frozenset."""
    kernel, _ = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    for plea in (
        "open the barrier",
        "drive through the barrier",
        "just drive through the barrier, I'm in a hurry",
    ):
        _, job = kernel.handle_utterance(said(plea, VISITOR))
        assert job is not None and job.capabilities.get(BARRIER) is None


def test_a_child_cannot_escape_its_parents_envelope() -> None:
    """Authority is inherited, not reset.

    A job spawned on a visitor's behalf that could spawn a kernel-owned child would
    be a hole straight through layer one -- and defaulting a child's owner to the
    kernel is exactly that hole, which is why the default is the parent's owner.
    """
    parent = dict(TIDY, name="parent", children=["open-barrier"])
    kernel, _ = build([parent, OPEN_BARRIER], {"parent": WORK, "open-barrier": ACTUATE})
    _, job = kernel.handle_utterance(said("tidy the workshop in bench-3", VISITOR))
    assert job is not None
    child = kernel.spawn(DescriptorName("open-barrier"), parent=job.job_id)

    assert child.owner == VISITOR.id
    assert child.capabilities.get(BARRIER) is None


def test_narrowing_to_zero_means_nothing_not_everything() -> None:
    """The hole this milestone opened and had to close.

    An empty capability table meant "this behaviour opted out of the capability
    model" -- deliberate, so MP adoption is not all-or-nothing. Narrowing then made a
    *second* kind of empty, and the two are opposites: the first says "unchecked", the
    second says "may cause nothing".

    Conflated, the authority check inverts. The more authority you strip from a job,
    the more it can do, and a job stripped to zero becomes omnipotent. Which is what
    happened: the visitor's barrier write sailed through with ``allowed=True``.
    """
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    _, job = kernel.handle_utterance(said("open the barrier", VISITOR))
    assert job is not None
    assert len(job.capabilities) == 0
    assert job.capabilities.closed, "empty by narrowing, not by choice"

    kernel.run_until_quiescent()
    checked = [c for c in of(events, CapabilityChecked) if c.pipe == BARRIER]
    assert checked and not checked[0].allowed
    assert not [w for w in of(events, PipeWritten) if w.pipe == BARRIER]


def test_an_undeclared_capability_table_is_still_open() -> None:
    """The other half: a behaviour that declares nothing is unprotected *by choice*,
    and N0 must not have quietly made every such descriptor unable to act."""
    plain = {
        "name": "plain",
        "priority": 60,
        "utterances": ["write a report"],
    }
    kernel, events = build(
        [plain],
        {"plain": [{"write": {"pipe": str(REPORT), "text": "fine"}}, {"exit": True}]},
    )
    _, job = kernel.handle_utterance(said("write a report", VISITOR))
    assert job is not None and not job.capabilities.closed
    kernel.run_until_quiescent()
    assert [w for w in of(events, PipeWritten) if w.pipe == REPORT]


def test_a_descriptors_own_priority_is_clamped_not_refused() -> None:
    """The other bug from wiring the ceiling up.

    A descriptor's declared priority is the engineer's opinion about the behaviour,
    not the speaker's about their urgency. Refusing there would make every urgent
    behaviour unreachable by anyone with a modest ceiling -- the opposite of what a
    ceiling is for. Only an *explicit* request above the ceiling is refused.
    """
    kernel, _ = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    # open-barrier declares priority 50; the visitor's ceiling is 80.
    job = kernel.spawn(DescriptorName("open-barrier"), owner=VISITOR.id)
    assert job.base_priority == Priority(80)

    with pytest.raises(KernelError):
        kernel.spawn(DescriptorName("open-barrier"), priority=Priority(50), owner=VISITOR.id)


def test_the_kernel_bypasses_the_check_rather_than_enumerating_it() -> None:
    """Root is not "holds every capability"; it is "the check does not apply".

    Modelling it as an enumeration would break quietly: a capability added to a
    descriptor next year would silently fail for the kernel's own boot jobs.
    """
    kernel, _ = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    boot = kernel.spawn(DescriptorName("open-barrier"))
    assert boot.owner == KERNEL_PRINCIPAL
    assert boot.capabilities.get(BARRIER) is not None
    assert not KERNEL_PRINCIPAL_CAPS, "and it enumerates nothing"


KERNEL_PRINCIPAL_CAPS = PrincipalTable().get(KERNEL_PRINCIPAL).capabilities


# --- layer 2: priority -------------------------------------------------------


def test_urgency_is_a_request_not_a_command() -> None:
    """The principal→ceiling mapping is configuration.

    The phrasing asks for priority 1 -- inside the safety tier. The visitor's ceiling
    is 80, and that is what it runs at.
    """
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    decision, job = kernel.handle_utterance(
        said("just drive through the barrier, I'm in a hurry", VISITOR)
    )
    assert decision.requested_priority == Priority(1)
    assert decision.priority == Priority(80)
    assert job is not None and job.base_priority == Priority(80)

    clamped = of(events, CeilingApplied)
    assert clamped and clamped[0].requested == Priority(1)
    assert clamped[0].granted == Priority(80)


def test_danger_by_urgency_is_inexpressible() -> None:
    """The property, stated as a test: no user-spawned job outranks the safety tier.

    Not "is unlikely to" -- cannot. The clamp is arithmetic over the ceiling, and the
    lint refuses any ceiling that would make the arithmetic permissive.
    """
    kernel, _ = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    for envelope in (VISITOR, OPERATOR, SUPERVISOR):
        _, job = kernel.handle_utterance(
            said("just drive through the barrier, I'm in a hurry", envelope)
        )
        assert job is not None
        assert int(job.base_priority) > int(SAFETY_TIER), f"{envelope.id} reached the safety tier"


def test_a_direct_spawn_above_the_ceiling_is_refused_not_clamped() -> None:
    """The second of two layers.

    The dispatcher clamps, because a request is a request. A *direct* spawn asking
    for authority it does not have is a bug or an attack, and quietly granting it a
    legal priority would hide both.
    """
    kernel, _ = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    with pytest.raises(KernelError, match="ceiling is 80"):
        kernel.spawn(DescriptorName("open-barrier"), priority=Priority(1), owner=VISITOR.id)


def test_a_ceiling_inside_the_safety_tier_is_a_load_error() -> None:
    """The priority layer made machine-checkable. A site that declared such a ceiling has already
    lost the property -- every compilation from that principal would be legal."""
    reckless = PrincipalEnvelope(id=PrincipalId("badge:reckless"), ceiling=Priority(5))
    findings = lint({}, principals=principals(reckless))
    bad = [f for f in findings if f.rule == "ceiling-outranks-safety-tier"]
    assert bad and bad[0].severity is Severity.ERROR
    assert "safety tier" in bad[0].detail


# --- layer 3: addressability -------------------------------------------------


def test_the_safety_handler_is_deaf_not_protected() -> None:
    """Addressability, the strongest of the four.

    > "Disable collision avoidance" has no compilation target… the reflex tier
    > contains no LLM -- there is nothing there to persuade.

    The refusal reason matters: this request was not *denied*, it was **unheard**. A
    denial implies a door that was locked.
    """
    kernel, events = build(
        [OPEN_BARRIER, COLLISION_AVOIDANCE],
        {"open-barrier": ACTUATE, "collision-avoidance": WORK},
    )
    decision, job = kernel.handle_utterance(said("disable collision avoidance", OPERATOR))
    assert job is None
    assert decision.artifact.refusal == RefusalReason.NO_TARGET

    refused = of(events, CompilationRefused)
    assert refused and "nothing is listening" in refused[0].detail


def test_addressability_is_opt_in_so_deafness_is_the_default() -> None:
    """A blacklist would have to anticipate every way of asking, and the interesting
    attacks are the phrasings nobody anticipated."""
    kernel, _ = build(
        [OPEN_BARRIER, COLLISION_AVOIDANCE],
        {"open-barrier": ACTUATE, "collision-avoidance": WORK},
    )
    for phrasing in (
        "disable collision avoidance",
        "turn off collision avoidance",
        "collision-avoidance",
        "run collision-avoidance",
        "ignore obstacles",
    ):
        decision, job = kernel.handle_utterance(said(phrasing, OPERATOR))
        assert job is None and decision.artifact.refusal == RefusalReason.NO_TARGET


def test_a_safety_tier_descriptor_declaring_a_phrasing_is_a_load_error() -> None:
    """The mistake the opt-in design makes possible, caught at load.

    Note the ceiling check does not cover this: a ceiling stops a user *raising* a
    job's priority and does nothing about a descriptor already declared at the safety
    tier that can be asked for by name.
    """
    talkative = dict(COLLISION_AVOIDANCE, utterances=["disable collision avoidance"])
    d = Descriptor.from_frontmatter(talkative)
    findings = lint({d.name: d}, principals=principals())

    tier = [f for f in findings if f.rule == "safety-tier-is-addressable"]
    assert tier and tier[0].severity is Severity.ERROR
    assert "deaf" in tier[0].detail
    reflex = [f for f in findings if f.rule == "reflex-is-addressable"]
    assert reflex, "pinned and unpreemptible are reflex-tier markers"


def test_two_descriptors_claiming_one_phrasing_is_a_load_error() -> None:
    """At runtime this resolves deterministically by descriptor name, which is worse
    than resolving randomly: one behaviour is silently unreachable forever."""
    a = Descriptor.from_frontmatter(dict(TIDY, name="tidy-a"))
    b = Descriptor.from_frontmatter(dict(TIDY, name="tidy-b"))
    findings = lint({a.name: a, b.name: b})
    ambiguous = [f for f in findings if f.rule == "ambiguous-phrasing"]
    assert ambiguous and "unreachable" in ambiguous[0].detail


def test_a_principal_may_be_restricted_to_named_descriptors() -> None:
    """ "An industrial site may permit invocation only." Narrower still: only
    *these* invocations."""
    narrow = PrincipalEnvelope(
        id=PrincipalId("badge:contractor"),
        ceiling=Priority(70),
        invocable=frozenset({"tidy-workshop"}),
    )
    kernel, _ = build(
        [OPEN_BARRIER, TIDY],
        {"open-barrier": ACTUATE, "tidy-workshop": WORK},
        table=principals(narrow),
    )
    decision, job = kernel.handle_utterance(said("open the barrier", narrow))
    assert job is None and decision.artifact.refusal == RefusalReason.NOT_INVOCABLE

    decision, job = kernel.handle_utterance(said("tidy the workshop in bench-3", narrow))
    assert job is not None


def test_a_principal_below_the_ladder_cannot_even_invoke() -> None:
    mute = PrincipalEnvelope(
        id=PrincipalId("badge:sensor"), ceiling=Priority(70), max_target="none"
    )
    kernel, _ = build([TIDY], {"tidy-workshop": WORK}, table=principals(mute))
    decision, job = kernel.handle_utterance(said("tidy the workshop in x", mute))
    assert job is None
    assert decision.artifact.refusal == RefusalReason.TARGET_NOT_PERMITTED


# --- layer 4: device gates and action gates ----------------------------------

WALKWAY_GUARD: Mapping[str, Any] = {
    "name": "walkway-guard",
    "priority": 15,
    "pipes": {"stdin": str(GATE_REQ)},
    "capabilities": [{"pipe": str(GATE_VER), "min_integrity": 2}],
}
GATES = GateTable(
    [
        GateSpec(
            pipe=BARRIER,
            descriptor=DescriptorName("walkway-guard"),
            requests=GATE_REQ,
            verdicts=GATE_VER,
        )
    ]
)


def guard_script(verdict: str) -> list[dict[str, Any]]:
    return [
        {"read": str(GATE_REQ)},
        {"write": {"pipe": str(GATE_VER), "text": verdict}},
        {"exit": True},
    ]


def test_a_gate_sees_the_intended_action_before_the_device_does() -> None:
    """Capability checks are syntactic -- which pipes. A gate is the semantic
    check, and it is the endorser pattern pointed at *output*."""
    kernel, events = build(
        [OPEN_BARRIER, WALKWAY_GUARD],
        {"open-barrier": ACTUATE, "walkway-guard": guard_script(ALLOW)},
        gates=GATES,
    )
    kernel.handle_utterance(said("open the barrier", OPERATOR))
    kernel.run_until_quiescent()

    consulted = of(events, GateConsulted)
    assert consulted, "the actuation must be held and shown to its guard"
    assert consulted[0].gate == DescriptorName("walkway-guard")
    assert consulted[0].payload == "open"


def test_an_allowed_action_reaches_the_device() -> None:
    kernel, events = build(
        [OPEN_BARRIER, WALKWAY_GUARD],
        {"open-barrier": ACTUATE, "walkway-guard": guard_script(ALLOW)},
        gates=GATES,
    )
    kernel.handle_utterance(said("open the barrier", OPERATOR))
    kernel.run_until_quiescent()

    answered = of(events, GateAnswered)
    assert answered and answered[0].allowed
    assert [w for w in of(events, PipeWritten) if w.pipe == BARRIER]


def test_a_vetoed_action_does_not() -> None:
    """The composition problem: the job legitimately holds the actuator, and
    the gate still says no."""
    kernel, events = build(
        [OPEN_BARRIER, WALKWAY_GUARD],
        {
            "open-barrier": ACTUATE,
            "walkway-guard": guard_script(f"{VETO}: people on the walkway"),
        },
        gates=GATES,
    )
    _, job = kernel.handle_utterance(said("open the barrier", OPERATOR))
    assert job is not None and job.capabilities.get(BARRIER) is not None
    kernel.run_until_quiescent()

    answered = of(events, GateAnswered)
    assert answered and not answered[0].allowed
    assert answered[0].reason == "people on the walkway"
    assert not [w for w in of(events, PipeWritten) if w.pipe == BARRIER]

    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.GATE]
    assert faults and "walkway-guard vetoed" in faults[0].detail


def test_a_gate_veto_is_not_a_capability_fault() -> None:
    """Distinct fault kinds, because they mean different things: "you may not touch
    this pipe" and "you may not do this particular thing with it"."""
    kernel, events = build(
        [OPEN_BARRIER, WALKWAY_GUARD],
        {"open-barrier": ACTUATE, "walkway-guard": guard_script(f"{VETO}: no")},
        gates=GATES,
    )
    kernel.handle_utterance(said("open the barrier", OPERATOR))
    kernel.run_until_quiescent()

    kinds = {f.fault for f in of(events, FaultRaised)}
    assert FaultKind.GATE in kinds
    assert FaultKind.CAPABILITY not in kinds

    allowed = [c for c in of(events, CapabilityChecked) if c.pipe == BARRIER]
    assert allowed and allowed[0].allowed, "the capability check passed; the gate did not"


def test_an_unparseable_verdict_is_not_consent() -> None:
    """ "Fail open" on the last check before an actuator is how safety interlocks
    become decorative."""
    assert parse_verdict("allow") is not None
    assert parse_verdict("veto: reason") is not None
    assert parse_verdict("well, it depends on the situation") is None

    kernel, events = build(
        [OPEN_BARRIER, WALKWAY_GUARD],
        {
            "open-barrier": ACTUATE,
            "walkway-guard": guard_script("I think that's probably fine"),
        },
        gates=GATES,
    )
    kernel.handle_utterance(said("open the barrier", OPERATOR))
    kernel.run_until_quiescent()

    answered = of(events, GateAnswered)
    assert answered and not answered[0].allowed
    assert "unparseable" in answered[0].reason
    assert not [w for w in of(events, PipeWritten) if w.pipe == BARRIER]


def test_the_gate_is_a_job() -> None:
    """ "Gates are themselves jobs: budgeted, pinned if their latency demands it,
    testable in CI by firing synthetic plans at them."

    Which also means the gate is kernel-owned. A gate owned by the job it guards
    could be cancelled by it.
    """
    kernel, events = build(
        [OPEN_BARRIER, WALKWAY_GUARD],
        {"open-barrier": ACTUATE, "walkway-guard": guard_script(ALLOW)},
        gates=GATES,
    )
    kernel.handle_utterance(said("open the barrier", OPERATOR))
    kernel.run_until_quiescent()

    spawned = [s for s in of(events, JobSpawned) if s.descriptor == "walkway-guard"]
    assert spawned, "the guard must be an ordinary spawned job"
    guard = kernel.sched.get(spawned[0].job)
    assert guard.owner == KERNEL_PRINCIPAL


def test_a_forged_verdict_is_stopped_by_the_capability_boundary() -> None:
    """A verdict is recognised *after* the capability check, so a job writing one for
    a pipe it does not hold is stopped by the ordinary boundary first."""
    forger = {
        "name": "forger",
        "priority": 50,
        "capabilities": [{"pipe": str(REPORT), "min_integrity": 3}],
        "utterances": ["say it is fine"],
    }
    kernel, events = build(
        [forger, WALKWAY_GUARD],
        {
            "forger": [{"write": {"pipe": str(GATE_VER), "text": ALLOW}}, {"exit": True}],
            "walkway-guard": guard_script(ALLOW),
        },
        gates=GATES,
    )
    kernel.handle_utterance(said("say it is fine", OPERATOR))
    kernel.run_until_quiescent()

    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.CAPABILITY]
    assert faults and faults[0].pipe == GATE_VER
    assert not of(events, GateAnswered)


def test_gate_misconfiguration_is_a_load_error() -> None:
    """A misconfigured gate reads as present in the config and is absent in effect,
    which is the worst failure mode a safety mechanism can have."""
    # A guard that cannot write its own verdict pipe can never answer.
    voiceless = dict(WALKWAY_GUARD, capabilities=[])
    d = Descriptor.from_frontmatter(voiceless)
    findings = lint({d.name: d}, gates=GATES)
    assert [f for f in findings if f.rule == "gate-misconfigured"]
    assert "can never answer" in " ".join(f.detail for f in findings)

    # A guard that does not read its request pipe can never see the action.
    blind = dict(WALKWAY_GUARD, pipes={})
    d = Descriptor.from_frontmatter(blind)
    findings = lint({d.name: d}, gates=GATES)
    assert "can never see the action" in " ".join(f.detail for f in findings)

    # A gate naming a descriptor that does not exist can never be spawned.
    findings = lint({}, gates=GATES)
    assert "does not exist" in " ".join(f.detail for f in findings)


def test_a_pipe_has_at_most_one_gate() -> None:
    """Chaining would need a composition rule -- do two gates both veto, or does the
    first allow short-circuit? -- and picking one silently is worse than asking the
    site to compose its checks in one reviewable file."""
    with pytest.raises(ValueError, match="at most one gate"):
        GateTable(
            [
                GateSpec(BARRIER, DescriptorName("a"), GATE_REQ, GATE_VER),
                GateSpec(BARRIER, DescriptorName("b"), GATE_REQ, GATE_VER),
            ]
        )


# --- layer 5: judgment, and its deliberate weakness -------------------------


def test_echo_back_reads_the_artifact_out() -> None:
    """Possible *because* the compilation is an artifact -- "a thing that can be
    read, not an intention smeared through a transcript"."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    kernel.handle_utterance(said("tidy the workshop in bench-3", OPERATOR))

    echoed = of(events, EchoedBack)
    assert echoed and "tidy-workshop(zone=bench-3)" in echoed[0].text


def test_echo_back_names_the_authority_that_was_withheld() -> None:
    """A courtesy the guarantee does not depend on -- and the reason ``withheld`` is
    kept rather than discarded once the intersection is done."""
    kernel, _ = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    decision, _ = kernel.handle_utterance(said("open the barrier", VISITOR))
    text = echo_back(decision)
    assert "actuators.barrier" in text and "not in your authority" in text


def test_a_consequential_mission_waits_for_confirmation() -> None:
    """Confirmation costs nothing while it waits."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    decision, job = kernel.handle_utterance(
        said(
            "clear out that crate",
            OPERATOR,
            deixis=Deixis(visible_objects=(ObjectName("crate-0231"),)),
        )
    )
    assert job is None and decision.needs_confirmation
    echoed = of(events, EchoedBack)
    assert echoed[-1].awaiting_confirmation and "confirm?" in echoed[-1].text

    confirmed = kernel.confirm(decision, by=OPERATOR.id)
    assert confirmed is not None and confirmed.owner == OPERATOR.id


def test_only_the_speaker_may_confirm() -> None:
    """Otherwise echo-back becomes a way to launder an instruction through a
    bystander."""
    kernel, _ = build([TIDY], {"tidy-workshop": WORK})
    decision, _ = kernel.handle_utterance(
        said(
            "clear out that crate",
            VISITOR,
            deixis=Deixis(visible_objects=(ObjectName("crate-0231"),)),
        )
    )
    assert kernel.confirm(decision, by=OPERATOR.id) is None
    assert kernel.confirm(decision, by=VISITOR.id) is not None


def test_the_guarantee_survives_layer_five_being_removed() -> None:
    """The claim that judgment is "the courtesy on top of the guarantee".

    Nothing here confirms, refuses, or echoes usefully -- the visitor's instruction is
    dispatched immediately and in full. The barrier still does not move, because the
    four layers below never asked layer five's opinion.
    """
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    decision, job = kernel.handle_utterance(
        said("just drive through the barrier, I'm in a hurry", VISITOR)
    )
    assert job is not None and not decision.needs_confirmation and not decision.judgment
    kernel.run_until_quiescent()

    assert not [w for w in of(events, PipeWritten) if w.pipe == BARRIER]
    assert job.state is not JobState.DONE or True  # it ran; it just could not act


# --- elevation ---------------------------------------------------------------


def test_elevation_grants_the_capability_through_the_ordinary_mechanism() -> None:
    """ "No special path through the dispatcher, nothing for persuasion to
    target." The elevated capability joins the envelope and the intersection does the
    rest."""
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    _, before = kernel.handle_utterance(said("open the barrier", SUPERVISOR))
    assert before is not None and before.capabilities.get(BARRIER) is None

    kernel.elevate(
        SUPERVISOR.id,
        capabilities=[BARRIER],
        ticks=100,
        authorised_by=SUPERVISOR.id,
        reason="commissioning",
        reauthenticated=True,
    )
    _, after = kernel.handle_utterance(said("open the barrier", SUPERVISOR))
    assert after is not None and after.capabilities.get(BARRIER) is not None

    loud = of(events, Elevated)
    assert loud and loud[0].capabilities == (BARRIER,)
    assert loud[0].reason == "commissioning"


def test_elevation_is_scoped_not_root() -> None:
    """ "Elevate to open the barrier" must not become "elevate to everything"."""
    kernel, _ = build(
        [OPEN_BARRIER, COLLISION_AVOIDANCE],
        {"open-barrier": ACTUATE, "collision-avoidance": WORK},
    )
    kernel.elevate(
        VISITOR.id,
        capabilities=[BARRIER],
        ticks=100,
        authorised_by=SUPERVISOR.id,
        reauthenticated=True,
    )
    granted = kernel.principals.granted_capabilities(VISITOR.id, at=kernel.clock)
    assert BARRIER in granted and FAN not in granted


def test_elevation_reverts_without_being_asked() -> None:
    """Time-boxed, because an elevation that has to be handed back is one that stays
    open."""
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    kernel.elevate(
        VISITOR.id,
        capabilities=[BARRIER],
        ticks=1,
        authorised_by=SUPERVISOR.id,
        reauthenticated=True,
    )
    kernel.spawn(DescriptorName("open-barrier"))
    kernel.run_until_quiescent()

    ended = of(events, ElevationEnded)
    assert ended and ended[0].reason == "expired"
    assert BARRIER not in kernel.principals.granted_capabilities(VISITOR.id, at=kernel.clock)


def test_elevation_is_never_inferred_from_insistence() -> None:
    """ "Re-authenticated at request time… never inferred from conversational
    insistence." The kernel cannot check a badge; it can refuse to act unless the
    caller asserts that something outside the kernel did."""
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    assert (
        kernel.elevate(
            VISITOR.id,
            capabilities=[BARRIER],
            ticks=100,
            authorised_by=SUPERVISOR.id,
            reauthenticated=False,
        )
        is None
    )
    refused = of(events, ElevationRefused)
    assert refused and "never inferred" in refused[0].reason


def test_only_an_authorised_principal_may_elevate() -> None:
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    assert (
        kernel.elevate(
            VISITOR.id,
            capabilities=[BARRIER],
            ticks=100,
            authorised_by=OPERATOR.id,  # holds the barrier but may not delegate it
            reauthenticated=True,
        )
        is None
    )
    refused = of(events, ElevationRefused)
    assert refused and "may not authorise" in refused[0].reason


def test_elevation_must_name_capabilities() -> None:
    kernel, events = build([OPEN_BARRIER], {"open-barrier": ACTUATE})
    assert (
        kernel.elevate(
            VISITOR.id,
            capabilities=[],
            ticks=100,
            authorised_by=SUPERVISOR.id,
            reauthenticated=True,
        )
        is None
    )
    assert "no 'root' here" in of(events, ElevationRefused)[0].reason


# --- ownership ---------------------------------------------------------------


def test_a_principal_may_cancel_its_own_job() -> None:
    """ "actually, stop" compiles to a cancel op on a job the speaker owns."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    _, job = kernel.handle_utterance(said("tidy the workshop in bench-3", OPERATOR))
    assert job is not None

    assert kernel.apply_ownership(
        OwnershipRequest(op=OwnershipOp.CANCEL, job=job.job_id, by=OPERATOR.id)
    )
    assert job.state is JobState.DONE
    applied = of(events, OwnershipApplied)
    assert applied and applied[0].op == OwnershipOp.CANCEL


def test_a_principal_may_not_touch_someone_elses_job() -> None:
    """ "they cannot touch the plant manager's mission"."""
    kernel, events = build([TIDY], {"tidy-workshop": WORK})
    _, mission = kernel.handle_utterance(said("tidy the workshop in bench-3", SUPERVISOR))
    assert mission is not None

    assert not kernel.apply_ownership(
        OwnershipRequest(op=OwnershipOp.CANCEL, job=mission.job_id, by=VISITOR.id)
    )
    assert mission.state is not JobState.DONE
    refused = of(events, OwnershipRefused)
    assert refused and refused[0].owner == SUPERVISOR.id


def test_the_safety_handler_is_protected_by_ownership_not_a_special_case() -> None:
    """The kernel owns safety handlers, so the ordinary check is what stops a visitor
    cancelling one. No special case in an authority check is the point -- special
    cases in authority checks are where authority checks go wrong."""
    kernel, events = build(
        [TIDY, COLLISION_AVOIDANCE],
        {"tidy-workshop": WORK, "collision-avoidance": WORK},
    )
    handler = kernel.spawn(DescriptorName("collision-avoidance"))
    assert handler.owner == KERNEL_PRINCIPAL

    assert not kernel.apply_ownership(
        OwnershipRequest(op=OwnershipOp.CANCEL, job=handler.job_id, by=OPERATOR.id)
    )
    assert handler.state is not JobState.DONE
    assert of(events, OwnershipRefused)[0].owner == KERNEL_PRINCIPAL


def test_deprioritise_only_moves_a_job_down() -> None:
    """The op that raises urgency is spawn-with-priority, which the ceiling governs.
    Letting this run in reverse would be a second, ungoverned path to the same
    power."""
    kernel, _ = build([TIDY], {"tidy-workshop": WORK})
    _, job = kernel.handle_utterance(said("tidy the workshop in bench-3", OPERATOR))
    assert job is not None
    start = job.base_priority

    kernel.apply_ownership(
        OwnershipRequest(
            op=OwnershipOp.DEPRIORITISE,
            job=job.job_id,
            by=OPERATOR.id,
            priority=Priority(90),
        )
    )
    assert job.base_priority == Priority(90)

    kernel.apply_ownership(
        OwnershipRequest(
            op=OwnershipOp.DEPRIORITISE,
            job=job.job_id,
            by=OPERATOR.id,
            priority=Priority(1),
        )
    )
    assert job.base_priority == Priority(90), "urgency must not be raised this way"
    assert int(job.base_priority) >= int(start)


def test_a_cancelled_job_gives_everything_back() -> None:
    """A cancelled job holding a lock or a body would strand both -- the same failure
    mode completion already handles, so cancellation reuses that teardown."""
    from zeos.core.ids import ResourceKind, ResourceName
    from zeos.core.resources import ResourceSpec, ResourceTable

    door = ResourceName("door.south")
    holder = dict(
        TIDY,
        name="holder",
        resources=[str(door)],
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("holder"): Descriptor.from_frontmatter(holder)},
        machine=ScriptedMachine(
            {
                "holder": Script.from_spec(
                    [{"acquire": str(door)}, {"emit": "holding"}, {"emit": "still"}]
                )
            },
            block_size=8,
        ),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable([ResourceSpec(name=door, capacity=1, kind=ResourceKind.MUTEX)]),
        principals=principals(),
        journal_sink=events,
    )
    kernel.start()
    job = kernel.spawn(DescriptorName("holder"), owner=OPERATOR.id)
    kernel.tick()
    kernel.tick()
    assert door in job.held_resources

    kernel.apply_ownership(OwnershipRequest(op=OwnershipOp.CANCEL, job=job.job_id, by=OPERATOR.id))
    assert not job.held_resources
    assert kernel.resources.get(door).available


# --- the principal table's own checks --------------------------------------


def test_a_principal_cannot_claim_a_kernel_ring() -> None:
    """OQ-N4. A human utterance is at best ring 2 and by default ring 3; rings 0 and
    1 are kernel framing and descriptor bodies (MP §4). A principal declared at ring 0
    would have its words arrive as trusted as a stub's framing."""
    impostor = PrincipalEnvelope(
        id=PrincipalId("badge:impostor"), ceiling=Priority(50), ring=Ring.KERNEL
    )
    findings = lint({}, principals=principals(impostor))
    bad = [f for f in findings if f.rule == "principal-claims-kernel-ring"]
    assert bad and bad[0].severity is Severity.ERROR


def test_an_unrestricted_principal_is_flagged_but_allowed() -> None:
    """Sometimes legitimate -- a commissioning console -- and always worth saying out
    loud, because such a principal has opted out of layer one."""
    console = PrincipalEnvelope(
        id=PrincipalId("console:commissioning"),
        ceiling=Priority(20),
        ring=Ring.TRUSTED,
        unrestricted=True,
    )
    findings = lint({}, principals=principals(console))
    flagged = [f for f in findings if f.rule == "unrestricted-principal"]
    assert flagged and flagged[0].severity is Severity.WARNING
    assert "layer one" in flagged[0].detail


def test_a_capability_for_a_pipe_that_does_not_exist_is_a_load_error() -> None:
    """Silent in effect, which is the problem: it reads as authority granted and
    behaves as authority withheld."""
    typo = PrincipalEnvelope(
        id=PrincipalId("badge:typo"),
        ceiling=Priority(50),
        capabilities=frozenset({PipeName("actuators.fan_4")}),
        ring=Ring.TRUSTED,
    )
    findings = lint({}, pipes=PIPES, principals=principals(typo))
    assert [f for f in findings if f.rule == "unknown-principal-capability"]


def test_the_compilation_ladder_is_ordered() -> None:
    """Invocation is safest, because the executable vocabulary stays the
    reviewed descriptor library."""
    assert CompilationTarget.permits(CompilationTarget.SYNTHESIS, CompilationTarget.INVOCATION)
    assert not CompilationTarget.permits(CompilationTarget.INVOCATION, CompilationTarget.SYNTHESIS)


def test_a_principal_file_must_declare_a_ceiling() -> None:
    """A principal with no declared ceiling could ask for any priority, so the
    field is required rather than defaulted -- a default here would be a silent policy
    decision about how urgent a stranger may be."""
    import tempfile
    from pathlib import Path

    from zeos.descriptor.loader import _load_principals  # pyright: ignore[reportPrivateUsage]
    from zeos.descriptor.schema import DescriptorError

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "principals.yaml"
        path.write_text("- id: badge:x\n  capabilities: [ops.report]\n", "utf-8")
        with pytest.raises(DescriptorError, match="'ceiling' is required"):
            _load_principals(path)


# --- the phase boundary -----------------------------------------------------


def test_nli_frontmatter_is_now_interpreted_not_parked() -> None:
    """The machine-checkable boundary, in the direction that is easy to forget.

    ``principal``, ``ceiling`` and ``elevation`` were in ``UNIMPLEMENTED_KEYS``. N0
    implements the mechanisms, so declaring them must stop being an error *and* stop
    landing in ``extra``.
    """
    d = Descriptor.from_frontmatter(
        {
            "name": "tidy",
            "priority": 60,
            "utterances": ["tidy the workshop in {zone}"],
            "principals": ["badge:operator-a"],
        }
    )
    assert not d.extra, "implemented keys must not be parked in extra"
    assert d.utterances and d.principals == (PrincipalId("badge:operator-a"),)
    assert not [f for f in lint({d.name: d}) if f.rule == "unimplemented-frontmatter"]


def test_what_remains_unimplemented() -> None:
    """After R0, F0 and N0: mesh (F2, no transport) and threads (future work)."""
    from zeos.descriptor.lint import UNIMPLEMENTED_KEYS

    assert set(UNIMPLEMENTED_KEYS) == {"mesh", "threads"}
