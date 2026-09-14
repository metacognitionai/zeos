# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Descriptor parsing and load-time lint -- the compile step."""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.core.ids import DescriptorName, ObjectName, OnComplete, OnFault, Placement
from zeos.core.pipes import PipeSpec
from zeos.descriptor.lint import Finding, LintOptions, Severity, lint
from zeos.descriptor.loader import load_case, split_frontmatter
from zeos.descriptor.schema import Descriptor, DescriptorError, parse_duration_ns
from zeos.machine.abi import DEFAULT, SyscallABI, Verb
from zeos.machine.base import OpKind

SMOKE = Path(__file__).parent.parent / "fixtures" / "smoke"


# --- durations --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("40ms", 40_000_000), ("2s", 2_000_000_000), ("1m", 60_000_000_000), ("500us", 500_000)],
)
def test_durations_parse(text: str, expected: int) -> None:
    assert parse_duration_ns(text) == expected


def test_bare_numbers_are_rejected_as_durations() -> None:
    """A bare number in a latency budget is an invitation to a unit error, and unit
    errors in latency budgets produce missed deadlines nobody can explain."""
    with pytest.raises(DescriptorError, match="need units"):
        parse_duration_ns(40, field_name="deadline")


def test_none_is_a_valid_duration() -> None:
    assert parse_duration_ns("none") is None
    assert parse_duration_ns(None) is None


# --- frontmatter ------------------------------------------------------------


def test_split_frontmatter() -> None:
    raw, body = split_frontmatter(
        "---\nname: x\npriority: 5\n---\n# Task\n\nDo the thing.\n", source="t"
    )
    assert raw["name"] == "x"
    assert body.startswith("# Task")


def test_missing_frontmatter_is_an_error() -> None:
    """A behaviour with no declared contract cannot be scheduled, protected, or
    linked -- so a body-only file is an error, not a descriptor."""
    with pytest.raises(DescriptorError, match="expected YAML frontmatter"):
        split_frontmatter("# Just a body\n", source="t")


def test_unterminated_frontmatter_is_an_error() -> None:
    with pytest.raises(DescriptorError, match="unterminated"):
        split_frontmatter("---\nname: x\n", source="t")


# --- schema -----------------------------------------------------------------


def test_priority_is_required() -> None:
    with pytest.raises(DescriptorError, match="'priority' is required"):
        Descriptor.from_frontmatter({"name": "x"})


def test_priority_must_be_in_range() -> None:
    with pytest.raises(DescriptorError, match="outside"):
        Descriptor.from_frontmatter({"name": "x", "priority": 5000})


def test_unknown_placement_is_rejected() -> None:
    with pytest.raises(DescriptorError, match="unknown placement"):
        Descriptor.from_frontmatter({"name": "x", "priority": 1, "placement": "moon"})


def test_completion_policies_parse() -> None:
    d = Descriptor.from_frontmatter({"name": "x", "priority": 1, "on_complete": "cancel-below:3"})
    assert d.on_complete.kind is OnComplete.CANCEL_BELOW and d.on_complete.depth == 3

    d2 = Descriptor.from_frontmatter(
        {"name": "y", "priority": 1, "on_complete": "replace-with:recharge"}
    )
    assert d2.on_complete.kind is OnComplete.REPLACE_WITH
    assert d2.on_complete.replacement == DescriptorName("recharge")


def test_cancel_below_needs_a_depth() -> None:
    with pytest.raises(DescriptorError, match="positive depth"):
        Descriptor.from_frontmatter({"name": "x", "priority": 1, "on_complete": "cancel-below"})


def test_fault_policy_with_handler() -> None:
    d = Descriptor.from_frontmatter({"name": "x", "priority": 1, "on_fault": "handler:triage"})
    assert d.on_fault.kind is OnFault.HANDLER
    assert d.on_fault.handler == DescriptorName("triage")


def test_unknown_frontmatter_keys_are_preserved_not_dropped() -> None:
    """A descriptor written against a later design must survive this kernel rather
    than silently losing configuration.

    Every field the five specs define is now first-class, so this uses keys from the
    open questions instead -- a ``slots:`` block for the PNP register-file idea
    (core OQ-6) and a federation ``topology`` reference (ZEOS-Distributed).
    """
    d = Descriptor.from_frontmatter(
        {
            "name": "x",
            "priority": 1,
            "context": {"window": 32768},  # first-class: must NOT land in extra
            "slots": {"bank": 64},
            "topology": "two-node",
        }
    )
    assert set(d.extra) == {"slots", "topology"}
    assert d.context.window == 32768


def test_mp_fields_are_first_class() -> None:
    d = Descriptor.from_frontmatter(
        {
            "name": "researcher",
            "priority": 50,
            "integrity": {"start": 2, "dynamics": "low-watermark"},
            "capabilities": [{"pipe": "mail.send", "min_integrity": 2}],
            "endorsers": ["web-summarizer"],
            "compartments": [
                {
                    "name": "reader",
                    "descriptor": "web-reader",
                    "integrity": 3,
                    "grants": ["web.fetch"],
                }
            ],
        }
    )
    assert d.integrity.start == 2 and d.integrity.is_dynamic
    assert d.capabilities[0].pipe == "mail.send"
    assert d.capabilities[0].min_integrity == 2
    assert d.endorsers == (DescriptorName("web-summarizer"),)
    assert d.compartments[0].grants == ("web.fetch",)


def test_static_integrity_dynamics_is_allowed_but_typo_is_not() -> None:
    d = Descriptor.from_frontmatter(
        {"name": "x", "priority": 1, "integrity": {"dynamics": "static"}}
    )
    assert not d.integrity.is_dynamic
    with pytest.raises(DescriptorError, match="low-watermark"):
        Descriptor.from_frontmatter(
            {"name": "y", "priority": 1, "integrity": {"dynamics": "lowwatermark"}}
        )


# --- case loading -----------------------------------------------------------


def test_load_case_reads_the_whole_tree() -> None:
    bundle = load_case(SMOKE)
    assert set(bundle.descriptors) == {
        DescriptorName("supervision"),
        DescriptorName("threshold-alarm"),
    }
    assert {p.name for p in bundle.pipes} == {"sensors.threshold", "actuators.unit_a", "ops.report"}
    assert len(bundle.vectors) == 1
    assert bundle.vectors[0].deadline_ns == 40_000_000
    assert bundle.world[ObjectName("plant.unit_a")] == "idle"


def test_scripts_are_extracted_from_frontmatter() -> None:
    """Scripts are M0 scaffolding, not part of the descriptor format -- so they are
    pulled out here rather than becoming a field on Descriptor."""
    bundle = load_case(SMOKE)
    assert set(bundle.scripts) == {"supervision", "threshold-alarm"}
    assert bundle.scripts["threshold-alarm"].steps


def test_boot_is_inferred_as_goals_not_reachable_another_way() -> None:
    """Handlers are dispatched by their vectors, so they must not also boot."""
    bundle = load_case(SMOKE)
    assert bundle.boot == (DescriptorName("supervision"),)


# --- lint -------------------------------------------------------------------


def test_the_smoke_case_lints_clean() -> None:
    bundle = load_case(SMOKE)
    findings = lint(bundle.descriptors, pipes=bundle.pipes, vectors=bundle.vectors)
    assert findings == (), [f.render() for f in findings]


SEAT = Path(__file__).resolve().parents[1] / "fixtures" / "seat"


def _speaking(body: str, **frontmatter: object) -> Descriptor:
    return Descriptor.from_frontmatter(
        {"name": "x", "priority": 50, "pipes": {"stdout": "ops.report", "tools": "act.a"}}
        | frontmatter,
        body=body,
    )


def _lint_body(d: Descriptor, *, abi: SyscallABI = DEFAULT) -> tuple[Finding, ...]:
    """Lint one descriptor with its pipes declared, so only the body rules can speak."""
    return lint({d.name: d}, pipes=[PipeSpec(n) for n in d.pipes.all_names()], abi=abi)


def test_the_seat_fixture_lints_clean() -> None:
    bundle = load_case(SEAT)
    findings = lint(bundle.descriptors, pipes=bundle.pipes)
    assert findings == (), [f.render() for f in findings]


def test_a_body_naming_a_verb_the_abi_lacks_is_rejected() -> None:
    """The body is the one hand-written copy of the vocabulary; a word the ABI does not
    declare is one the grammar will never offer and the API seat will never parse."""
    d = _speaking("Say your number, then `record tools N;` to keep it.")
    findings = _lint_body(d)
    assert [f.rule for f in findings] == ["unknown-body-verb"]
    assert findings[0].severity is Severity.ERROR
    assert "`record tools N;`" in findings[0].detail and "'record'" in findings[0].detail


def test_a_body_naming_a_pipe_the_descriptor_does_not_bind_is_rejected() -> None:
    d = _speaking("When done, `write peer go;` to wake the other job.")
    findings = _lint_body(d)
    assert [f.rule for f in findings] == ["unbound-body-pipe"]
    assert findings[0].severity is Severity.ERROR
    assert "'peer'" in findings[0].detail


def test_a_pipe_verb_named_without_its_pipe_is_rejected() -> None:
    d = _speaking("Then `write;` and stop.")
    assert [f.rule for f in _lint_body(d)] == ["unbound-body-pipe"]


def test_placeholders_in_a_payload_are_not_checked() -> None:
    d = _speaking("`say N;` then `write tools TARGET;` then `write stdout go;` then `exit;`")
    assert _lint_body(d) == ()


def test_an_extra_binding_counts_as_bound() -> None:
    d = _speaking("`write peer N;` sets the second counter.", pipes={"peer": "count.progress_b"})
    assert _lint_body(d) == ()


def test_prose_with_a_semicolon_is_not_read_as_a_command() -> None:
    d = _speaking("Count up; then stop. The status line names `count.b` and `stdin` is a pipe.")
    assert _lint_body(d) == ()


def test_every_command_in_a_span_is_checked() -> None:
    d = _speaking("From 47 the count is `say 48; say 49; shout 50;` and no more.")
    findings = _lint_body(d)
    assert [f.rule for f in findings] == ["unknown-body-verb"]
    assert "'shout'" in findings[0].detail


def test_the_body_is_checked_against_the_abi_it_is_given() -> None:
    custom = SyscallABI(
        verbs=(Verb("send", OpKind.WRITE, pipe=True, text=True), Verb("done", OpKind.EXIT)),
        aliases=("out",),
        terminator="!",
    )
    d = _speaking("`send out hello!` then `done!`", pipes={"out": "ops.report"})
    assert _lint_body(d, abi=custom) == ()
    assert [f.rule for f in _lint_body(d)] == [], "the default ABI's terminator is not '!'"
    d = _speaking("`send out hello;` then `done;`", pipes={"out": "ops.report"})
    assert [f.rule for f in _lint_body(d)] == ["unknown-body-verb", "unknown-body-verb"]


def test_unpreemptible_with_a_large_budget_is_rejected() -> None:
    """Core §3: masking must be paired with a small budget. This is the
    cli-with-no-sti bug, and it is an error rather than a warning."""
    d = Descriptor.from_frontmatter(
        {"name": "x", "priority": 1, "preemptible": False, "budget": {"tokens": 200_000}}
    )
    findings = lint({d.name: d})
    assert [f.rule for f in findings] == ["unpreemptible-large-budget"]
    assert findings[0].severity is Severity.ERROR


def test_unpreemptible_with_no_budget_is_rejected() -> None:
    d = Descriptor.from_frontmatter({"name": "x", "priority": 1, "preemptible": False})
    findings = lint({d.name: d})
    assert [f.rule for f in findings] == ["unpreemptible-unbounded"]


def test_unknown_child_is_rejected() -> None:
    d = Descriptor.from_frontmatter({"name": "parent", "priority": 1, "children": ["ghost"]})
    findings = lint({d.name: d})
    assert [f.rule for f in findings] == ["unknown-child"]


def test_equal_priority_write_conflict_is_flagged() -> None:
    """At equal priority there is nothing to arbitrate with,
    so the interleaving is unspecified."""
    a = Descriptor.from_frontmatter({"name": "a", "priority": 50, "writes": ["robot.position"]})
    b = Descriptor.from_frontmatter({"name": "b", "priority": 50, "writes": ["robot.*"]})
    findings = lint({a.name: a, b.name: b})
    assert [f.rule for f in findings] == ["concurrent-write"]
    assert findings[0].severity is Severity.WARNING


def test_different_priorities_do_not_conflict() -> None:
    a = Descriptor.from_frontmatter({"name": "a", "priority": 10, "writes": ["robot.position"]})
    b = Descriptor.from_frontmatter({"name": "b", "priority": 50, "writes": ["robot.position"]})
    assert lint({a.name: a, b.name: b}) == ()


def test_offboard_handler_behind_a_slow_link_is_rejected() -> None:
    """A deadline tighter than the link cannot be met
    off-platform at any load -- a type error, not a scheduling preference."""
    bundle = load_case(SMOKE)
    handler = bundle.descriptors[DescriptorName("threshold-alarm")]
    offboard = Descriptor.from_frontmatter(
        {
            "name": str(handler.name),
            "priority": int(handler.priority),
            "placement": Placement.OFFBOARD.value,
            "preemptible": False,
            "budget": {"tokens": 64},
        }
    )
    descriptors = dict(bundle.descriptors) | {offboard.name: offboard}
    findings = lint(
        descriptors,
        pipes=bundle.pipes,
        vectors=bundle.vectors,
        options=LintOptions(link_rtt_p99_ns=120_000_000),  # 120ms link, 40ms deadline
    )
    assert any(f.rule == "unplaceable-handler" for f in findings)
    assert all(f.severity is Severity.ERROR for f in findings if f.rule == "unplaceable-handler")


def test_platform_handler_behind_the_same_link_is_fine() -> None:
    bundle = load_case(SMOKE)
    findings = lint(
        bundle.descriptors,
        pipes=bundle.pipes,
        vectors=bundle.vectors,
        options=LintOptions(link_rtt_p99_ns=120_000_000),
    )
    assert not [f for f in findings if f.rule == "unplaceable-handler"]


def test_priority_is_unaffected_by_declaration_order() -> None:
    """Determinism: findings must not depend on dict insertion order."""
    a = Descriptor.from_frontmatter({"name": "a", "priority": 50, "writes": ["x.y"]})
    b = Descriptor.from_frontmatter({"name": "b", "priority": 50, "writes": ["x.y"]})
    forward = lint({a.name: a, b.name: b})
    backward = lint({b.name: b, a.name: a})
    assert [f.detail for f in forward] == [f.detail for f in backward]
