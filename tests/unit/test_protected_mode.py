# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Protected Mode units: segments, the watermark, capabilities, and schema width."""

from __future__ import annotations

import pytest

from zeos.core.capabilities import (
    Capability,
    CapabilityTable,
    FieldKind,
    FieldSpec,
    RateLimit,
    Schema,
    check_write,
    parse_payload,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    Perm,
    PipeName,
    Principal,
    Ring,
    SegmentId,
)
from zeos.core.integrity import can_write_at, demote_for_boundary, effective_integrity
from zeos.core.pipes import PipeSpec
from zeos.core.segments import TAG_DESCRIPTOR, Provenance, SegmentTable
from zeos.descriptor.lint import LintOptions, Severity, lint
from zeos.descriptor.schema import Descriptor

# --- segments ---------------------------------------------------------------


def table_with_segments() -> SegmentTable:
    table = SegmentTable(block_size=4)
    table.inject(
        SegmentId(1),
        start=0,
        end=8,
        ring=Ring.DESCRIPTOR,
        integrity=Integrity(1),
        provenance=Provenance(
            pipe=PipeName("kernel"),
            principal=Principal.KERNEL,
            injected_at=0,
            tag=TAG_DESCRIPTOR,
        ),
    )
    table.inject(
        SegmentId(2),
        start=8,
        end=16,
        ring=Ring.EXTERNAL,
        integrity=Integrity(3),
        provenance=Provenance(
            pipe=PipeName("web.fetch"), principal=Principal.TOOL, injected_at=8, tag="web.fetch"
        ),
    )
    return table


def test_external_content_defaults_to_no_execute_bit() -> None:
    table = table_with_segments()
    assert Perm.X in table.get(SegmentId(1)).perms  # descriptor body directs
    assert Perm.X not in table.get(SegmentId(2)).perms  # web content informs only


def test_blocks_map_exactly_because_segments_are_block_aligned() -> None:
    table = table_with_segments()
    assert table.blocks_for(table.get(SegmentId(1))) == frozenset({0, 1})
    assert table.blocks_for(table.get(SegmentId(2))) == frozenset({2, 3})


def test_allowed_blocks_is_the_union_of_readable_segments() -> None:
    table = table_with_segments()
    assert table.allowed_blocks() == frozenset({0, 1, 2, 3})
    table.revoke([SegmentId(2)], Perm.R)
    assert table.allowed_blocks() == frozenset({0, 1})
    assert table.denied_blocks() == frozenset({2, 3})


def test_revocation_is_prospective_only() -> None:
    """Acknowledged in the spec (A4): dropping R removes the segment from future
    forward passes but cannot unwrite what it already influenced."""
    table = table_with_segments()
    record = table.get(SegmentId(2))
    record.attn.ema = 0.9  # it was attended before revocation
    table.revoke([SegmentId(2)], Perm.R)
    assert not record.readable
    assert record.attn.ema == 0.9, "history is not rewritten by revocation"


def test_by_tag_finds_segments_by_source() -> None:
    table = table_with_segments()
    assert [s.id for s in table.by_tag("web.fetch")] == [SegmentId(2)]
    assert [s.id for s in table.by_tag(TAG_DESCRIPTOR)] == [SegmentId(1)]


def test_retract_removes_a_span_and_renumbers_what_follows() -> None:
    """The primitive behind a status-region rewrite.

    ``evict_to_stub`` leaves a handle because an evicted span might be wanted again.
    A retracted span cannot be: it is content the kernel regenerates rather than
    recalls, so there is no record, no stub and no store span left over -- and
    everything downstream closes the gap.
    """
    table = table_with_segments()
    assert table.retract(SegmentId(1)) == 8

    assert SegmentId(1) not in table
    assert [s.id for s in table.all()] == [SegmentId(2)]
    following = table.get(SegmentId(2))
    assert (following.start, following.end) == (0, 8), "the gap is closed, not left open"
    assert table.blocks_for(following) == frozenset({0, 1}), "alignment survives the shift"


def test_retract_refuses_a_span_that_is_not_resident_or_still_open() -> None:
    """Both guards matter: the open segment is the one the machine is still writing
    into, and a stubbed span's tokens are already out of the window."""
    table = table_with_segments()
    table.open_output(
        SegmentId(3),
        at=16,
        ring=Ring.TRUSTED,
        integrity=Integrity(2),
        token_clock=16,
    )
    with pytest.raises(RuntimeError, match="still open"):
        table.retract(SegmentId(3))

    table.evict_to_stub(SegmentId(2), SegmentId(4), stub_tokens=4, token_clock=16)
    with pytest.raises(RuntimeError, match="not resident"):
        table.retract(SegmentId(2))


def test_fork_preserves_ring_and_integrity() -> None:
    """A compartment that could launder taint by being forked would defeat the
    entire mechanism."""
    source = table_with_segments()
    child = SegmentTable(block_size=4)
    assert source.fork_into(child) == 2
    assert child.get(SegmentId(2)).ring is Ring.EXTERNAL
    assert child.get(SegmentId(2)).integrity == Integrity(3)


def test_trunc_drops_segments_at_or_after_the_cut() -> None:
    table = table_with_segments()
    assert table.trunc(8) == (SegmentId(2),)
    assert len(table) == 1


# --- integrity --------------------------------------------------------------


def test_watermark_takes_the_worst_thing_meaningfully_attended() -> None:
    table = table_with_segments()
    demotion = demote_for_boundary(
        Integrity(1),
        table=table,
        mass_this_block={SegmentId(1): 0.4, SegmentId(2): 0.6},
        theta_read=0.2,
    )
    assert demotion.moved and demotion.after == Integrity(3)
    assert demotion.because == (SegmentId(2),)


def test_below_threshold_attention_does_not_demote() -> None:
    table = table_with_segments()
    demotion = demote_for_boundary(
        Integrity(1),
        table=table,
        mass_this_block={SegmentId(2): 0.05},
        theta_read=0.2,
    )
    assert not demotion.moved


def test_a_masked_segment_cannot_demote() -> None:
    """Masking controls visibility, integrity controls authority -- and content the
    forward pass never saw cannot have influenced anything."""
    table = table_with_segments()
    table.revoke([SegmentId(2)], Perm.R)
    demotion = demote_for_boundary(
        Integrity(1), table=table, mass_this_block={SegmentId(2): 0.9}, theta_read=0.2
    )
    assert not demotion.moved


def test_watermark_never_improves_on_its_own() -> None:
    table = table_with_segments()
    demotion = demote_for_boundary(
        Integrity(3), table=table, mass_this_block={SegmentId(1): 0.9}, theta_read=0.2
    )
    assert demotion.after == Integrity(3), "reading clean content does not clean a job"


def test_session_floor_implements_seteuid_drop() -> None:
    assert effective_integrity(Integrity(1), Integrity(3)) == Integrity(3)
    assert effective_integrity(Integrity(3), Integrity(1)) == Integrity(3)
    assert effective_integrity(Integrity(2), None) == Integrity(2)


def test_write_permission_is_lower_is_more_trusted() -> None:
    assert can_write_at(Integrity(1), Integrity(2))
    assert can_write_at(Integrity(2), Integrity(2))
    assert not can_write_at(Integrity(3), Integrity(2))


# --- capabilities -----------------------------------------------------------


def caps(*capabilities: Capability) -> CapabilityTable:
    return CapabilityTable(capabilities)


def test_privileged_write_by_a_dirty_job_is_a_privilege_fault() -> None:
    result = check_write(
        capabilities=caps(Capability(pipe=PipeName("mail.send"), min_integrity=Integrity(2))),
        pipe=PipeName("mail.send"),
        current_integrity=Integrity(3),
        payload="anything",
        now_ns=0,
    )
    assert not result.allowed and result.fault is FaultKind.PRIVILEGE


def test_unheld_pipe_is_a_capability_fault_when_any_are_declared() -> None:
    result = check_write(
        capabilities=caps(Capability(pipe=PipeName("report.out"))),
        pipe=PipeName("mail.send"),
        current_integrity=Integrity(0),
        payload="x",
        now_ns=0,
    )
    assert not result.allowed and result.fault is FaultKind.CAPABILITY


def test_a_descriptor_declaring_no_capabilities_runs_unprotected() -> None:
    """MP adoption is not all-or-nothing: a tree with no capabilities declared is
    running without this layer by choice, and phase 1 lets it."""
    result = check_write(
        capabilities=caps(),
        pipe=PipeName("anything"),
        current_integrity=Integrity(3),
        payload="x",
        now_ns=0,
    )
    assert result.allowed


def test_schema_violation_is_a_capability_fault() -> None:
    schema = Schema.parse("verdict", {"price": "number", "in_stock": "bool"})
    table = caps(Capability(pipe=PipeName("out"), min_integrity=Integrity(3), schema=schema))
    ok = check_write(
        capabilities=table,
        pipe=PipeName("out"),
        current_integrity=Integrity(3),
        payload="price=12.5 in_stock=true",
        now_ns=0,
    )
    assert ok.allowed

    bad = check_write(
        capabilities=table,
        pipe=PipeName("out"),
        current_integrity=Integrity(3),
        payload="price=cheap in_stock=true",
        now_ns=0,
    )
    assert not bad.allowed and bad.fault is FaultKind.CAPABILITY
    assert "expected a number" in bad.detail


def test_extra_fields_are_rejected() -> None:
    """A schema that tolerated extra keys would not bound the channel at all."""
    schema = Schema.parse("verdict", {"price": "number"})
    result = check_write(
        capabilities=caps(
            Capability(pipe=PipeName("out"), min_integrity=Integrity(3), schema=schema)
        ),
        pipe=PipeName("out"),
        current_integrity=Integrity(3),
        payload="price=1 smuggled=lots-of-prose",
        now_ns=0,
    )
    assert not result.allowed and "unexpected field" in result.detail


def test_rate_limit_admits_then_refuses() -> None:
    table = caps(
        Capability(
            pipe=PipeName("out"),
            min_integrity=Integrity(3),
            rate=RateLimit(max_events=2, window_ns=1_000),
        )
    )
    for _ in range(2):
        assert check_write(
            capabilities=table,
            pipe=PipeName("out"),
            current_integrity=Integrity(3),
            payload="x",
            now_ns=0,
        ).allowed
    refused = check_write(
        capabilities=table,
        pipe=PipeName("out"),
        current_integrity=Integrity(3),
        payload="x",
        now_ns=0,
    )
    assert not refused.allowed and "rate limit" in refused.detail


# --- schema width as the security dial --------------------------------------


def test_narrow_schema_smuggles_almost_nothing() -> None:
    """``{price: number, in_stock: bool}`` is a poor smuggling route."""
    schema = Schema.parse("verdict", {"price": "number", "in_stock": "bool"})
    assert schema.capacity_bits() == pytest.approx(65.0)


def test_wide_string_schema_smuggles_plenty() -> None:
    schema = Schema.parse("summary", {"summary": "string(2000)"})
    assert schema.capacity_bits() > 10_000


def test_enum_capacity_is_logarithmic() -> None:
    schema = Schema.parse("verdict", {"call": "enum(buy, hold, sell)"})
    assert schema.capacity_bits() == pytest.approx(1.585, abs=0.01)


def test_payload_parsing() -> None:
    assert parse_payload("a=1 b=two") == {"a": "1", "b": "two"}
    assert parse_payload("noise here") == {}


def test_string_field_respects_max_length() -> None:
    spec = FieldSpec(kind=FieldKind.STRING, max_length=4)
    assert spec.validate("abcd") is None
    assert spec.validate("abcde") is not None


# --- MP lint ----------------------------------------------------------------

DIRTY_PIPES = [
    PipeSpec(PipeName("web.fetch"), ring=Ring.EXTERNAL, principal=Principal.TOOL),
    PipeSpec(PipeName("mail.send"), ring=Ring.TRUSTED, principal=Principal.TOOL),
]


def deputy(**overrides: object) -> Descriptor:
    spec: dict[str, object] = {
        "name": "deputy",
        "priority": 50,
        "integrity": {"start": 2, "dynamics": "static"},
        "capabilities": [
            {"pipe": "web.fetch", "min_integrity": 3},
            {"pipe": "mail.send", "min_integrity": 2},
        ],
    }
    spec.update(overrides)
    return Descriptor.from_frontmatter(spec)


def test_confused_deputy_by_construction_is_rejected() -> None:
    d = deputy()
    findings = lint({d.name: d}, pipes=DIRTY_PIPES)
    deputy_findings = [f for f in findings if f.rule == "confused-deputy"]
    assert deputy_findings and deputy_findings[0].severity is Severity.ERROR


@pytest.mark.parametrize(
    "mitigation",
    [
        {"integrity": {"start": 2, "dynamics": "low-watermark"}},
        {"endorsers": ["web-summarizer"]},
        {"compartments": [{"descriptor": "web-reader", "grants": ["web.fetch"]}]},
    ],
)
def test_any_single_mitigation_satisfies_the_rule(mitigation: dict[str, object]) -> None:
    """The escape hatches are alternatives, not a checklist."""
    d = deputy(**mitigation)
    findings = lint({d.name: d}, pipes=DIRTY_PIPES)
    assert not [f for f in findings if f.rule == "confused-deputy"]


def test_no_privileged_capability_means_no_deputy_problem() -> None:
    d = deputy(capabilities=[{"pipe": "web.fetch", "min_integrity": 3}])
    findings = lint({d.name: d}, pipes=DIRTY_PIPES)
    assert not [f for f in findings if f.rule == "confused-deputy"]


def test_wide_endorsement_schema_is_flagged() -> None:
    d = Descriptor.from_frontmatter(
        {
            "name": "summariser",
            "priority": 50,
            "capabilities": [{"pipe": "mail.send", "min_integrity": 2, "schema": "wide"}],
        },
        schemas={"wide": Schema.parse("wide", {"summary": "string(2000)"})},
    )
    findings = lint({d.name: d}, pipes=DIRTY_PIPES, options=LintOptions())
    wide = [f for f in findings if f.rule == "wide-endorsement-schema"]
    assert wide and wide[0].severity is Severity.WARNING
    assert "bits" in wide[0].detail


def test_capability_naming_an_unknown_schema_is_a_load_error() -> None:
    from zeos.descriptor.schema import DescriptorError

    with pytest.raises(DescriptorError, match="unknown schema"):
        Descriptor.from_frontmatter(
            {
                "name": "x",
                "priority": 1,
                "capabilities": [{"pipe": "out", "schema": "nonexistent"}],
            }
        )


def test_descriptor_name_is_carried_on_findings() -> None:
    d = deputy()
    findings = lint({d.name: d}, pipes=DIRTY_PIPES)
    assert all(f.descriptor == DescriptorName("deputy") for f in findings if f.descriptor)
