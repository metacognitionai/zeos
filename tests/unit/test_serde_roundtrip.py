# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Every journalled event must round-trip exactly.

This is the foundation of acceptance criterion 1 (determinism): if an event cannot
survive encode/decode, replay silently diverges from the run it claims to reproduce.
Rather than hand-writing a case per event type, the test walks the registry and
synthesises an instance of each from its annotations -- so a newly added event is
covered the moment it is registered, and an event with an unserialisable field
fails here instead of during a run.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any

import pytest

from zeos.core import serde
from zeos.core.clock import Clock
from zeos.core.events import EVENT_REGISTRY, Event
from zeos.journal.codec import from_line, to_line


def _sample(tp: Any, salt: int) -> Any:
    """Build a deterministic sample value for an annotation."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is types.UnionType or origin is typing.Union:
        # Exercise the populated branch of ``X | None``, not the None branch --
        # None round-trips trivially and would hide field-level bugs.
        non_none = [a for a in args if a is not type(None)]
        return _sample(non_none[0], salt)

    supertype = getattr(tp, "__supertype__", None)
    if supertype is not None:
        return _sample(supertype, salt)

    if isinstance(tp, type) and issubclass(tp, enum.Flag):
        members = [m for m in tp if m.value]
        return members[0] if members else tp(0)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return list(tp)[salt % len(list(tp))]

    if tp is int:
        return salt
    if tp is float:
        return float(salt) + 0.5
    if tp is bool:
        return salt % 2 == 0
    if tp is str:
        return f"s{salt}"

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_sample(args[0], salt + i) for i in range(2))
        return tuple(_sample(a, salt + i) for i, a in enumerate(args))
    if origin is frozenset:
        return frozenset(_sample(args[0], salt + i) for i in range(2))
    if origin is dict:
        return {f"k{i}": _sample(args[1], salt + i) for i in range(2)}

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build(tp, salt)

    raise AssertionError(f"test cannot synthesise a sample for {tp!r}")


def _build(cls: type, salt: int) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for i, fld in enumerate(dataclasses.fields(cls)):
        if not fld.init:
            continue
        kwargs[fld.name] = _sample(hints[fld.name], salt + i * 7 + 1)
    return cls(**kwargs)


EVENT_TYPES = sorted(EVENT_REGISTRY.items())


def test_registry_is_populated() -> None:
    assert len(EVENT_REGISTRY) > 40, "event alphabet looks truncated"


@pytest.mark.parametrize("kind,cls", EVENT_TYPES, ids=[k for k, _ in EVENT_TYPES])
def test_event_roundtrips_through_serde(kind: str, cls: type[Event]) -> None:
    original = _build(cls, salt=3)
    restored = serde.decode(cls, serde.encode(original))
    assert restored == original, f"{kind} did not survive serde"


@pytest.mark.parametrize("kind,cls", EVENT_TYPES, ids=[k for k, _ in EVENT_TYPES])
def test_event_roundtrips_through_journal_line(kind: str, cls: type[Event]) -> None:
    original = _build(cls, salt=11)
    seq, restored = from_line(to_line(42, original))
    assert seq == 42
    assert restored == original, f"{kind} did not survive the journal wire format"


@pytest.mark.parametrize("kind,cls", EVENT_TYPES, ids=[k for k, _ in EVENT_TYPES])
def test_journal_line_is_stable(kind: str, cls: type[Event]) -> None:
    """Byte-identical output for identical input. The determinism gate compares
    bytes, so key ordering must not wander between encodes."""
    original = _build(cls, salt=5)
    assert to_line(1, original) == to_line(1, original)


def test_kinds_are_unique_and_namespaced() -> None:
    for kind, cls in EVENT_TYPES:
        assert kind == cls.KIND
        assert kind.islower(), f"{kind} should be lowercase"


def test_missing_optional_field_falls_back_to_default() -> None:
    """A journal written before a field was added still replays."""
    from zeos.core.events import FaultRaised
    from zeos.core.ids import FaultKind, JobId

    raw = {
        "clock": {"token_clock": 1, "virtual_ns": 2},
        "job": 7,
        "fault": FaultKind.BUDGET.value,
        "detail": "over budget",
        # 'segment' and 'pipe' omitted: both declare defaults
    }
    restored = serde.decode(FaultRaised, raw)
    assert restored.job == JobId(7)
    assert restored.segment is None and restored.pipe is None


def test_missing_required_field_is_an_error() -> None:
    from zeos.core.events import JobDispatched

    with pytest.raises(serde.SerdeError, match="missing required field"):
        serde.decode(JobDispatched, {"clock": {"token_clock": 0, "virtual_ns": 0}})


def test_unencodable_value_is_rejected_loudly() -> None:
    with pytest.raises(serde.SerdeError, match="cannot encode"):
        serde.encode_value(object())


def test_clock_is_monotonic_by_contract() -> None:
    clock = Clock(token_clock=5, virtual_ns=100)
    assert clock.tick_tokens(3).token_clock == 8
    assert clock.at_virtual(200).virtual_ns == 200
    with pytest.raises(ValueError, match="backwards"):
        clock.at_virtual(50)
    with pytest.raises(ValueError, match="monotonic"):
        clock.tick_tokens(-1)
