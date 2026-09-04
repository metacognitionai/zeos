# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Journal wire format.

One JSON object per line. Keys are sorted and separators are fixed so that the
same run produces byte-identical output -- the determinism gate compares bytes, not
parsed structures, because a comparison that tolerates reordering would not catch
the class of bug we most care about (nondeterministic iteration order leaking into
kernel decisions).

The line is deliberately flat -- ``{"seq": 12, "kind": "job.preempted", "job": 3, ...}``
-- so that a journal is greppable and jq-able without a schema.
"""

from __future__ import annotations

import json
from typing import Any, Final

from zeos.core import serde
from zeos.core.events import EVENT_REGISTRY, Event

__all__ = ["encode_record", "decode_record", "to_line", "from_line", "JournalFormatError"]

_SEQ: Final = "seq"
_KIND: Final = "kind"


class JournalFormatError(ValueError):
    """A journal line that cannot be parsed back into an event."""


def encode_record(seq: int, event: Event) -> dict[str, Any]:
    payload = serde.encode(event)
    if _SEQ in payload or _KIND in payload:
        raise JournalFormatError(
            f"{type(event).__name__} declares a field named {_SEQ!r} or {_KIND!r}, "
            "which collides with the record envelope"
        )
    return {_SEQ: seq, _KIND: event.KIND, **payload}


def decode_record(raw: dict[str, Any]) -> tuple[int, Event]:
    try:
        seq = int(raw[_SEQ])
        kind = str(raw[_KIND])
    except KeyError as exc:
        raise JournalFormatError(f"journal record missing {exc.args[0]!r}") from exc
    cls = EVENT_REGISTRY.get(kind)
    if cls is None:
        raise JournalFormatError(f"unknown event kind {kind!r}")
    fields = {k: v for k, v in raw.items() if k not in (_SEQ, _KIND)}
    return seq, serde.decode(cls, fields)


def to_line(seq: int, event: Event) -> str:
    return json.dumps(
        encode_record(seq, event),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def from_line(line: str) -> tuple[int, Event]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise JournalFormatError(f"malformed journal line: {exc}") from exc
    if not isinstance(raw, dict):
        raise JournalFormatError("journal line is not a JSON object")
    return decode_record(raw)  # pyright: ignore[reportUnknownArgumentType]
