# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Pure, stdlib-only structural serialisation for kernel data.

Everything the kernel journals must round-trip exactly: ``decode(encode(x)) == x``.
That property is what makes the determinism gate (acceptance criterion 1) checkable,
so this module is deliberately small, explicit, and total over the type shapes the
kernel actually uses. Unsupported shapes raise rather than degrading silently -- a
field that cannot round-trip is a bug we want at author time, not at replay time.

Supported field types:
    primitives          int, float, str, bool, None
    enums               any ``enum.Enum`` whose values are primitives
    optionals           ``X | None``
    homogeneous tuples  ``tuple[X, ...]``
    fixed tuples        ``tuple[X, Y]``
    frozensets          ``frozenset[X]``   (encoded sorted, for stable output)
    mappings            ``dict[str, X]``   (encoded key-sorted, for stable output)
    nested dataclasses  any frozen dataclass reachable from a registered root

No I/O happens here. JSON encoding lives in ``zeos.journal``.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any, TypeVar, cast

__all__ = ["encode", "decode", "encode_value", "decode_value", "SerdeError"]

T = TypeVar("T")


class SerdeError(TypeError):
    """A value or type annotation that cannot round-trip."""


# Resolved type hints are cached per class: ``get_type_hints`` is expensive and the
# kernel encodes on every journalled event.
_HINT_CACHE: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    cached = _HINT_CACHE.get(cls)
    if cached is None:
        # include_extras=False: Annotated metadata is documentation, not structure.
        cached = typing.get_type_hints(cls)
        _HINT_CACHE[cls] = cached
    return cached


def _is_frozen_dataclass(tp: Any) -> bool:
    if not (isinstance(tp, type) and dataclasses.is_dataclass(tp)):
        return False
    params: Any = getattr(tp, "__dataclass_params__", None)
    return bool(getattr(params, "frozen", False))


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return ``(inner, was_optional)`` for ``X | None``; else ``(tp, False)``."""
    origin = typing.get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) != len(typing.get_args(tp)):
            if len(args) != 1:
                raise SerdeError(f"only ``X | None`` unions are supported, got {tp!r}")
            return args[0], True
    return tp, False


def encode_value(value: Any) -> Any:
    """Encode a single value structurally. Type-directed decoding happens in
    ``decode_value``; encoding can be driven by the runtime value alone."""
    if value is None or isinstance(value, int | float | str | bool):
        # bool before int matters for JSON fidelity, but isinstance covers both.
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, frozenset):
        # Sorted so that identical sets always encode to identical bytes.
        return sorted(encode_value(v) for v in cast("frozenset[Any]", value))
    if isinstance(value, tuple):
        return [encode_value(v) for v in cast("tuple[Any, ...]", value)]
    if isinstance(value, dict):
        items = cast("dict[str, Any]", value)
        return {k: encode_value(items[k]) for k in sorted(items)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    raise SerdeError(f"cannot encode value of type {type(value).__name__}")


def decode_value(tp: Any, raw: Any) -> Any:
    """Decode ``raw`` into the shape described by annotation ``tp``."""
    inner, optional = _unwrap_optional(tp)
    if raw is None:
        if optional or inner is type(None):
            return None
        raise SerdeError(f"got None for non-optional {tp!r}")

    # NewType is transparent at runtime, so it only needs unwrapping on the way in.
    # The kernel's identifiers (JobId, PipeName, ...) are all NewTypes, so this is
    # on the hot path for essentially every event.
    supertype = getattr(inner, "__supertype__", None)
    if supertype is not None:
        return decode_value(supertype, raw)

    if inner in (int, float, str, bool):
        return inner(raw)
    if inner is Any:
        return raw
    if isinstance(inner, type) and issubclass(inner, enum.Enum):
        return inner(raw)

    origin = typing.get_origin(inner)
    args = typing.get_args(inner)

    if origin is tuple:
        seq = cast("list[Any]", raw)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode_value(args[0], v) for v in seq)
        if len(args) != len(seq):
            raise SerdeError(f"arity mismatch decoding {inner!r} from {seq!r}")
        return tuple(decode_value(a, v) for a, v in zip(args, seq, strict=True))
    if origin is frozenset:
        return frozenset(decode_value(args[0], v) for v in cast("list[Any]", raw))
    if origin is dict:
        mapping = cast("dict[str, Any]", raw)
        return {k: decode_value(args[1], mapping[k]) for k in sorted(mapping)}

    if _is_frozen_dataclass(inner):
        return decode(cast("type[Any]", inner), cast("dict[str, Any]", raw))

    raise SerdeError(f"cannot decode annotation {tp!r}")


def encode(obj: Any) -> dict[str, Any]:
    """Encode a frozen dataclass instance to a plain dict."""
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        raise SerdeError(f"encode expects a dataclass instance, got {type(obj)!r}")
    return cast("dict[str, Any]", encode_value(obj))


def decode[T](cls: type[T], raw: dict[str, Any]) -> T:
    """Reconstruct a frozen dataclass instance from a plain dict.

    Fields absent from ``raw`` fall back to their declared default, so a journal
    written by an older build still replays when a field is added with a default.
    """
    if not _is_frozen_dataclass(cls):
        raise SerdeError(f"decode expects a frozen dataclass, got {cls!r}")
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    # ``_is_frozen_dataclass`` has already established this; the cast is only to
    # satisfy the ``DataclassInstance`` protocol that ``fields`` is annotated with.
    for field in dataclasses.fields(cast("Any", cls)):
        if not field.init:
            continue
        if field.name not in raw:
            has_default = (
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
            )
            if has_default:
                continue
            raise SerdeError(f"{cls.__name__}: missing required field {field.name!r}")
        kwargs[field.name] = decode_value(hints[field.name], raw[field.name])
    return cls(**kwargs)
