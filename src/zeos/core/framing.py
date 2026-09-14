# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The kernel's frames: the tags it puts around its own text (MP §5.3).

A frame is carried on ``CONTROL`` tokens, which a machine cannot decode unless the
kernel enables it, so the frame is what carries authority. Ordinary tokens that spell
a frame tag are an imitation: inert, alarmed on as a spoof fault, and shown escaped
to a source that reads its context as text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from zeos.core.ids import TokenKind
from zeos.machine.base import Token

__all__ = ["FRAMES", "frame_tokens", "imitates_frame", "opens_frame", "shown"]

FRAMES: tuple[str, ...] = ("KERNEL", "RESUME", "FAULT", "STATUS", "STUB")

_OPENER = re.compile(r"^</?(?:" + "|".join(FRAMES) + r")(?=[\s>]|$)")


def opens_frame(word: str) -> bool:
    """Whether a whitespace token begins a frame tag, opening or closing."""
    return _OPENER.match(word) is not None


def frame_tokens(text: str) -> tuple[Token, ...]:
    """Tokenise one kernel frame: its outer tags as ``CONTROL``, everything inside as
    ``NORMAL``.

    The opening tag runs from the first word to the first word ending in ``>``, so
    ``<STATUS obj>`` and ``<FAULT kind=x>`` are whole tags; the closing tag is the last
    word when it closes the same frame. Only those two are the kernel's. A word inside the
    body that spells a tag, a device value or a quoted request say, stays ``NORMAL``, so
    the kernel never promotes an imitation into a frame of its own.
    """
    words = text.split()
    opener = _OPENER.match(words[0]) if words else None
    if opener is None:
        return tuple(Token(w, TokenKind.NORMAL) for w in words)
    name = opener.group(0).lstrip("</")
    end = next((i for i, w in enumerate(words) if w.endswith(">")), len(words) - 1)
    kinds = [TokenKind.NORMAL] * len(words)
    for i in range(end + 1):
        kinds[i] = TokenKind.CONTROL
    if len(words) - 1 > end and words[-1] == f"</{name}>":
        kinds[-1] = TokenKind.CONTROL
    return tuple(Token(w, k) for w, k in zip(words, kinds, strict=True))


def imitates_frame(tokens: Iterable[Token]) -> bool:
    """Whether ordinary tokens spell a frame tag."""
    return any(t.kind is TokenKind.NORMAL and opens_frame(t.text) for t in tokens)


def shown(token: Token) -> str:
    """A token as a text-only source sees it: a frame as it is, an imitation escaped."""
    if token.kind is TokenKind.NORMAL and opens_frame(token.text):
        return token.text.replace("<", "&lt;").replace(">", "&gt;")
    return token.text
