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
    """Tokenise kernel text: frame tags as ``CONTROL``, the words inside as ``NORMAL``.

    A tag runs from its opener to the first word ending in ``>``, so ``<STATUS obj>``
    and ``<FAULT kind=x>`` are whole tags and the value between them keeps its kind.
    """
    out: list[Token] = []
    in_tag = False
    for word in text.split():
        if not in_tag and opens_frame(word):
            in_tag = True
        out.append(Token(word, TokenKind.CONTROL if in_tag else TokenKind.NORMAL))
        if in_tag and word.endswith(">"):
            in_tag = False
    return tuple(out)


def imitates_frame(tokens: Iterable[Token]) -> bool:
    """Whether ordinary tokens spell a frame tag."""
    return any(t.kind is TokenKind.NORMAL and opens_frame(t.text) for t in tokens)


def shown(token: Token) -> str:
    """A token as a text-only source sees it: a frame as it is, an imitation escaped."""
    if token.kind is TokenKind.NORMAL and opens_frame(token.text):
        return token.text.replace("<", "&lt;").replace(">", "&gt;")
    return token.text
