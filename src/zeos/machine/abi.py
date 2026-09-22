# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The syscall ABI: the small command language a model speaks to the kernel, declared once.

A model asks the kernel for things by emitting text such as ``write stdout hello;``. Which
words exist, what each one means to the kernel, and how a command ends are declared here
as data, so that the prose a model is told, the pattern a reply is searched with, and any
grammar a sampler is constrained by are all renderings of the same declaration rather
than copies of it.

Nothing here knows how a model is made to obey. A backend that can constrain sampling
renders a grammar from the ABI; one that cannot puts ``prose()`` in its prompt and finds
the command with ``pattern()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from zeos.core.ids import PipeName
from zeos.machine.base import NAMES_A_TARGET, MachineRequest, OpKind, tokens_from_text

__all__ = ["DEFAULT", "SyscallABI", "Verb"]


@dataclass(frozen=True, slots=True)
class Verb:
    """One command word, the kernel operation it asks for, and the arguments it takes."""

    name: str
    op: OpKind = OpKind.NONE
    pipe: bool = False
    text: bool = False
    doc: str = ""

    def signature(self, terminator: str) -> str:
        argument = "<name>" if self.op in NAMES_A_TARGET else "<text>"
        parts = [self.name] + (["<pipe>"] if self.pipe else []) + ([argument] if self.text else [])
        return " ".join(parts) + terminator


@dataclass(frozen=True, slots=True)
class SyscallABI:
    """The command vocabulary one case's jobs speak.

    ``aliases`` are the pipe names a command may use; the kernel resolves them against the
    descriptor's ``pipes:`` bindings. ``max_text`` bounds a payload, or is None for no
    bound; short on purpose in the default, since a roomy payload lets one command carry
    a whole plan.
    """

    verbs: tuple[Verb, ...]
    aliases: tuple[str, ...] = ("stdin", "stdout", "tools")
    terminator: str = ";"
    max_text: int | None = 16

    def __post_init__(self) -> None:
        if not self.verbs:
            raise ValueError("an ABI needs at least one verb")
        names = [v.name.lower() for v in self.verbs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate verbs: {sorted(n for n in names if names.count(n) > 1)}")
        if not self.terminator or self.terminator.isspace():
            raise ValueError("the terminator must be a visible character")
        if self.max_text is not None and self.max_text < 1:
            raise ValueError("max_text must be at least 1, or None")
        named = [v.name for v in self.verbs if v.op in NAMES_A_TARGET and v.pipe]
        if named:
            raise ValueError(f"these verbs name a target, so they take no pipe: {sorted(named)}")

    def verb(self, name: str) -> Verb | None:
        wanted = name.lower()
        return next((v for v in self.verbs if v.name.lower() == wanted), None)

    @property
    def lines(self) -> tuple[Verb, ...]:
        """Verbs that ask the kernel for nothing; a job may issue any number before a call."""
        return tuple(v for v in self.verbs if v.op is OpKind.NONE)

    @property
    def calls(self) -> tuple[Verb, ...]:
        """Verbs that ask the kernel for something and so end a job's round."""
        return tuple(v for v in self.verbs if v.op is not OpKind.NONE)

    def parse(self, line: str) -> MachineRequest:
        """The request one completed command asks for. An undeclared verb, a pipe verb with
        no pipe, or a naming verb with no name is a ``MALFORMED`` request carrying the words."""
        command = line.strip().rstrip(self.terminator).strip()
        head, _, rest = command.partition(" ")
        verb = self.verb(head)
        if verb is None:
            return MachineRequest(op=OpKind.MALFORMED, text=command)
        if verb.op is OpKind.NONE:
            return MachineRequest()
        rest = rest.strip()
        pipe: str | None = None
        if verb.pipe:
            pipe, _, rest = rest.partition(" ")
            if not pipe:
                return MachineRequest(op=OpKind.MALFORMED, text=command)
        rest = rest.strip()
        if verb.op in NAMES_A_TARGET:
            # The kernel resolves this argument as a name, so it travels in ``text``;
            # a payload of tokens is not something it can look up.
            if not rest:
                return MachineRequest(op=OpKind.MALFORMED, text=command)
            return MachineRequest(op=verb.op, text=rest)
        payload = tokens_from_text(rest) if verb.text else ()
        return MachineRequest(
            op=verb.op, pipe=PipeName(pipe) if pipe is not None else None, payload=payload
        )

    def pattern(self) -> re.Pattern[str]:
        """One command anywhere in free text, for a reply that could not be constrained."""
        names = "|".join(re.escape(v.name) for v in self.verbs)
        end = re.escape(self.terminator)
        return re.compile(rf"\b({names})\b([^{end}]*){end}", re.IGNORECASE)

    def prose(self) -> str:
        """The vocabulary as a model is told it, one indented line per verb."""
        width = max(len(v.signature(self.terminator)) for v in self.verbs) + 4
        return "\n".join(f"    {v.signature(self.terminator):<{width}}{v.doc}" for v in self.verbs)


#: An example vocabulary, and what a seat speaks unless told otherwise.
DEFAULT = SyscallABI(
    verbs=(
        Verb("say", text=True, doc="think out loud. No effect, and nobody reads it"),
        Verb(
            "write",
            OpKind.WRITE,
            pipe=True,
            text=True,
            doc="put text on a pipe. This is how anything happens",
        ),
        Verb("read", OpKind.READ, pipe=True, doc="sleep until something arrives on that pipe"),
        Verb("exit", OpKind.EXIT, doc="finish"),
    ),
)
