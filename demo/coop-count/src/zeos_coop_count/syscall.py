# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The syscall ABI: a tiny semicolon-terminated language a model uses to ask the kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from zeos.core.ids import PipeName
from zeos.machine.base import MachineRequest, OpKind, tokens_from_text

__all__ = ["ALIASES", "build_grammar", "parse_line", "SyscallParser"]

#: All pipe aliases the kernel can resolve; a descriptor's grammar offers only its own.
ALIASES = ("stdin", "stdout", "tools")

#: Longest a single ``say`` or ``write`` payload may run, bounded in the grammar itself.
#: Short on purpose: a roomy payload lets one command carry a whole plan, and a plan is
#: several steps inside what is meant to be one.
MAX_TEXT = 16


def build_grammar(pipes: Sequence[str], *, valued: Sequence[str] = ()) -> str:
    """GBNF for one descriptor, given the pipe aliases it may name."""
    if not pipes:
        # A job with no pipes can still think and exit, but it cannot do any I/O.
        alternatives = ""
        call = "exit"
    else:
        alternatives = " | ".join(f'"{p}"' for p in pipes)
        # A job with no `stdin` gets no `read`, so it cannot block on a pipe it never reads.
        call = "write | read | exit" if "stdin" in pipes else "write | exit"
    lines = [
        "root  ::= line* call",
        'line  ::= "say " text end',
        f"call  ::= {call}",
        'exit  ::= "exit" end',
    ]
    if pipes:
        lines += ['write ::= "write " pipe " " text end']
        if "stdin" in pipes:
            # ``pipe`` is used by ``read`` alone, so it goes away when reading does.
            lines += ['read  ::= "read " pipe end', f"pipe  ::= {alternatives}"]
    plain = [p for p in pipes if p not in set(valued)]
    if pipes:
        # A write to a valued pipe carries a number; a write to any other carries a message.
        parts: list[str] = []
        if plain:
            parts.append('"write " plain " " text end')
        if any(p in set(valued) for p in pipes):
            parts.append('"write " valued " " number end')
        lines = [line for line in lines if not line.startswith("write ")]
        lines.append("write ::= " + " | ".join(parts))
        if plain:
            lines.append("plain ::= " + " | ".join(f'"{p}"' for p in plain))
        if any(p in set(valued) for p in pipes):
            lines.append("valued ::= " + " | ".join(f'"{p}"' for p in pipes if p in set(valued)))
            # No leading zeros, so a job cannot write something like ``000000000``.
            lines.append('number ::= "0" | [1-9] [0-9]{0,8}')
    lines += [
        f"text  ::= [^;<\\n]{{1,{MAX_TEXT}}}",
        'end   ::= "; "',
    ]
    return "\n".join(lines) + "\n"


#: What terminates a command; a semicolon, because whitespace splitting eats newlines.
TERMINATOR = ";"


def parse_line(line: str) -> MachineRequest:
    """Turn one completed command into at most one kernel request."""
    head, _, rest = line.strip().partition(" ")
    match head:
        case "write":
            pipe, _, payload = rest.partition(" ")
            if not pipe:
                return MachineRequest()
            return MachineRequest(
                op=OpKind.WRITE,
                pipe=PipeName(pipe),
                payload=tokens_from_text(payload),
            )
        case "read":
            return (
                MachineRequest(op=OpKind.READ, pipe=PipeName(rest.strip()))
                if rest.strip()
                else MachineRequest()
            )
        case "exit":
            return MachineRequest(op=OpKind.EXIT)
        case _:
            return MachineRequest()


@dataclass
class SyscallParser:
    """Collects decoded pieces and yields a request on the step that closes a command."""

    buffer: str = ""
    #: Every completed command, in order, kept for the transcript the CLI prints.
    lines: list[str] = field(default_factory=list[str])

    def feed(self, piece: str) -> MachineRequest:
        self.buffer += piece
        if TERMINATOR not in self.buffer:
            return MachineRequest()
        line, _, rest = self.buffer.partition(TERMINATOR)
        self.buffer = rest
        line = line.strip()
        self.lines.append(line)
        return parse_line(line)

    def reset(self) -> None:
        self.buffer = ""
