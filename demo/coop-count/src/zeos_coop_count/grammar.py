# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The syscall ABI rendered as a GBNF grammar, so a llama.cpp sampler can only emit commands."""

from __future__ import annotations

from collections.abc import Sequence

from zeos.machine.abi import SyscallABI

__all__ = ["build_grammar"]


def _alternatives(names: Sequence[str]) -> str:
    return " | ".join(f'"{n}"' for n in names)


def build_grammar(abi: SyscallABI, pipes: Sequence[str], *, valued: Sequence[str] = ()) -> str:
    """GBNF for one descriptor, given the pipe aliases it may name.

    One round is any number of the ABI's request-free verbs, then one call. A call that
    takes a pipe is offered only the aliases this descriptor binds, and a job with no
    ``stdin`` is offered no read at all, so it cannot block on a pipe it never reads.
    """
    valued = [p for p in pipes if p in set(valued)]
    plain = [p for p in pipes if p not in set(valued)]
    readable = list(pipes) if "stdin" in pipes else []

    def payload(verb_name: str, verb_pipe: bool, verb_text: bool) -> list[str]:
        if not verb_pipe:
            return [f'"{verb_name} " text end' if verb_text else f'"{verb_name}" end']
        if verb_text:
            parts: list[str] = []
            if plain:
                parts.append(f'"{verb_name} " plain " " text end')
            if valued:
                parts.append(f'"{verb_name} " valued " " number end')
            return parts
        return [f'"{verb_name} " readable end'] if readable else []

    lines: list[str] = []
    calls: list[str] = []
    for verb in abi.lines:
        lines += [f"{verb.name} ::= " + " | ".join(payload(verb.name, verb.pipe, verb.text))]
    for verb in abi.calls:
        alternatives = payload(verb.name, verb.pipe, verb.text)
        if alternatives:
            calls.append(verb.name)
            lines.append(f"{verb.name} ::= " + " | ".join(alternatives))
    if not calls:
        raise ValueError(f"no call of the ABI can be made with pipes {list(pipes)}")

    rules = [
        "root  ::= line* call" if abi.lines else "root  ::= call",
        *(["line  ::= " + " | ".join(v.name for v in abi.lines)] if abi.lines else []),
        "call  ::= " + " | ".join(calls),
        *lines,
    ]
    if plain:
        rules.append("plain ::= " + _alternatives(plain))
    if valued:
        rules.append("valued ::= " + _alternatives(valued))
        # No leading zeros, so a job cannot write something like ``000000000``.
        rules.append('number ::= "0" | [1-9] [0-9]{0,8}')
    if readable:
        rules.append("readable ::= " + _alternatives(readable))
    bound = f"{{1,{abi.max_text}}}" if abi.max_text is not None else "+"
    rules += [
        f"text  ::= [^{abi.terminator}<\\n]{bound}",
        f'end   ::= "{abi.terminator} "',
    ]
    return "\n".join(rules) + "\n"
