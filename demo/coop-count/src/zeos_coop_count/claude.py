# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A command source that answers kernel syscalls with the Claude API instead of local weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from anthropic import Anthropic

from zeos.machine.abi import DEFAULT, SyscallABI
from zeos.machine.seat import Turn

__all__ = ["ClaudeSource", "DEFAULT_MODEL", "one_command", "prompt_for", "system_for"]

DEFAULT_MODEL = "claude-opus-5"

#: Effort the seat asks for. High is the only level every model works at: opus is
#: exact either way, and sonnet loses the thread of an interrupted turn at medium.
EFFORT = "high"

#: The syscall ABI written as prose, since an API seat cannot enforce a grammar.
SYSTEM = """\
You are a job running under ZEOS, a transformer operating system. Your context is your
own: it opens with your goal, and everything after it is either something you said or
something that arrived on a pipe.

Reply with exactly ONE command and nothing else. Every command ends with `{terminator}`.

{commands}

The only pipes you may name are: {aliases}. Naming any other is refused by the kernel.
{valued}
A `read` costs you the machine: the kernel runs another job until your pipe has
something, and what arrives appears in your context. That is how you wait -- there is
no other way, and waiting costs nothing while you do.

Say nothing except the one command. No explanation, no formatting, no quotes.\
"""

_VALUED = "These pipes carry a value rather than prose, so write them a bare number: {names}.\n"

#: The turn that asks for the next command, so the prompt does not end on an assistant turn.
_ASK = "You have the machine. Your next command:"


def system_for(aliases: Sequence[str], valued: Sequence[str], *, abi: SyscallABI = DEFAULT) -> str:
    """The ABI as one descriptor sees it: only the pipes it binds, and which carry a number."""
    return SYSTEM.format(
        terminator=abi.terminator,
        commands=abi.prose(),
        aliases=", ".join(aliases) or "none",
        valued=_VALUED.format(names=", ".join(valued)) if valued else "",
    )


#: What the job has already done, said plainly rather than left to be read out of the
#: transcript. Without it a model reissues the command it has just issued: the goal's
#: own closing line ("your next command is `say 1;`") stays literally true-looking for
#: ever, and its own working is words in the same undifferentiated run of text.
_DONE = "Your last command was `{last}{terminator}` and it is done. Do not issue it again."


def prompt_for(turn: Turn, *, abi: SyscallABI = DEFAULT) -> str:
    done = f"{_DONE.format(last=turn.last, terminator=abi.terminator)}\n\n" if turn.last else ""
    return f"{turn.transcript}\n\n{done}{_ASK}"


def one_command(text: str, *, abi: SyscallABI = DEFAULT) -> str:
    """The single command in a reply, or the whole reply as spoken if it holds none."""
    match = abi.pattern().search(text)
    if match is None:
        return " ".join(text.split()) or "nothing"
    return f"{match.group(1).lower()}{match.group(2).rstrip()}"


class ClaudeSource:
    """A command source that asks the Claude API for each command."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        abi: SyscallABI = DEFAULT,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        valued: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        # One call per command, so a long run makes many and needs more than two retries.
        self._client = Anthropic(max_retries=6)
        self._model = model
        self._abi = abi
        self._aliases: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (descriptors or {}).items()
        }
        self._valued: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in (valued or {}).items()}

    def system(self, descriptor: str) -> str:
        return system_for(
            self._aliases.get(descriptor, self._abi.aliases),
            self._valued.get(descriptor, ()),
            abi=self._abi,
        )

    def next_command(self, turn: Turn) -> str:
        """One API call, giving one command."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt_for(turn, abi=self._abi)}
        ]
        # Server-side fallback is offered on some models and refused with a 400 on
        # others, so it is asked for only on the default this seat was written against.
        served = (
            {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
            if self._model == DEFAULT_MODEL
            else {}
        )
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=4096,
            # Lower effort makes the model keep counting where it should hand over and sleep.
            output_config={"effort": EFFORT},
            system=self.system(turn.descriptor),
            messages=messages,
            **served,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"the model refused to produce a command for job {turn.job}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(
                f"no text in the response for job {turn.job} (stop_reason={response.stop_reason})"
            )
        return one_command(text.strip(), abi=self._abi)
