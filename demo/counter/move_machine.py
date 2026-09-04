# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A machine backend whose decode step asks a model for its next move.

Everything structural stays inherited from ``ScriptedMachine`` -- blocks, padding,
masks, injection, transcripts -- so the kernel above notices no difference. Only
token *generation* changes: each decode shows the model its job's transcript and
maps the one-move reply onto the same shapes a scripted step would produce -- some
emitted tokens, or a spawn/exit request.

The protocol is one move per decode, mirroring one step per decode in the scripted
backend. Control moves do not appear in the transcript, and the transcript renders
flat, so every move made is fed back into the prompt.

*Which* model answers is the subclass's whole job: ``_ask`` takes the rendered
prompt and returns the move. ``claude_machine`` asks the Claude API;
``qwen_machine`` asks a local llama.cpp model. A run of any of them is not
deterministic in the way the tape is; the determinism gate never applies here.
"""

from __future__ import annotations

from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import (
    AttentionHint,
    DecodeResult,
    MachineRequest,
    OpKind,
    Token,
    tokens_from_text,
)
from zeos.machine.scripted import ScriptedMachine

__all__ = ["SYSTEM", "MoveMachine"]

SYSTEM = (
    "You are a job running under ZEOS, a transformer operating system. Your context "
    "so far is given each turn; it begins with your goal. Reply with exactly one "
    "move and nothing else:\n"
    "- the next tokens of your output (usually one word or number),\n"
    "- SPAWN to hand on to a fresh instance of yourself,\n"
    "- EXIT to finish and terminate.\n"
    "Your context renders as flat text and control moves are not written into it, "
    "so your moves so far are listed separately; trust that list for what you have "
    "already done."
)


class MoveMachine(ScriptedMachine):
    """``ScriptedMachine`` with the tape replaced by a model's one-move replies."""

    def __init__(self, *, block_size: int = 16) -> None:
        super().__init__(scripts=None, block_size=block_size)
        self._descriptor: dict[JobId, str] = {}
        self._moves: dict[JobId, list[str]] = {}

    def _ask(self, job: JobId, prompt: str) -> str:
        """Return the model's next move for ``prompt``. The subclass's whole job."""
        raise NotImplementedError

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        super().create_context(job, descriptor)
        self._descriptor[job] = descriptor
        self._moves[job] = []

    def destroy_context(self, job: JobId) -> None:
        super().destroy_context(job)
        self._descriptor.pop(job, None)
        self._moves.pop(job, None)

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        context = " ".join(t.text for t in ctx.tokens if t.kind is TokenKind.NORMAL)
        made = "; ".join(self._moves[job]) or "none"
        prompt = f"Context:\n{context}\n\nYour moves so far: {made}\n\nYour next move:"
        move = self._ask(job, prompt)

        tokens: tuple[Token, ...] = ()
        request = MachineRequest()
        if move.upper() == "SPAWN":
            request = MachineRequest(op=OpKind.SPAWN, text=self._descriptor[job])
        elif move.upper() == "EXIT":
            request = MachineRequest(op=OpKind.EXIT)
        else:
            tokens = tokens_from_text(move)
        # Every move, emissions included: a control move leaves no trace in the
        # transcript, and the transcript itself renders flat, so this list is the
        # only account a job has of what it has already done.
        self._moves[job].append(move if request.op is not OpKind.NONE else f"SAID {move}")

        before = len(ctx.tokens)
        ctx.tokens.extend(tokens)
        after = len(ctx.tokens)
        return DecodeResult(
            tokens=tokens,
            request=request,
            attention=None,
            attention_hint=AttentionHint(),
            at_block_boundary=after % self.block_size == 0 and after != before,
        )
