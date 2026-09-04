# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What every machine seat shares: the syscall ABI, the pipe aliases, and the callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import (
    AttentionHint,
    ControlTokenViolation,
    DecodeResult,
    MachineRequest,
    OpKind,
    Token,
)
from zeos.machine.scripted import ScriptedMachine
from zeos_coop_count.syscall import ALIASES, TERMINATOR, SyscallParser

__all__ = ["CommandSeat", "CommandSource", "SyscallSeat", "Turn", "words_of"]


def words_of(command: str, *, lead: bool) -> list[str]:
    """Split one command into the words that carry it out, one per decode.

    Every word but the first carries a leading space, because the transcript is
    rebuilt by concatenation and the first token of a context has nothing to follow.
    """
    words = command.split()
    if not words[-1].endswith(TERMINATOR):
        words[-1] += TERMINATOR
    return [w if (i == 0 and not lead) else " " + w for i, w in enumerate(words)]


class SyscallSeat:
    """The ABI half of a machine backend, leaving the model half to the subclass."""

    def __init__(
        self,
        *,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        valued: Mapping[str, Sequence[str]] | None = None,
        on_command: Callable[[JobId, str, MachineRequest], None] | None = None,
        on_arrival: Callable[[JobId, str], None] | None = None,
    ) -> None:
        #: Maps a descriptor name to the pipe aliases it binds.
        self._descriptors: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (descriptors or {}).items()
        }
        #: Per descriptor, the aliases whose payload is a number rather than prose.
        self._valued: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in (valued or {}).items()}
        self._on_command = on_command
        self._on_arrival = on_arrival

    # -- what a descriptor may name ------------------------------------------

    def aliases(self, descriptor: str) -> tuple[str, ...]:
        """The pipe aliases this descriptor binds, or all of them if the seat knows none."""
        return self._descriptors.get(descriptor, ALIASES)

    def valued(self, descriptor: str) -> tuple[str, ...]:
        return self._valued.get(descriptor, ())

    # -- the ABI --------------------------------------------------------------

    def consume(self, job: JobId, parser: SyscallParser, piece: str) -> MachineRequest:
        """Feed one decoded piece in, and report the request if it closed a command."""
        closed_before = len(parser.lines)
        request = parser.feed(piece)
        if len(parser.lines) > closed_before and self._on_command is not None:
            self._on_command(job, parser.lines[-1], request)
        return request

    def note_arrival(self, job: JobId, text: str) -> None:
        if self._on_arrival is not None:
            self._on_arrival(job, text)


@dataclass(frozen=True, slots=True)
class Turn:
    """Everything a source is told about the job whose turn it is to speak."""

    job: JobId
    #: The descriptor this job runs, which is what decides its pipes and its prose.
    descriptor: str
    #: The job's context as text, kernel control tokens dropped: goal, working, arrivals.
    transcript: str
    #: How many commands this job has already completed, so a source that plays a tape
    #: needs no per-job state of its own.
    issued: int
    #: The last command this job completed, or empty before it has issued one. The
    #: transcript holds it too, but only as words among the goal's own words, and a
    #: model that cannot pick it out reissues it.
    last: str = ""


class CommandSource(Protocol):
    """Where a seat's next command comes from. One method, and no kernel to speak of."""

    def next_command(self, turn: Turn) -> str:
        """One complete syscall command, such as ``say 41`` or ``write tools 50``."""
        ...


@dataclass
class _State:
    """What a seat knows about a job that ``ScriptedMachine`` does not."""

    descriptor: str
    parser: SyscallParser = field(default_factory=SyscallParser)
    #: Words of the current command not yet given to the kernel; one leaves per decode.
    pending: list[str] = field(default_factory=list[str])
    tags: tuple[str, ...] = ("descriptor",)
    spoken: bool = False


class CommandSeat(ScriptedMachine, SyscallSeat):
    """A machine backend that spends one decode per word of one command.

    The seat is the whole machine half: the token bookkeeping, the parser that turns a
    closed command into a kernel request, the arrival notices, and the transcript the
    next command is chosen from. What it does not decide is what the job says next --
    that comes from a ``CommandSource``, so a tape, an API and a local process are three
    collaborators rather than three subclasses, and the same case runs under any of them.
    """

    def __init__(
        self,
        *,
        source: CommandSource,
        block_size: int = 16,
        on_command: Callable[[JobId, str, MachineRequest], None] | None = None,
        on_arrival: Callable[[JobId, str], None] | None = None,
    ) -> None:
        ScriptedMachine.__init__(self, scripts=None, block_size=block_size)
        SyscallSeat.__init__(self, on_command=on_command, on_arrival=on_arrival)
        self._source = source
        self._state: dict[JobId, _State] = {}

    # -- lifecycle -----------------------------------------------------------

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        super().create_context(job, descriptor)
        self._state[job] = _State(descriptor=descriptor)

    def destroy_context(self, job: JobId) -> None:
        super().destroy_context(job)
        self._state.pop(job, None)

    def _state_of(self, job: JobId) -> _State:
        state = self._state.get(job)
        if state is None:
            raise KeyError(f"no context for job {job}; create_context first")
        return state

    # -- the ops that differ from a tape -------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        state = self._state_of(job)
        if not state.pending:
            turn = Turn(
                job=job,
                descriptor=state.descriptor,
                transcript=self.render(job),
                issued=len(state.parser.lines),
                last=state.parser.lines[-1] if state.parser.lines else "",
            )
            state.pending = words_of(self._source.next_command(turn), lead=state.spoken)

        piece = state.pending.pop(0)
        token = Token(piece, TokenKind.NORMAL)
        if not allow_control and token.kind is TokenKind.CONTROL:
            raise ControlTokenViolation(
                f"job {job} produced a CONTROL token while control tokens were disabled"
            )
        state.spoken = True

        before = len(ctx.tokens)
        ctx.tokens.append(token)
        after = len(ctx.tokens)

        request = self.consume(job, state.parser, piece)
        if request.op is not OpKind.NONE:
            # The round is over, so anything the source added past the command is dropped.
            state.pending.clear()
            state.parser.reset()

        return DecodeResult(
            tokens=(token,),
            request=request,
            # No source here can measure attention, so only the hint below is available.
            attention=None,
            attention_hint=AttentionHint(tags=state.tags),
            at_block_boundary=after % self.block_size == 0 and after != before,
        )

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        start, end = super().inject(job, tokens)
        state = self._state_of(job)
        text = " ".join(t.text for t in tokens if t.kind is TokenKind.NORMAL)
        if text:
            # Only reported once the job has spoken; before that this is still the prompt.
            if state.spoken:
                self.note_arrival(job, text)
            # Foreign content is the most likely thing the next step attends to.
            state.tags = ("self",)
        return start, end

    def fork(self, parent: JobId, child: JobId) -> int:
        n = super().fork(parent, child)
        src = self._state_of(parent)
        if child not in self._state:
            self._state[child] = _State(descriptor=src.descriptor)
        dst = self._state_of(child)
        dst.spoken = src.spoken
        dst.tags = src.tags
        return n

    # -- outside the Protocol -------------------------------------------------

    def lines(self, job: JobId) -> tuple[str, ...]:
        """The commands this job has issued, for display only."""
        return tuple(self._state_of(job).parser.lines)

    def close(self) -> None:
        self._state.clear()

    def render(self, job: JobId) -> str:
        """The transcript as a source reads it, with kernel control tokens dropped.

        Rendered fresh every call rather than kept as a conversation alongside it, because
        the kernel rewrites status regions and evicts spans in place and a copy drifts.
        """
        return "".join(
            t.text if t.text.startswith(" ") else " " + t.text
            for t in self.transcript(job)
            if t.kind is TokenKind.NORMAL
        ).strip()
