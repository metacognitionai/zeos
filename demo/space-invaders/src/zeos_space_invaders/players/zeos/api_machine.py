# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A ``MachineBackend`` that decodes one word at a time off a streaming chat API.

* ZEOS's unit is one token (ZEOS-AM §6.1); a chat API's unit is a whole
  generation. The generation runs on its own thread and ``decode`` takes one word
  off its queue, returning an empty step when the queue is dry so the job stays
  RUNNING and preemptible while the model thinks.
* The transcript ``T`` is the source of truth: every request is rebuilt from it.
  The mask is enforced by not transmitting, and a context change of any kind is
  expressed by asking again.
* Reasoning enters ``T`` and is accounted -- budget, token clock, boundaries --
  but is marked ``THINK`` and never transmitted.
* The syscall extractor is fed only by the content channel, so a request can
  never be read out of reasoning.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import JobId, PipeName, TokenKind
from zeos.machine.base import (
    AttentionHint,
    ContextStats,
    ControlTokenViolation,
    DecodeResult,
    MachineRequest,
    OpKind,
    SpliceResult,
    Token,
    tokens_from_text,
)

__all__ = [
    "ALIASES",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_STALL_S",
    "FORMATS",
    "PAD_TOKEN",
    "VERBS",
    "APIMachineBase",
    "Native",
    "Piece",
    "parse_syscall",
]


#: Reserved no-op used for block padding. CONTROL because padding is kernel framing
#: a job must not be able to emit (ZEOS-AM §9); it never reaches the wire.
PAD_TOKEN = Token("<pad>", TokenKind.CONTROL)

DEFAULT_BLOCK_SIZE = 16

#: How long a dry ``decode`` waits before handing the kernel an empty step. It
#: bounds how soon an interrupt can displace a stalled job and how much wall clock
#: a driver burns per tick; machine-side, because the kernel has no clock.
DEFAULT_STALL_S = 0.001

#: Where an element of ``T`` came from. ``IN`` and ``OUT`` rebuild the chat roles;
#: ``PAD``, ``THINK`` and ``VOID`` are counted by ``stats`` and never transmitted.
#: ``VOID`` is output of a cancelled generation that completed no syscall.
IN, OUT, PAD, THINK, VOID = "in", "out", "pad", "think", "void"

#: What to do with the tail of a cancelled generation. ``syscall`` voids everything
#: after the last completed syscall (default for ``json``: a fragment of a call
#: carries nothing and re-sending it teaches broken documents); ``keep`` voids
#: nothing (default for ``text``: a half-finished plan is worth handing back);
#: ``drop`` voids everything the generation produced.
PARTIAL = ("syscall", "keep", "drop")

#: The pipe aliases ``Kernel._resolve_pipe`` resolves against a descriptor's own
#: ``pipes:`` block; anything else reaches the capability check as a literal name.
ALIASES = ("stdin", "stdout", "tools")

#: Where a turn goes to sleep; the model is never told the pipe exists.
STDIN = "stdin"

#: Where a move is actuated. The alias, not the pipe: the kernel resolves it and
#: the capability check still runs afterwards.
STDOUT = "stdout"

#: How a request is extracted from what the model produced. ``json`` is the default
#: because it is the only format that is both enforced and widely served.
FORMATS = ("json", "text")

#: The verbs the ``text`` ABI defines. A clause begins at one of these and ends at
#: the next semicolon; ``text`` has no grammar, so "everything since the last
#: semicolon" would hand ``parse_syscall`` the model's prose as a clause.
VERBS = ("read", "write", "exit")

#: Characters that end a decoded element, per format, so that no decode step can
#: carry two syscalls inside one scheduling quantum. ``json`` breaks on
#: punctuation because a compact object contains no whitespace.
_TERMINATORS = {"text": ";", "json": "{}[],"}


def parse_syscall(clause: str) -> MachineRequest:
    """The request a ``text``-format clause asks for, or ``NONE``.

    Semicolon-terminated rather than newline-terminated because the kernel loads
    a descriptor body through ``tokens_from_text``, which splits on whitespace: a
    model never sees a newline in its own context and does not produce one.
    """
    body = clause.strip().rstrip(";").strip()
    if not body:
        return MachineRequest()
    verb, _, rest = body.partition(" ")
    alias, _, arg = rest.strip().partition(" ")
    return build_request(verb, alias, arg)


def build_request(op: str, pipe: str, text: str = "") -> MachineRequest:
    """One request, whatever format it was expressed in.

    Aliases pass through unresolved: a pipe the job was not granted is refused by
    the kernel's capability check, a fault a parser must not quietly drop.
    """
    if op == "exit":
        return MachineRequest(op=OpKind.EXIT)
    if not pipe:
        return MachineRequest()
    if op == "read":
        return MachineRequest(op=OpKind.READ, pipe=PipeName(pipe))
    if op == "write":
        return MachineRequest(
            op=OpKind.WRITE, pipe=PipeName(pipe), payload=tokens_from_text(text)
        )
    return MachineRequest()


@dataclass(frozen=True, slots=True)
class Native:
    """What a locally-served behaviour is given when the kernel decodes it.

    A reflex must not cost a forward pass, and the kernel cannot tell the
    difference: a native behaviour is decoded through ``decode``, appends to
    ``T``, and is scheduled, preempted and charged exactly as a served job is.
    """

    job: JobId
    descriptor: str
    #: How many times this behaviour has been decoded; its program counter.
    step: int
    #: Everything the kernel has handed this job since its last step, recorded at
    #: ``inject`` because a mark held at the context length breaks under the pager.
    arrived: str


@dataclass(frozen=True, slots=True)
class Piece:
    """One thing off the wire, typed so a syscall is never read out of reasoning."""

    #: ``text`` | ``thinking``
    kind: str
    text: str


@dataclass
class _Generation:
    """One completion in flight, on its own thread.

    The queue is the only thing that crosses between the two threads, and only
    the producer may close the stream, so ``stop`` is a request rather than an
    order. Everything below ``error`` is consumer-side state only.
    """

    #: The mask the request was built under.
    mask: frozenset[int] | None = None
    q: queue.SimpleQueue[Piece | None] = field(default_factory=queue.SimpleQueue)
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    #: The producer has put its sentinel; nothing more is coming.
    done: bool = False
    #: Raised on the kernel's thread by the next ``decode``, where it reaches somebody.
    error: BaseException | None = None

    #: Text received on each channel but not yet a whole element.
    carry: dict[str, str] = field(default_factory=dict)
    #: Whole ``(origin, text, channel)`` elements waiting, one per decode step.
    ready: deque[tuple[str, str, str]] = field(default_factory=deque)
    #: Elements this turn decoded on the content channel.
    produced: int = 0

    #: Wall clock when the request went out: the one clock this backend reads, and
    #: it never reaches the kernel.
    opened: float = 0.0
    #: The vendor's stream object, published by ``_stream`` for ``_abort`` to close.
    wire: object = None
    #: Wall clock when the first piece came off the wire: the server's share.
    first_at: float = 0.0

    #: ``json``: everything the content channel has produced, and how far the
    #: object scanner has read it.
    buf: str = ""
    scan: int = 0
    depth: int = 0
    obj_start: int | None = None
    in_string: bool = False
    escaped: bool = False

    #: The channel the last piece came in on; a switch is an element boundary.
    channel: str | None = None

    #: ``|T|`` when this generation opened, and ``|T|`` just after the element that
    #: completed its most recent syscall; the ``syscall`` policy voids from there.
    first_offset: int = 0
    committed: int = 0


@dataclass
class _Context:
    """Per-job state: the token sequence and blocks the kernel meets at offsets."""

    descriptor: str
    tokens: list[Token] = field(default_factory=list)
    #: Parallel to ``tokens``: IN, OUT, PAD, THINK or VOID, one per element.
    origin: list[str] = field(default_factory=list)
    #: ``None`` is ⊥, not ∅; collapsing them is a privilege escalation (ZEOS-AM §8.2).
    mask: frozenset[int] | None = None
    gen: _Generation | None = None
    #: ``text`` format only: the clause being accumulated.
    clause: str = ""
    #: Natively-served jobs only; see ``Native``.
    step: int = 0
    arrived: list[Token] = field(default_factory=list)


class APIMachineBase:
    """A ``MachineBackend`` over a streaming chat API, vendor-agnostic.

    A subclass implements ``_request`` and ``_stream`` and nothing else, so a
    second vendor is a wire format and a continuation policy, not a second machine.
    """

    def __init__(
        self,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        stall_s: float = DEFAULT_STALL_S,
        max_tokens: int = 512,
        syscall_format: str = "json",
        partial: str | None = None,
        actions: Sequence[str] = (),
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if syscall_format not in FORMATS:
            raise ValueError(
                f"unknown syscall_format {syscall_format!r}; "
                f"expected one of {', '.join(FORMATS)}"
            )
        self._block_size = block_size
        self._stall_s = stall_s
        self._max_tokens = max_tokens
        self.syscall_format = syscall_format
        #: Defaulted from the format because the right answer differs per format.
        self.partial = partial or ("keep" if syscall_format == "text" else "syscall")
        if self.partial not in PARTIAL:
            raise ValueError(
                f"unknown partial policy {self.partial!r}; "
                f"expected one of {', '.join(PARTIAL)}"
            )
        #: The moves a reply may name -- the game's knowledge, not the machine's.
        #: Empty means the payload is unconstrained.
        self.actions = tuple(actions)
        self._ctx: dict[JobId, _Context] = {}
        #: Descriptors served by local code rather than by the API. See ``Native``.
        self._behaviours: dict[str, Callable[[Native], DecodeResult]] = {}
        #: Cancelled generations whose producer has not noticed yet; never joined
        #: from ``decode``, so the next question does not queue behind one.
        self._graveyard: list[_Generation] = []
        self.generations = 0
        self.cancellations = 0
        self.words = 0
        self.roundtrips: list[float] = []
        self.ttfts: list[float] = []
        self.last_roundtrip: float | None = None
        self.thinking_words = 0
        #: Kept apart from ``words`` because a reflex reaches no model at all.
        self.native_words = 0
        self.voided = 0

    # -- locally-served behaviours -------------------------------------------

    def register_behaviour(
        self, descriptor: str, behaviour: Callable[[Native], DecodeResult]
    ) -> None:
        """Serve ``descriptor`` with local code instead of a forward pass."""
        self._behaviours[descriptor] = behaviour

    def _decode_native(self, job: JobId, ctx: _Context) -> DecodeResult:
        """One step of a locally-served behaviour.

        The CONTROL check has to be made here rather than reasoned about: ``_runs``
        drops framing by origin, not by kind, so a CONTROL token carrying origin
        OUT would put the kernel's own framing on the wire (ZEOS-AM §9).
        """
        arrived = " ".join(
            t.text for t in ctx.arrived if t.kind is not TokenKind.CONTROL
        )
        ctx.arrived.clear()
        result = self._behaviours[ctx.descriptor](
            Native(job=job, descriptor=ctx.descriptor, step=ctx.step, arrived=arrived)
        )
        ctx.step += 1
        if any(t.kind is TokenKind.CONTROL for t in result.tokens):
            raise ControlTokenViolation(
                f"native behaviour {ctx.descriptor!r} returned a CONTROL token "
                "while control tokens were disabled for this step (ZEOS-AM §9)"
            )
        if result.tokens:
            ctx.tokens.extend(result.tokens)
            ctx.origin.extend([OUT] * len(result.tokens))
            self.native_words += len(result.tokens)
        return result

    # -- the vendor seam -----------------------------------------------------

    def _request(self, ctx: _Context, *, continuing: bool) -> object:
        """Build the vendor's request. Runs on the kernel's thread.

        ``ctx`` is shared mutable state a producer thread must never read, so what
        crosses over is an immutable request. ``continuing`` is true when the
        transcript ends with a run the job itself decoded; vendors disagree about
        what that means, so the subclass decides.
        """
        raise NotImplementedError

    def _stream(
        self, request: object, descriptor: str, gen: _Generation
    ) -> Iterator[Piece]:
        """Issue the request and yield ``Piece``s. Runs on the producer thread.

        The generator owns the HTTP stream, closes it on exit, and must publish it
        as ``gen.wire`` while open so that ``_abort`` can shut the connection from
        the kernel's thread. It must not touch the context (see ``_request``).
        """
        raise NotImplementedError

    # -- what the vendor should ask the server to enforce --------------------

    def syscall_schema(self) -> dict[str, object]:
        """The strict JSON schema for the ``json`` format: one move and nothing else.

        ``move`` is an enum over the descriptor's vocabulary, so a reply naming no
        move is unrepresentable. The op and pipe are supplied by the machine
        because each has one legal value; there is no array because a server that
        does not enforce ``minItems`` makes an empty array a legal no-op reply.
        """
        return {
            "name": "move",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"move": {"enum": list(self.actions)}},
                "required": ["move"],
                "additionalProperties": False,
            },
        }

    # -- lifecycle -----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        self._ctx[job] = _Context(descriptor=descriptor)

    def destroy_context(self, job: JobId) -> None:
        ctx = self._ctx.pop(job, None)
        if ctx is not None:
            self._cancel(ctx)

    def close(self, timeout: float = 1.0) -> None:
        """Stop every producer and wait briefly for the threads to notice.

        Bounded, because a producer blocked on a socket cannot be made to return;
        only the subclass's request timeout stops it being parked indefinitely.
        """
        for ctx in self._ctx.values():
            self._cancel(ctx)
        for gen in self._graveyard:
            if gen.thread is not None:
                gen.thread.join(timeout=timeout)
        self._graveyard.clear()

    def _ctx_of(self, job: JobId) -> _Context:
        ctx = self._ctx.get(job)
        if ctx is None:
            raise KeyError(f"no context for job {job}; create_context first")
        return ctx

    def stats(self, job: JobId) -> ContextStats:
        n = len(self._ctx_of(job).tokens)
        return ContextStats(
            resident_tokens=n,
            blocks=self._block_count(n),
            open_segment_tokens=n % self._block_size,
        )

    def _block_count(self, n: int) -> int:
        return (n + self._block_size - 1) // self._block_size

    @property
    def stranded(self) -> int:
        """Cancelled producers still running, and still billed; live, not cumulative."""
        return sum(
            1 for g in self._graveyard if g.thread is not None and g.thread.is_alive()
        )

    # -- generations ---------------------------------------------------------

    def invalidate(self, job: JobId) -> None:
        """Drop whatever is being generated for ``job``; a no-op if nothing is.

        The machine cannot see a preemption, and a resume whose diff is empty
        injects nothing, so the driver -- which reads the journal -- has to say so.
        """
        ctx = self._ctx.get(job)
        if ctx is not None:
            self._cancel(ctx)

    def _cancel(self, ctx: _Context) -> None:
        """Ask the producer to stop, and walk away from it.

        Walking away is the load-bearing half: the producer may not see the flag
        for seconds, and that wait must not sit in front of the next question.
        """
        gen, ctx.gen = ctx.gen, None
        # A half-built clause belongs to the completion being dropped; kept, it is
        # spliced onto the next completion's first words.
        ctx.clause = ""
        if gen is None or gen.done:
            return
        self._void_tail(ctx, gen)
        gen.stop.set()
        self._abort(gen)
        self.cancellations += 1
        self._graveyard.append(gen)
        self._graveyard[:] = [
            g for g in self._graveyard if g.thread is not None and g.thread.is_alive()
        ]

    def _void_tail(self, ctx: _Context, gen: _Generation) -> None:
        """Mark the cancelled generation's uncommitted output non-transmittable.

        Marked, never deleted: the kernel has already charged these tokens and
        extended its segment table over them, so shortening ``T`` here would leave
        offsets the kernel holds pointing past the end -- ``trunc`` is the kernel's
        to call. Only the projection to the wire changes.
        """
        if self.partial == "keep":
            return
        cut = gen.first_offset if self.partial == "drop" else gen.committed
        for index in range(max(cut, 0), len(ctx.origin)):
            if ctx.origin[index] == OUT:
                ctx.origin[index] = VOID
                self.voided += 1

    def _abort(self, gen: _Generation) -> None:
        """Shut a cancelled generation's connection, from whichever thread cancelled.

        The flag alone is cooperative, and on a server that serves one request at
        a time the next generation would wait behind a reply nobody will read.
        Closing under the producer makes its read raise, so ``_produce`` drops the
        exception for a generation that was asked to stop.
        """
        wire = gen.wire
        gen.wire = None
        if wire is None:
            return
        with contextlib.suppress(Exception):
            wire.close()  # type: ignore[attr-defined]

    def _start(self, ctx: _Context) -> _Generation:
        """Open a generation for the context as it now stands."""
        runs = self._runs(ctx)
        continuing = bool(runs) and runs[-1][0] == OUT
        gen = _Generation(
            mask=ctx.mask,
            first_offset=len(ctx.tokens),
            committed=len(ctx.tokens),
            opened=time.monotonic(),
        )
        gen.thread = threading.Thread(
            target=self._produce,
            args=(gen, self._request(ctx, continuing=continuing), ctx.descriptor),
            name=f"api-machine-{ctx.descriptor or 'job'}",
            daemon=True,
        )
        self.generations += 1
        gen.thread.start()
        return gen

    def _produce(self, gen: _Generation, request: object, descriptor: str) -> None:
        """The whole of the other thread; the sentinel tells the consumer it died."""
        stream = self._stream(request, descriptor, gen)
        try:
            for piece in stream:
                if not gen.first_at:
                    gen.first_at = time.monotonic()
                if gen.stop.is_set():
                    break
                gen.q.put(piece)
        except BaseException as exc:  # re-raised on the kernel's thread by decode
            # After a stop, a read that raises is `_abort` closing the socket.
            if not gen.stop.is_set():
                gen.error = exc
        finally:
            stream.close()
            gen.q.put(None)

    # -- turning the stream into elements ------------------------------------

    def _split(
        self, text: str, *, flush: bool, prose: bool = False
    ) -> tuple[list[str], str]:
        """Whole elements out of a run of text, and the remainder to keep.

        ``flush`` releases a trailing fragment, correct only once the stream ended.
        ``prose`` (the reasoning channel) splits on whitespace whatever the format,
        because a paragraph of thinking must not cost one token of budget. In
        ``json`` whitespace is content, so splitting on it would lose the space
        when the extractor rebuilds the document.
        """
        terminators = "" if prose else _TERMINATORS[self.syscall_format]
        out: list[str] = []
        rest = text
        if terminators and self.syscall_format != "text":
            while True:
                cut = min((rest.find(t) for t in terminators if t in rest), default=-1)
                if cut < 0:
                    break
                out.append(rest[: cut + 1])
                rest = rest[cut + 1 :]
            if flush and rest:
                out.append(rest)
                rest = ""
            return out, rest
        while True:
            head = rest.lstrip()
            parts = head.split(None, 1)
            if len(parts) == 2:
                word, rest = parts[0], parts[1]
            elif len(parts) == 1 and flush:
                word, rest = parts[0], ""
            else:
                rest = head
                break
            cut = min((word.find(t) for t in terminators if t in word), default=-1)
            if 0 <= cut < len(word) - 1:
                rest = f"{word[cut + 1 :]} {rest}"
                word = word[: cut + 1]
            out.append(word)
        return out, rest

    def _absorb(self, gen: _Generation, piece: Piece) -> None:
        """One piece off the wire becomes zero or more elements of ``T``.

        A channel switch flushes the previous channel's fragment so ``T`` keeps
        arrival order.
        """
        if gen.channel is not None and gen.channel != piece.kind:
            self._flush_channel(gen, gen.channel)
        gen.channel = piece.kind
        origin = THINK if piece.kind == "thinking" else OUT
        words, gen.carry[piece.kind] = self._split(
            gen.carry.get(piece.kind, "") + piece.text,
            flush=False,
            prose=origin is THINK,
        )
        for word in words:
            gen.ready.append((origin, word, piece.kind))

    def _flush_channel(self, gen: _Generation, kind: str) -> None:
        """Release one channel's held-back fragment as a whole element."""
        carried = gen.carry.get(kind, "")
        if not carried:
            return
        origin = THINK if kind == "thinking" else OUT
        words, gen.carry[kind] = self._split(carried, flush=True, prose=origin is THINK)
        for word in words:
            gen.ready.append((origin, word, kind))

    def _drain_carry(self, gen: _Generation) -> None:
        """Release every held-back fragment; only correct once the stream ended."""
        for kind in list(gen.carry):
            self._flush_channel(gen, kind)

    def _next_element(self, gen: _Generation) -> tuple[str, str, str] | None:
        """The next ``(origin, text, channel)``, or ``None`` after one ``stall_s``."""
        first = True
        while True:
            if gen.ready:
                return gen.ready.popleft()
            if gen.done:
                return None
            try:
                item = gen.q.get(timeout=self._stall_s) if first else gen.q.get_nowait()
            except queue.Empty:
                return None
            first = False
            if item is None:
                gen.done = True
                self._drain_carry(gen)
            else:
                self._absorb(gen, item)

    # -- extracting the request ----------------------------------------------

    def _request_from(
        self, ctx: _Context, gen: _Generation, word: str
    ) -> MachineRequest:
        """The request this element completed, if it completed one.

        Never called for the reasoning channel: the extractor is not fed, rather
        than fed and filtered.
        """
        if self.syscall_format == "text":
            return self._request_from_text(ctx, word)
        return self._request_from_json(gen, word)

    def _request_from_text(self, ctx: _Context, word: str) -> MachineRequest:
        """A clause begins at a verb and ends at the next semicolon; see ``VERBS``."""
        if word.rstrip(";") in VERBS:
            ctx.clause = word.rstrip(";")
        elif ctx.clause:
            ctx.clause = f"{ctx.clause} {word}"
        if word.endswith(";") and ctx.clause:
            clause, ctx.clause = ctx.clause, ""
            return parse_syscall(clause)
        return MachineRequest()

    def _request_from_json(self, gen: _Generation, word: str) -> MachineRequest:
        """Scan the half-arrived content for the completed move object."""
        gen.buf += word
        while gen.scan < len(gen.buf):
            char = gen.buf[gen.scan]
            gen.scan += 1
            if gen.in_string:
                if gen.escaped:
                    gen.escaped = False
                elif char == "\\":
                    gen.escaped = True
                elif char == '"':
                    gen.in_string = False
                continue
            if char == '"':
                gen.in_string = True
            elif char == "{":
                gen.depth += 1
                if gen.depth == 1:
                    gen.obj_start = gen.scan - 1
            elif char == "}":
                closing = gen.depth == 1 and gen.obj_start is not None
                gen.depth -= 1
                if closing:
                    raw = gen.buf[gen.obj_start : gen.scan]
                    gen.obj_start = None
                    return self._step_request(raw)
        return MachineRequest()

    def _step_request(self, raw: str) -> MachineRequest:
        """The move object as a request; a malformed one means an unenforced schema."""
        try:
            step = json.loads(raw)
        except ValueError:
            return MachineRequest()
        if not isinstance(step, dict):
            return MachineRequest()
        move = str(step.get("move", ""))
        if not move:
            return MachineRequest()
        return build_request("write", STDOUT, move)

    # -- the request ---------------------------------------------------------

    def _runs(self, ctx: _Context) -> list[tuple[str, str]]:
        """``T`` projected onto the wire, as ``(origin, text)`` runs.

        Roles come from provenance, not a kept history. Masked blocks are not
        transmitted, which is the mask enforced rather than requested; padding,
        reasoning and voided output are dropped by the same rule, which makes this
        a second page table the kernel cannot see -- a declared deviation from
        ZEOS-AM §8 that breaks nothing in AM-I3.
        """
        visible = self._visible(ctx)
        runs: list[tuple[str, list[str]]] = []
        # A skipped span can hide a change of speaker, so it breaks a run.
        broken = False
        pairs = zip(ctx.tokens, ctx.origin, strict=True)
        for index, (token, origin) in enumerate(pairs):
            if origin in (PAD, VOID) or index // self._block_size not in visible:
                broken = True
                continue
            if origin == THINK:
                continue
            if runs and runs[-1][0] == origin and not broken:
                runs[-1][1].append(token.text)
            else:
                runs.append((origin, [token.text]))
            broken = False
        return [(origin, " ".join(words)) for origin, words in runs]

    def _messages(self, ctx: _Context) -> list[dict[str, str]]:
        """The runs as an OpenAI-shaped message list; a vendor overrides the mapping."""
        return [
            {
                "role": (
                    "assistant" if origin == OUT else "system" if n == 0 else "user"
                ),
                "content": text,
            }
            for n, (origin, text) in enumerate(self._runs(ctx))
        ]

    # -- the five operations -------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        """Run to the next token boundary: four outcomes, two of them empty steps.

        The empty steps are the reason a job waiting on a served model is still a
        job the kernel can take the machine away from.
        """
        if allow_control:
            # ZEOS-AM §9 wants reserved-token control at the sampler; filtering
            # decoded text afterwards is what that requirement exists to forbid.
            raise NotImplementedError(
                "decode(allow_control=True) needs sampler-side reserved token ids "
                "(ZEOS-AM §9); an HTTP chat API exposes no sampler, and "
                "post-filtering decoded text is not an implementation of it"
            )
        ctx = self._ctx_of(job)

        if ctx.descriptor in self._behaviours:
            return self._decode_native(job, ctx)

        if ctx.gen is None:
            ctx.gen = self._start(ctx)
            return self._stall()

        if ctx.gen.error is not None:
            error, ctx.gen = ctx.gen.error, None
            raise error

        element = self._next_element(ctx.gen)

        if element is None:
            if not ctx.gen.done:
                return self._stall()
            # One reply, one turn. The job is put to sleep on ``stdin`` here rather
            # than asked to say so itself: ``game.state`` is level-triggered, so a
            # read is only a yield, and waiting has to be a state the kernel puts
            # the job in rather than an instruction in its prose.
            ctx.gen = None
            ctx.clause = ""
            return DecodeResult(
                tokens=(),
                request=build_request("read", STDIN),
                attention=None,
                attention_hint=AttentionHint(tags=("self",)),
            )

        origin, word, _channel = element

        # Kind is assigned from origin and never read off text, so a model cannot
        # produce a CONTROL token however it spells itself.
        token = Token(word, TokenKind.NORMAL)
        before = len(ctx.tokens)
        ctx.tokens.append(token)
        ctx.origin.append(origin)
        after = len(ctx.tokens)

        if origin is THINK:
            self.thinking_words += 1
            request = MachineRequest()
        else:
            ctx.gen.produced += 1
            self.words += 1
            request = self._request_from(ctx, ctx.gen, word)
            if request.op is not OpKind.NONE:
                # What comes after this point, if cancelled, is a fragment of a
                # call that never happened.
                ctx.gen.committed = len(ctx.tokens)
                self.last_roundtrip = round(time.monotonic() - ctx.gen.opened, 4)
                self.roundtrips.append(self.last_roundtrip)
                if ctx.gen.first_at:
                    self.ttfts.append(round(ctx.gen.first_at - ctx.gen.opened, 4))

        return DecodeResult(
            tokens=(token,),
            # Not measurable through a chat API; the hint says so (ZEOS-AM §7.2).
            attention=None,
            attention_hint=AttentionHint(tags=("self",)),
            request=request,
            at_block_boundary=after % self._block_size == 0 and after != before,
        )

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        """Append foreign tokens, and invalidate whatever is being generated.

        A request in flight cannot be amended, so the arrival reaches the model in
        the next request, built from a ``T`` that holds both the half-finished
        reply and the thing that interrupted it. Only the socket is thrown away.
        """
        ctx = self._ctx_of(job)
        start = len(ctx.tokens)
        ctx.tokens.extend(tokens)
        ctx.origin.extend([IN] * len(tokens))
        if ctx.descriptor in self._behaviours:
            ctx.arrived.extend(tokens)
        if tokens:
            # An append moves no existing offset, so unlike ``splice`` the order
            # of cancel and mutate does not matter here.
            self._cancel(ctx)
        return start, len(ctx.tokens)

    def trunc(self, job: JobId, at: int) -> int:
        ctx = self._ctx_of(job)
        if at < 0 or at > len(ctx.tokens):
            raise IndexError(f"trunc at {at} outside [0, {len(ctx.tokens)}]")
        dropped = len(ctx.tokens) - at
        if dropped:
            # Cancelled *before* the sequence moves. See ``splice``.
            self._cancel(ctx)
            del ctx.tokens[at:]
            del ctx.origin[at:]
            ctx.clause = ""
        return dropped

    def fork(self, parent: JobId, child: JobId) -> int:
        """Copy the parent's context into the child, which keeps its own descriptor."""
        src = self._ctx_of(parent)
        dst = self._ctx.get(child)
        if dst is None:
            dst = _Context(descriptor=src.descriptor)
            self._ctx[child] = dst
        # Cancelled *before* the child's sequence is replaced. See ``splice``.
        self._cancel(dst)
        dst.tokens = list(src.tokens)
        dst.origin = list(src.origin)
        dst.mask = src.mask
        dst.clause = ""
        return len(src.tokens)

    def splice(
        self, job: JobId, start: int, end: int, tokens: Sequence[Token]
    ) -> SpliceResult:
        """Replace ``[start, end)``; this is how eviction lands.

        A generation in flight is renumbered, not cancelled, unless the splice
        reaches its own output: the machine's own offsets inherit the renumbering
        obligation of ZEOS-AM §6.5, and cancelling on every eviction would make
        the pager and the job mutually exclusive.
        """
        ctx = self._ctx_of(job)
        if not 0 <= start <= end <= len(ctx.tokens):
            raise IndexError(f"splice [{start}, {end}) outside {len(ctx.tokens)}")
        downstream = len(ctx.tokens) - end
        gen = ctx.gen
        if gen is not None and end <= gen.first_offset:
            # Wholly upstream: every token this generation produced keeps its
            # identity and changes its index by the same amount.
            shift = len(tokens) - (end - start)
            gen.first_offset += shift
            gen.committed += shift
        else:
            self._cancel(ctx)
        ctx.tokens[start:end] = list(tokens)
        ctx.origin[start:end] = [IN] * len(tokens)
        return SpliceResult(tokens_in=len(tokens), invalidated_downstream=downstream)

    # -- the serving-stack contract ------------------------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        """Install the allowed-block bitmap, the MMU.

        Only a narrowing invalidates the generation: a decoding job widens the
        mask at every block boundary as its open segment grows, and every other
        route to a wider mask has already cancelled the generation on its own path.
        """
        ctx = self._ctx_of(job)
        if ctx.mask == allowed_blocks:
            return
        # Compared through ``_visible`` so that ⊥ is measured against what is
        # resident and a first mask naming every live block is not a narrowing.
        before = self._visible(ctx)
        ctx.mask = allowed_blocks
        if not before <= self._visible(ctx):
            self._cancel(ctx)

    def _visible(self, ctx: _Context) -> frozenset[int]:
        every = frozenset(range(self._block_count(len(ctx.tokens))))
        # ⊥ and ∅ are different (ZEOS-AM §8.2). Out-of-range indices drop rather
        # than grant, which is the fail-closed direction the invariant requires.
        return every if ctx.mask is None else every & ctx.mask

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        return self._visible(self._ctx_of(job))

    def pad_to_block(self, job: JobId) -> int:
        """Close the current block with reserved no-op tokens.

        Two segments sharing a block could not be masked apart. Padding does not
        invalidate a generation: pads are never transmitted, so the request is
        byte-identical, and the kernel pads before almost every injection.
        """
        ctx = self._ctx_of(job)
        remainder = len(ctx.tokens) % self._block_size
        if remainder == 0:
            return 0
        padding = self._block_size - remainder
        ctx.tokens.extend([PAD_TOKEN] * padding)
        ctx.origin.extend([PAD] * padding)
        return padding

    def blocks_for_range(self, job: JobId, start: int, end: int) -> frozenset[int]:
        self._ctx_of(job)
        if end <= start:
            return frozenset()
        first = start // self._block_size
        last = (end - 1) // self._block_size
        return frozenset(range(first, last + 1))

    def transcript(self, job: JobId) -> tuple[Token, ...]:
        return tuple(self._ctx_of(job).tokens)

    # -- the empty step ------------------------------------------------------

    @staticmethod
    def _stall() -> DecodeResult:
        """A step that produced nothing (ZEOS-AM §6.1), leaving the job RUNNING."""
        return DecodeResult(
            tokens=(),
            attention=None,
            attention_hint=AttentionHint(),
        )
