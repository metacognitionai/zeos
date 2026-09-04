# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The M0 backend: jobs as scripted token streams.

No GPU, no model, no tokenizer. A job's behaviour is a list of steps, each of which
emits some tokens and may request one kernel service. This is what makes the whole
skeleton testable: a run is a pure function of (descriptor tree, scripts, event
schedule), so the determinism gate is meaningful and unit tests need no mocks.

Two things are simulated rather than faked, because getting them wrong would make
the MP/VM logic above unreliable when the real machine arrives:

**Block structure.** Segments are block-aligned by construction,
and almost all MP/VM work lands at block boundaries -- mask changes, watermark
demotion, eviction batching, status refresh. A backend without blocks would let us
write boundary-free logic that breaks at M1, so blocks, padding, and boundary
signalling are all real here.

**Control-token unforgeability.** A script cannot emit a CONTROL token unless the
kernel enabled control tokens for that step. This is structural, not a scan: it
mirrors tokenizer control-ID disabling and sampler-side reservation rather than
string matching, which would teach the wrong lesson.

One thing *is* fiction, and is labelled as such throughout: **attention mass**.
This backend cannot measure it, so it returns an ``AttentionHint`` for the kernel
to resolve. Mechanism conclusions (demotion fires, the clock evicts, faults route)
are valid; policy conclusions (eviction regret, θ sensitivity, taint-creep rates)
are not available from any run of this backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from zeos.core.ids import JobId, PipeName, SegmentId, TokenKind
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

__all__ = ["Step", "Script", "ScriptedMachine", "PAD_TOKEN", "ScriptExhausted"]

#: Reserved no-op token used for block padding.
PAD_TOKEN = Token("<pad>", TokenKind.CONTROL)

DEFAULT_BLOCK_SIZE = 16


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Normalise a YAML scalar-or-list into a tuple of strings.

    Case files are hand-written YAML, so ``attend: web.fetch`` and
    ``attend: [web.fetch]`` should both work -- accepting only the list form is the
    kind of papercut that makes a case directory tedious to author.
    """
    if isinstance(value, list | tuple):
        items: Sequence[Any] = cast("Sequence[Any]", value)
        return tuple(str(v) for v in items)
    return (str(value),)


class ScriptExhausted(RuntimeError):
    """A job was decoded past the end of its script without an explicit ``exit``.

    Treated as an error rather than an implicit exit: a script that runs off the end
    is almost always an authoring mistake, and silently completing the job would
    hide it behind a passing test.
    """


@dataclass(frozen=True, slots=True)
class Step:
    """One decode step: some emitted tokens, and at most one kernel request."""

    emit: str = ""
    request: MachineRequest = field(default_factory=MachineRequest)
    hint: AttentionHint | None = None
    #: Emit the tokens as CONTROL. Only legal when the kernel enabled control for
    #: this step; otherwise the machine raises rather than letting it through.
    control: bool = False

    def tokens(self) -> tuple[Token, ...]:
        if not self.emit:
            return ()
        kind = TokenKind.CONTROL if self.control else TokenKind.NORMAL
        return tokens_from_text(self.emit, kind)


@dataclass(frozen=True, slots=True)
class Script:
    """A job's scripted behaviour, keyed by descriptor name in the machine."""

    steps: tuple[Step, ...] = ()

    @staticmethod
    def from_spec(spec: Iterable[Mapping[str, Any]]) -> Script:
        """Build from the YAML form used in case directories and fixtures.

        Recognised keys, at most one request per step::

            - emit: "Checking fan 3."
            - read: plant.tags
            - write:  {pipe: ops.report, text: "shift nominal"}
            - write:  {pipe: peers.a2b, text: "go", then_read: peers.b2a}
            - select: [user.commands, sensors.alarms]
            - fault:  4                       # stub segment id
            - need:   "maintenance log for pump 4"
            - acquire: door.south
            - release: door.south
            - spawn:  clear-bench
            - exit:   true
            - attend: [web.fetch]             # attention hint for this step
        """
        steps: list[Step] = []
        for raw in spec:
            emit = str(raw.get("emit", "") or "")
            hint: AttentionHint | None = None
            if "attend" in raw:
                hint = AttentionHint(tags=_as_str_tuple(raw["attend"]))

            request = MachineRequest()
            if "read" in raw:
                request = MachineRequest(op=OpKind.READ, pipe=PipeName(str(raw["read"])))
            elif "write" in raw:
                if not isinstance(raw["write"], Mapping):
                    raise ValueError("'write' step must be a mapping with 'pipe'")
                w = cast("Mapping[str, Any]", raw["write"])
                after = w.get("then_read")
                request = MachineRequest(
                    op=OpKind.WRITE if after is None else OpKind.WRITE_READ,
                    pipe=PipeName(str(w["pipe"])),
                    payload=tokens_from_text(str(w.get("text", ""))),
                    read_pipe=None if after is None else PipeName(str(after)),
                )
            elif "select" in raw:
                request = MachineRequest(
                    op=OpKind.SELECT,
                    pipes=tuple(PipeName(p) for p in _as_str_tuple(raw["select"])),
                )
            elif "acquire" in raw:
                request = MachineRequest(op=OpKind.ACQUIRE, resource=str(raw["acquire"]))
            elif "release" in raw:
                request = MachineRequest(op=OpKind.RELEASE, resource=str(raw["release"]))
            elif "fault" in raw:
                request = MachineRequest(op=OpKind.FAULT, segment=SegmentId(int(str(raw["fault"]))))
            elif "need" in raw:
                request = MachineRequest(op=OpKind.NEED, text=str(raw["need"]))
            elif "spawn" in raw:
                request = MachineRequest(op=OpKind.SPAWN, text=str(raw["spawn"]))
            elif raw.get("exit"):
                request = MachineRequest(op=OpKind.EXIT)

            steps.append(
                Step(
                    emit=emit,
                    request=request,
                    hint=hint,
                    control=bool(raw.get("control", False)),
                )
            )
        return Script(steps=tuple(steps))


@dataclass
class _Context:
    """Per-job materialised state. The machine owns tokens and blocks; the kernel
    owns segments, rings, integrity and residency."""

    tokens: list[Token] = field(default_factory=list["Token"])
    pc: int = 0  # program counter into the script
    mask: frozenset[int] | None = None  # None = unrestricted
    script: Script = field(default_factory=Script)


class ScriptedMachine:
    """A ``MachineBackend`` driven by scripts instead of weights."""

    def __init__(
        self,
        scripts: Mapping[str, Script] | None = None,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self._block_size = block_size
        self._scripts: dict[str, Script] = dict(scripts or {})
        self._ctx: dict[JobId, _Context] = {}

    # -- lifecycle ----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    def register_script(self, descriptor: str, script: Script) -> None:
        self._scripts[descriptor] = script

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        self._ctx[job] = _Context(script=self._scripts.get(descriptor, Script()))

    def destroy_context(self, job: JobId) -> None:
        self._ctx.pop(job, None)

    def _ctx_of(self, job: JobId) -> _Context:
        ctx = self._ctx.get(job)
        if ctx is None:
            raise KeyError(f"no context for job {job}; create_context first")
        return ctx

    def stats(self, job: JobId) -> ContextStats:
        ctx = self._ctx_of(job)
        n = len(ctx.tokens)
        return ContextStats(
            resident_tokens=n,
            blocks=self._block_count(n),
            open_segment_tokens=n % self._block_size,
        )

    def _block_count(self, n_tokens: int) -> int:
        return (n_tokens + self._block_size - 1) // self._block_size

    # -- the five ops -------------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        if ctx.pc >= len(ctx.script.steps):
            raise ScriptExhausted(
                f"job {job} decoded past the end of its script "
                f"({len(ctx.script.steps)} steps); scripts must end with 'exit'"
            )
        step = ctx.script.steps[ctx.pc]
        ctx.pc += 1

        tokens = step.tokens()
        if not allow_control and any(t.kind is TokenKind.CONTROL for t in tokens):
            # Structural, not punitive: at MP1 the sampler simply cannot select
            # these ids, so reaching this branch means the backend has a bug.
            raise ControlTokenViolation(
                f"job {job} emitted a CONTROL token while control tokens were "
                "disabled for this step"
            )

        before = len(ctx.tokens)
        ctx.tokens.extend(tokens)
        after = len(ctx.tokens)

        return DecodeResult(
            tokens=tokens,
            request=step.request,
            attention=None,  # this backend cannot measure; see attention_hint
            attention_hint=step.hint or AttentionHint(),
            at_block_boundary=after % self._block_size == 0 and after != before,
        )

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        ctx = self._ctx_of(job)
        start = len(ctx.tokens)
        ctx.tokens.extend(tokens)
        return start, len(ctx.tokens)

    def trunc(self, job: JobId, at: int) -> int:
        ctx = self._ctx_of(job)
        if at < 0 or at > len(ctx.tokens):
            raise IndexError(f"trunc at {at} outside [0, {len(ctx.tokens)}]")
        dropped = len(ctx.tokens) - at
        del ctx.tokens[at:]
        return dropped

    def fork(self, parent: JobId, child: JobId) -> int:
        """Copy the parent's materialised context into the child.

        Copy-on-write is an accounting story rather than a memory one at this scale:
        the token list is copied and the *cost model* is what the kernel reasons
        about. A real backend shares KV blocks instead.

        If the child already has a context, its own script and program counter are
        preserved and only the tokens are copied in. That matters for compartments:
        a compartment child runs *its own* behaviour over a slice of the parent's
        context, so inheriting the parent's script would be exactly wrong.
        """
        src = self._ctx_of(parent)
        existing = self._ctx.get(child)
        if existing is not None:
            existing.tokens = list(src.tokens)
            existing.mask = src.mask
        else:
            self._ctx[child] = _Context(
                tokens=list(src.tokens), pc=src.pc, mask=src.mask, script=src.script
            )
        return len(src.tokens)

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        ctx = self._ctx_of(job)
        if not 0 <= start <= end <= len(ctx.tokens):
            raise IndexError(f"splice [{start}, {end}) outside context of {len(ctx.tokens)}")
        downstream = len(ctx.tokens) - end
        ctx.tokens[start:end] = list(tokens)
        return SpliceResult(tokens_in=len(tokens), invalidated_downstream=downstream)

    # -- serving-stack contract ---------------------------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        self._ctx_of(job).mask = allowed_blocks

    def clear_mask(self, job: JobId) -> None:
        self._ctx_of(job).mask = None

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        ctx = self._ctx_of(job)
        every = frozenset(range(self._block_count(len(ctx.tokens))))
        return every if ctx.mask is None else every & ctx.mask

    def pad_to_block(self, job: JobId) -> int:
        ctx = self._ctx_of(job)
        remainder = len(ctx.tokens) % self._block_size
        if remainder == 0:
            return 0
        padding = self._block_size - remainder
        ctx.tokens.extend([PAD_TOKEN] * padding)
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

    # -- synthetic attention -------------------------------------------------

    def recency_profile(self, job: JobId, scale: float) -> dict[int, float]:
        """Recency weighting over currently visible blocks, normalised to 1.0.

        SYNTHETIC. Present so the kernel has a deterministic default when a script
        gives no explicit hint. The shape (exponential decay toward the tail) is a
        plausible guess and nothing more -- which is exactly why no policy claim can
        rest on it.
        """
        visible = sorted(self.visible_blocks(job))
        if not visible:
            return {}
        newest = visible[-1]
        weights = {b: pow(2.718281828, -(newest - b) / max(scale, 1e-6)) for b in visible}
        total = sum(weights.values())
        return {b: w / total for b, w in weights.items()} if total > 0 else {}
