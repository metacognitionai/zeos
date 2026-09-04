# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The machine backend: the five ops, block boundaries, masking, attention.

This is the most important interface in the system, because it is the one M1 swaps
for a real paged-KV serving stack. Everything above it -- scheduler, segments,
integrity, paging -- is written against this and nothing else.

The contract mirrors the four asks ZEOS makes of a serving stack, which is the whole
point: the four things ZEOS needs from a serving stack are (a) an allowed-block
bitmap fed to attention, (b) per-segment attention mass aggregated per block,
(c) reserved-token control in tokenizer and sampler, and (d) grammar-constrained
decoding for endorser schemas. All four appear here, so a backend that satisfies
this interface satisfies the spec.

Division of responsibility, which is easy to get wrong:

* The **machine** owns the materialised token sequence and its blocks. It knows
  offsets and lengths, and it enforces the mask.
* The **kernel** owns the segment table, rings, integrity, and residency. Segment
  metadata is PCB state, not context.

They meet at token offsets. The machine reports what it did; the kernel decides
what it means.

Attention mass returned by a backend that cannot measure it (i.e. M0) is
**synthetic**. See ``ScriptedMachine`` -- it is enough to validate mechanism and is
not evidence about policy.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from zeos.core.ids import JobId, PipeName, SegmentId, TokenKind

__all__ = [
    "Token",
    "OpKind",
    "MachineRequest",
    "AttentionHint",
    "DecodeResult",
    "SpliceResult",
    "ContextStats",
    "MachineBackend",
    "ControlTokenViolation",
    "MaskViolation",
    "tokens_from_text",
    "render",
]


class ControlTokenViolation(RuntimeError):
    """A stream tried to emit a CONTROL token the kernel had not enabled.

    In M0 this is impossible-by-construction rather than detected-and-rejected: a
    backend raises this only if it has a bug. At MP1 the equivalent is the sampler
    mask, which makes fabrication of kernel framing structurally unavailable rather
    than merely punished.
    """


class MaskViolation(RuntimeError):
    """A backend attempted to attend a segment excluded by the allowed-block bitmap.

    Like the above, this signals a backend bug. Hard enforcement means the attention
    never happens; the kernel journals an ``AttentionDenied`` and continues.
    """


@dataclass(frozen=True, slots=True)
class Token:
    """One token. In M0 a token is a short string, which keeps transcripts readable
    without changing any of the accounting -- every ``Token`` counts as one token."""

    text: str
    kind: TokenKind = TokenKind.NORMAL

    def __str__(self) -> str:
        return self.text


def tokens_from_text(text: str, kind: TokenKind = TokenKind.NORMAL) -> tuple[Token, ...]:
    """Whitespace tokenisation. Crude on purpose: M0 measures *counts and
    boundaries*, and a real tokenizer arrives with the real machine."""
    return tuple(Token(word, kind) for word in text.split())


def render(tokens: Sequence[Token]) -> str:
    return " ".join(t.text for t in tokens)


class OpKind(enum.StrEnum):
    """What a decoding job is asking the kernel to do.

     In ZEOS a job's only effects are pipe writes, and its only inputs are pipe reads
     (core §4.3: tool calls are pipe I/O). FAULT and NEED are the paging grammars
    . There is deliberately no YIELD: jobs cannot volunteer
     scheduling decisions (core Appendix A, rule 2).
    """

    NONE = "none"
    READ = "read"  # blocking read; deschedules
    WRITE = "write"  # checked against the capability record
    SELECT = "select"  # block until any of several pipes is readable
    WRITE_READ = "write_read"  # hand over and sleep, with no boundary between
    FAULT = "fault"  # reference to a stub handle
    NEED = "need"  # structured request for content believed to exist
    ACQUIRE = "acquire"  # take a resource; blocks if full
    RELEASE = "release"  # give it back
    SPAWN = "spawn"  # start a child job
    EXIT = "exit"  # job is done


@dataclass(frozen=True, slots=True)
class MachineRequest:
    """A kernel service requested by the job at a token boundary."""

    op: OpKind = OpKind.NONE
    pipe: PipeName | None = None
    pipes: tuple[PipeName, ...] = ()  # for SELECT
    payload: tuple[Token, ...] = ()  # for WRITE
    segment: SegmentId | None = None  # for FAULT
    resource: str | None = None  # for ACQUIRE / RELEASE
    text: str | None = None  # for NEED / SPAWN (descriptor name)
    read_pipe: PipeName | None = None  # the read half of WRITE_READ


@dataclass(frozen=True, slots=True)
class AttentionHint:
    """A *synthetic* attention declaration, used only by backends that cannot
    measure the real thing.

    The kernel resolves ``tags`` -- source labels like a pipe name, ``"descriptor"``,
    or ``"self"`` -- to segments, because the kernel owns the segment table and the
    machine does not. Untagged mass falls back to recency weighting over resident
    blocks.

    This type exists to make the fiction visible in the type system. A real backend
    returns ``DecodeResult.attention`` and leaves this ``None``; only M0 populates
    it. Anywhere this appears, policy conclusions are unavailable.
    """

    tags: tuple[str, ...] = ()
    #: Fraction of total mass given to the tagged segments; the remainder is spread
    #: by recency. 1.0 means "this step attended the tagged content and nothing else".
    tag_weight: float = 0.8
    #: Larger = flatter recency profile over untagged resident blocks.
    recency_scale: float = 8.0


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """The outcome of running one job up to the next token boundary."""

    tokens: tuple[Token, ...]
    request: MachineRequest = field(default_factory=MachineRequest)
    #: Measured attention mass per **KV block** for this step. Block-granular
    #: because that is what a paged serving stack actually aggregates, and because
    #: segments are block-aligned by construction the kernel can sum it per segment
    #: exactly. ``None`` when the backend cannot measure.
    #:
    #: **Units are normative: summed over layers and heads, normalised per block**,
    #: so the values for one step sum to 1.0 over the blocks that received attention
    #: (ZEOS-AM §7.1). This is not a formatting preference. The
    #: kernel sums these per segment and compares against ``theta_read``, whose
    #: default means "a fifth of a block's attention", so a backend reporting raw
    #: weights that sum to the head count would satisfy this type, pass every test,
    #: and get every integrity-demotion and eviction decision wrong by orders of
    #: magnitude without raising anything.
    attention: Mapping[int, float] | None = None
    #: Synthetic stand-in, used when ``attention`` is None. See ``AttentionHint``.
    attention_hint: AttentionHint | None = None
    #: True when this step landed exactly on a block boundary. **Advisory, and no
    #: kernel path consumes it.** Note that "landed on" is not "crossed into": a step
    #: taking a context from 3 to 6 tokens with a block size of 4 crosses a boundary
    #: and reports False. The kernel derives boundaries itself from
    #: ``stats().resident_tokens // block_size``, which is correct under multi-token
    #: steps. Do not start consuming this without redefining it (ZEOS-AM §6.1.1).
    at_block_boundary: bool = False


@dataclass(frozen=True, slots=True)
class SpliceResult:
    tokens_in: int
    #: Tokens downstream of the splice whose KV is invalidated -- the ``d`` term in
    #: the cost model. Accounted rather than measured in M0.
    invalidated_downstream: int


@dataclass(frozen=True, slots=True)
class ContextStats:
    resident_tokens: int
    blocks: int
    open_segment_tokens: int


@runtime_checkable
class MachineBackend(Protocol):
    """What the kernel requires of any execution substrate."""

    @property
    def block_size(self) -> int:
        """KV block size. Segments are block-aligned to this."""
        ...

    def create_context(self, job: JobId, descriptor: str) -> None:
        """Open a context for a job. ``descriptor`` identifies which behaviour is
        running, which a real backend may use to select a model or adapter and
        which the scripted backend uses to find the script."""
        ...

    def destroy_context(self, job: JobId) -> None: ...

    def stats(self, job: JobId) -> ContextStats: ...

    # -- the five ops -------------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        """Run to the next token boundary.

        ``allow_control`` is sampler-side reservation: the model
        can never fabricate kernel framing unless the kernel explicitly enables it.
        """
        ...

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        """Append foreign tokens. Returns ``(start_offset, end_offset)``.

        The only entry path for foreign tokens, which is what makes provenance
        total.
        """
        ...

    def trunc(self, job: JobId, at: int) -> int:
        """Drop everything from offset ``at``. Returns tokens dropped."""
        ...

    def fork(self, parent: JobId, child: JobId) -> int:
        """Copy-on-write context copy. Returns shared token count."""
        ...

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        """Replace ``[start, end)`` with ``tokens``. Only at segment boundaries."""
        ...

    # -- the serving-stack contract --------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        """Install the allowed-block bitmap -- the MMU. Enforcement is per forward
        pass and involves no model cooperation."""
        ...

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        """Blocks the job may currently attend.

        The kernel filters resolved attention through this, so that a masked
        segment contributes no mass no matter what the job attempted. That is the
        difference between hard enforcement and a request the job could decline.
        """
        ...

    def pad_to_block(self, job: JobId) -> int:
        """Close the current block with reserved no-op tokens. Returns padding
        added, bounded by ``block_size - 1``."""
        ...

    def blocks_for_range(self, job: JobId, start: int, end: int) -> frozenset[int]:
        """Which KV blocks back a token range -- the segment→bitmap mapping."""
        ...

    def transcript(self, job: JobId) -> tuple[Token, ...]:
        """The full resident token sequence. The transcript is the source of truth;
        KV is a discardable materialisation of it (core §2.1)."""
        ...
