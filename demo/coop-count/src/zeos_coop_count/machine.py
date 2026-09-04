# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A machine backend that runs a real llama.cpp forward pass for each decode step, owning
the token sequence and its blocks while the kernel owns segments, rings and integrity."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import llama_cpp

from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import (
    AttentionHint,
    ContextStats,
    ControlTokenViolation,
    DecodeResult,
    MachineRequest,
    MaskViolation,
    OpKind,
    SpliceResult,
    Token,
)
from zeos_coop_count.seat import SyscallSeat
from zeos_coop_count.syscall import SyscallParser, build_grammar

__all__ = ["LlamaModel", "LlamaMachine", "PAD_TOKEN", "DEFAULT_BLOCK_SIZE"]

#: Reserved no-op used for block padding, marked CONTROL so a job cannot emit it.
PAD_TOKEN = Token("<pad>", TokenKind.CONTROL)

DEFAULT_BLOCK_SIZE = 16

_BACKEND_READY = False


def _init_backend() -> None:
    global _BACKEND_READY
    if not _BACKEND_READY:
        llama_cpp.llama_backend_init()
        _BACKEND_READY = True


class LlamaModel:
    """Loaded weights, kept separate from any context over them so they load once."""

    def __init__(self, path: str | Path, *, n_gpu_layers: int = 0) -> None:
        _init_backend()
        self.path = str(path)
        params = llama_cpp.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        model = llama_cpp.llama_model_load_from_file(self.path.encode(), params)
        if not model:
            raise RuntimeError(f"could not load model {self.path}")
        self.model = model
        self.vocab = llama_cpp.llama_model_get_vocab(model)
        self.n_vocab = llama_cpp.llama_vocab_n_tokens(self.vocab)

    def tokenize(self, text: str, *, add_bos: bool, parse_special: bool = False) -> list[int]:
        """Turn text into token ids, treating text that spells a special token as plain text."""
        raw = text.encode("utf-8")
        cap = len(raw) + 8
        out = (llama_cpp.llama_token * cap)()
        n = llama_cpp.llama_tokenize(self.vocab, raw, len(raw), out, cap, add_bos, parse_special)
        if n < 0:
            raise RuntimeError(f"tokenization overflow for {text!r}")
        return list(out[:n])

    def piece(self, token: int) -> str:
        buf = ctypes.create_string_buffer(64)
        n = llama_cpp.llama_token_to_piece(self.vocab, token, buf, 64, 0, True)
        if n < 0:
            raise RuntimeError(f"detokenization failed for id {token}")
        return bytes(buf[:n]).decode("utf-8", errors="replace")

    def free(self) -> None:
        if getattr(self, "model", None):
            llama_cpp.llama_model_free(self.model)
            self.model = None


@dataclass
class _Context:
    """Per-job state: one llama sequence, one grammar, one parser."""

    seq: int
    descriptor: str
    #: The kernel-visible token sequence T.
    tokens: list[Token] = field(default_factory=list[Token])
    #: The llama token ids backing T, flat.
    ids: list[int] = field(default_factory=list[int])
    #: ``spans[i]`` is how many llama ids element ``i`` of T occupies.
    spans: list[int] = field(default_factory=list[int])
    #: How many llama ids have had a forward pass; the rest are flushed on the next decode.
    n_in_kv: int = 0
    mask: frozenset[int] | None = None
    sampler: object | None = None
    parser: SyscallParser = field(default_factory=SyscallParser)
    grammar: str = ""
    #: Tags for the attention hint, naming what this step most likely attends to.
    tags: tuple[str, ...] = ()
    #: Whether the model's own turn has been opened.
    turn_open: bool = False
    #: Which element of T carries the turn marker in its span, so truncating past it
    #: reopens the prompt turn.
    turn_index: int | None = None
    #: Whether the job has decoded anything yet.
    spoken: bool = False

    def kv_offset(self, at: int) -> int:
        """Turn a kernel offset into a llama position, the one place the two units meet."""
        return sum(self.spans[:at])


class LlamaMachine(SyscallSeat):
    """A machine backend with one llama sequence per job, whose decode samples one token."""

    def __init__(
        self,
        model: LlamaModel,
        *,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        valued: Mapping[str, Sequence[str]] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        n_ctx: int = 8192,
        n_batch: int = 512,
        n_seq_max: int = 8,
        n_threads: int = 8,
        enforce_mask: bool = True,
        chat_template: str | None = "chatml",
        on_command: Callable[[JobId, str, MachineRequest], None] | None = None,
        on_arrival: Callable[[JobId, str], None] | None = None,
    ) -> None:
        super().__init__(
            descriptors=descriptors,
            valued=valued,
            on_command=on_command,
            on_arrival=on_arrival,
        )
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self._model = model
        self._block_size = block_size
        self._n_batch = n_batch
        self._enforce_mask = enforce_mask
        if chat_template not in (None, "chatml"):
            raise ValueError(f"unknown chat_template {chat_template!r}")
        self._chat_template = chat_template

        params = llama_cpp.llama_context_default_params()
        # llama shares its ``n_ctx`` across all sequences, so scale it up to give each job
        # the full ``n_ctx`` asked for here.
        params.n_ctx = n_ctx * n_seq_max
        params.n_batch = n_batch
        params.n_ubatch = n_batch
        params.n_seq_max = n_seq_max
        params.n_threads = n_threads
        params.n_threads_batch = n_threads
        params.no_perf = True
        ctx = llama_cpp.llama_init_from_model(model.model, params)
        if not ctx:
            raise RuntimeError("could not create llama context")
        self._ctx = ctx
        self._mem = llama_cpp.llama_get_memory(ctx)
        self._n_ctx_seq = int(llama_cpp.llama_n_ctx_seq(ctx))
        if self._n_ctx_seq < n_ctx:
            raise RuntimeError(
                f"asked for {n_ctx} tokens per job but llama allotted {self._n_ctx_seq}"
            )
        self._batch = llama_cpp.llama_batch_init(n_batch, 0, n_seq_max)

        self._contexts: dict[JobId, _Context] = {}
        self._free_seqs = list(range(n_seq_max))
        #: Which sequence the one shared logits buffer currently describes.
        self._logits_owner: int | None = None

        self._pad_id = self._choose_pad(model)
        #: Ids the sampler may never select.
        self._reserved = frozenset({self._pad_id})

    @staticmethod
    def _choose_pad(model: LlamaModel) -> int:
        """Pick the id block padding is made of, preferring a newline because it reads as text."""
        newline = model.tokenize("\n", add_bos=False)
        if len(newline) == 1:
            return newline[0]
        pad = llama_cpp.llama_vocab_pad(model.vocab)
        if pad is None or pad < 0:
            pad = llama_cpp.llama_vocab_eos(model.vocab)
        return int(pad)

    # -- lifecycle ----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    #: The chat turn markers, the only strings tokenized with ``parse_special`` true.
    _CHATML_OPEN = "<|im_start|>user\n"
    _CHATML_TURN = "<|im_end|>\n<|im_start|>assistant\n"
    _CHATML_ARRIVE = "<|im_end|>\n<|im_start|>user\n"

    def _frame_into(self, ctx: _Context, text: str, index: int, *, before: bool) -> None:
        """Fold chat framing into the span of the token at ``index``, so no kernel offset moves."""
        ids = self._model.tokenize(text, add_bos=False, parse_special=True)
        if not ids:
            return
        at = ctx.kv_offset(index) if before else ctx.kv_offset(index + 1)
        if at < ctx.n_in_kv:
            self._rewind(ctx, at)
        ctx.ids[at:at] = ids
        ctx.spans[index] += len(ids)

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        if job in self._contexts:
            self.destroy_context(job)
        if not self._free_seqs:
            raise RuntimeError("out of llama sequences; raise n_seq_max")
        seq = self._free_seqs.pop(0)
        ctx = _Context(seq=seq, descriptor=descriptor, tags=("descriptor",))
        ctx.grammar = build_grammar(self.aliases(descriptor), valued=self.valued(descriptor))
        self._build_sampler(ctx)
        self._contexts[job] = ctx

    def destroy_context(self, job: JobId) -> None:
        ctx = self._contexts.pop(job, None)
        if ctx is None:
            return
        llama_cpp.llama_memory_seq_rm(self._mem, ctx.seq, -1, -1)
        self._free_sampler(ctx)
        if self._logits_owner == ctx.seq:
            self._logits_owner = None
        self._free_seqs.append(ctx.seq)

    def _ctx_of(self, job: JobId) -> _Context:
        ctx = self._contexts.get(job)
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

    def _block_count(self, n: int) -> int:
        return (n + self._block_size - 1) // self._block_size

    # -- sampler ------------------------------------------------------------

    def _build_sampler(self, ctx: _Context) -> None:
        """Build a fresh sampler chain: reserved-id bias, then grammar, then greedy choice.

        Greedy is not a quality choice: a temperature would stop a run being a function of
        its own history, and the journal has to reproduce byte for byte.
        """
        params = llama_cpp.llama_sampler_chain_default_params()
        params.no_perf = True
        chain = llama_cpp.llama_sampler_chain_init(params)

        bias = (llama_cpp.llama_logit_bias * len(self._reserved))()
        for i, tid in enumerate(sorted(self._reserved)):
            bias[i].token = tid
            bias[i].bias = float("-inf")
        llama_cpp.llama_sampler_chain_add(
            chain,
            llama_cpp.llama_sampler_init_logit_bias(self._model.n_vocab, len(bias), bias),
        )
        llama_cpp.llama_sampler_chain_add(
            chain,
            llama_cpp.llama_sampler_init_grammar(self._model.vocab, ctx.grammar.encode(), b"root"),
        )
        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_greedy())
        ctx.sampler = chain

    def _free_sampler(self, ctx: _Context) -> None:
        if ctx.sampler is not None:
            llama_cpp.llama_sampler_free(ctx.sampler)
            ctx.sampler = None

    def _reset_grammar(self, ctx: _Context) -> None:
        """Rebuild the sampler so the grammar accepts another request, as it cannot rewind."""
        self._free_sampler(ctx)
        self._build_sampler(ctx)
        ctx.parser.reset()

    # -- the KV --------------------------------------------------------------

    def _rewind(self, ctx: _Context, kv_at: int) -> None:
        """Drop this sequence's KV from ``kv_at`` on, clearing it all if the model cannot cut."""
        if kv_at > 0 and llama_cpp.llama_memory_seq_rm(self._mem, ctx.seq, kv_at, -1):
            ctx.n_in_kv = min(ctx.n_in_kv, kv_at)
        else:
            llama_cpp.llama_memory_seq_rm(self._mem, ctx.seq, -1, -1)
            ctx.n_in_kv = 0
        if self._logits_owner == ctx.seq:
            self._logits_owner = None

    def _flush(self, ctx: _Context, *, want_logits: bool) -> None:
        """Run forward passes for everything appended since the last one."""
        pending = ctx.ids[ctx.n_in_kv :]
        if not pending:
            return
        if len(ctx.ids) > self._n_ctx_seq:
            # llama would report this as a bare -1, so say what ran out and what to change.
            raise RuntimeError(
                f"job context is {len(ctx.ids)} llama tokens, over the {self._n_ctx_seq} "
                f"this machine allots each job. Raise --n-ctx, or lower the descriptor's "
                f"context.window so the kernel evicts before llama runs out."
            )
        batch = self._batch
        for offset in range(0, len(pending), self._n_batch):
            chunk = pending[offset : offset + self._n_batch]
            last_chunk = offset + len(chunk) >= len(pending)
            batch.n_tokens = len(chunk)
            for i, tid in enumerate(chunk):
                batch.token[i] = tid
                batch.pos[i] = ctx.n_in_kv + offset + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = ctx.seq
                batch.logits[i] = 1 if (want_logits and last_chunk and i == len(chunk) - 1) else 0
            rc = llama_cpp.llama_decode(self._ctx, batch)
            if rc != 0:
                raise RuntimeError(
                    f"llama_decode failed ({rc}) for seq {ctx.seq}: "
                    f"{len(chunk)} tokens at positions "
                    f"{ctx.n_in_kv + offset}..{ctx.n_in_kv + offset + len(chunk) - 1}, "
                    f"n_in_kv={ctx.n_in_kv}, |ids|={len(ctx.ids)}, "
                    f"|T|={len(ctx.tokens)}, n_ctx_seq={self._n_ctx_seq}"
                )
        ctx.n_in_kv = len(ctx.ids)
        self._logits_owner = ctx.seq if want_logits else None

    def _ensure_logits(self, ctx: _Context) -> None:
        """Make the shared logits buffer describe this sequence, redoing one token if need be."""
        if ctx.n_in_kv == len(ctx.ids) and self._logits_owner == ctx.seq:
            return
        if ctx.n_in_kv == len(ctx.ids):
            if not ctx.ids:
                raise RuntimeError(
                    "cannot decode an empty context; the kernel injects the descriptor "
                    "body before the first decode"
                )
            self._rewind(ctx, len(ctx.ids) - 1)
        self._flush(ctx, want_logits=True)

    def _encode(
        self, tokens: Sequence[Token], ctx: _Context
    ) -> tuple[list[Token], list[int], list[int]]:
        """Turn kernel tokens into ids and spans, splitting runs where the token kind changes."""
        out_tokens: list[Token] = []
        out_ids: list[int] = []
        out_spans: list[int] = []
        first = not ctx.ids
        i = 0
        while i < len(tokens):
            kind = tokens[i].kind
            j = i
            while j < len(tokens) and tokens[j].kind is kind:
                j += 1
            run = tokens[i:j]
            for k, tok in enumerate(run):
                # A leading space marks a word boundary, which the very first token lacks.
                text = tok.text if (first and i == 0 and k == 0) else " " + tok.text
                ids = self._model.tokenize(text, add_bos=(first and i == 0 and k == 0))
                if not ids:
                    ids = [self._pad_id]
                out_tokens.append(tok)
                out_ids.extend(ids)
                out_spans.append(len(ids))
            i = j
        return out_tokens, out_ids, out_spans

    # -- the five ops --------------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        if self._chat_template == "chatml" and not ctx.turn_open and ctx.tokens:
            # Everything injected so far was the prompt, so close that turn and open the
            # model's own.
            ctx.turn_index = len(ctx.tokens) - 1
            self._frame_into(ctx, self._CHATML_TURN, ctx.turn_index, before=False)
            ctx.turn_open = True
        self._ensure_logits(ctx)

        tid = llama_cpp.llama_sampler_sample(ctx.sampler, self._ctx, -1)

        if llama_cpp.llama_vocab_is_eog(self._model.vocab, tid):
            # End of generation is an EXIT request the kernel checks, not an exception.
            return DecodeResult(
                tokens=(),
                request=MachineRequest(op=OpKind.EXIT),
                attention=None,
                attention_hint=AttentionHint(tags=ctx.tags),
            )
        if tid in self._reserved and not allow_control:
            raise ControlTokenViolation(
                f"job {job} sampled reserved id {tid} while control tokens were disabled"
            )

        piece = self._model.piece(tid)
        token = Token(piece, TokenKind.CONTROL if tid in self._reserved else TokenKind.NORMAL)
        ctx.spoken = True

        before = len(ctx.tokens)
        ctx.tokens.append(token)
        ctx.ids.append(tid)
        ctx.spans.append(1)
        after = len(ctx.tokens)

        request = self.consume(job, ctx.parser, piece)
        if request.op is not OpKind.NONE and request.op is not OpKind.EXIT:
            self._reset_grammar(ctx)

        return DecodeResult(
            tokens=(token,),
            request=request,
            # llama.cpp gives no attention weights, so the hint channel carries a guess instead.
            attention=None,
            attention_hint=AttentionHint(tags=ctx.tags),
            at_block_boundary=after % self._block_size == 0 and after != before,
        )

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        ctx = self._ctx_of(job)
        start = len(ctx.tokens)
        first = not ctx.tokens
        new_tokens, new_ids, new_spans = self._encode(tokens, ctx)
        ctx.tokens.extend(new_tokens)
        ctx.ids.extend(new_ids)
        ctx.spans.extend(new_spans)
        # An arrival lands mid-flight when the model's turn is open; anything before that
        # is still the prompt being put together.
        if new_tokens and self._chat_template == "chatml":
            if first:
                self._frame_into(ctx, self._CHATML_OPEN, start, before=True)
            elif ctx.turn_open:
                # Close the job's turn and let the sender speak; ``decode`` reopens the
                # job's turn later, so a run of arrivals shares one turn.
                self._frame_into(ctx, self._CHATML_ARRIVE, start, before=True)
                ctx.turn_open = False
                ctx.turn_index = None
        # Reporting keys on ``spoken``, not on the framing above, because a run of arrivals
        # shares one turn but each one still arrived.
        if ctx.spoken and new_tokens:
            self.note_arrival(job, " ".join(t.text for t in new_tokens))
        # New content is the most likely thing the next step attends to.
        if new_tokens:
            ctx.tags = ("self",)
        return start, len(ctx.tokens)

    def trunc(self, job: JobId, at: int) -> int:
        ctx = self._ctx_of(job)
        if at < 0 or at > len(ctx.tokens):
            raise IndexError(f"trunc at {at} outside [0, {len(ctx.tokens)}]")
        dropped = len(ctx.tokens) - at
        if dropped == 0:
            return 0
        kv_at = ctx.kv_offset(at)
        self._rewind(ctx, kv_at)
        del ctx.tokens[at:]
        del ctx.ids[kv_at:]
        del ctx.spans[at:]
        if ctx.turn_index is not None and at <= ctx.turn_index:
            ctx.turn_index = None
            ctx.turn_open = False
        return dropped

    def fork(self, parent: JobId, child: JobId) -> int:
        src = self._ctx_of(parent)
        self._flush(src, want_logits=False)
        existing = self._contexts.get(child)
        if existing is None:
            self.create_context(child, src.descriptor)
            existing = self._ctx_of(child)
        else:
            llama_cpp.llama_memory_seq_rm(self._mem, existing.seq, -1, -1)
        llama_cpp.llama_memory_seq_cp(self._mem, src.seq, existing.seq, 0, -1)
        existing.tokens = list(src.tokens)
        existing.ids = list(src.ids)
        existing.spans = list(src.spans)
        existing.n_in_kv = src.n_in_kv
        existing.turn_open = src.turn_open
        existing.turn_index = src.turn_index
        existing.spoken = src.spoken
        # A child starts from the parent's visibility and can only lose ground from there.
        existing.mask = src.mask
        if self._logits_owner == existing.seq:
            self._logits_owner = None
        return len(src.tokens)

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        """Replace a range of tokens, the one operation that moves the offsets after it."""
        ctx = self._ctx_of(job)
        if not 0 <= start <= end <= len(ctx.tokens):
            raise IndexError(f"splice [{start}, {end}) outside context of {len(ctx.tokens)}")
        downstream = len(ctx.tokens) - end

        kv_start = ctx.kv_offset(start)
        kv_end = ctx.kv_offset(end)
        tail_tokens = ctx.tokens[end:]
        tail_ids = ctx.ids[kv_end:]
        tail_spans = ctx.spans[end:]

        head = _Context(seq=ctx.seq, descriptor=ctx.descriptor)
        head.ids = ctx.ids[:kv_start]
        new_tokens, new_ids, new_spans = self._encode(tokens, head)

        self._rewind(ctx, kv_start)
        ctx.tokens[start:] = new_tokens + tail_tokens
        ctx.ids[kv_start:] = new_ids + tail_ids
        ctx.spans[start:] = new_spans + tail_spans
        return SpliceResult(tokens_in=len(new_tokens), invalidated_downstream=downstream)

    # -- the serving-stack contract ------------------------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        """Install the allowed-block bitmap, refusing one that narrows what the job can see."""
        ctx = self._ctx_of(job)
        if self._enforce_mask:
            live = frozenset(range(self._block_count(len(ctx.tokens))))
            if not live <= allowed_blocks:
                raise MaskViolation(
                    f"job {job}: this backend cannot enforce an allowed-block bitmap "
                    f"(llama.cpp exposes no per-block attention mask), so it refuses to "
                    f"accept one that narrows {sorted(live)} to "
                    f"{sorted(live & allowed_blocks)}. See ZEOS-AM §8.1."
                )
        ctx.mask = allowed_blocks

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        ctx = self._ctx_of(job)
        # No mask means every block, which is not the same as a mask allowing none.
        every = frozenset(range(self._block_count(len(ctx.tokens))))
        return every if ctx.mask is None else every & ctx.mask

    def pad_to_block(self, job: JobId) -> int:
        ctx = self._ctx_of(job)
        remainder = len(ctx.tokens) % self._block_size
        if remainder == 0:
            return 0
        padding = self._block_size - remainder
        ctx.tokens.extend([PAD_TOKEN] * padding)
        ctx.ids.extend([self._pad_id] * padding)
        ctx.spans.extend([1] * padding)
        return padding

    def blocks_for_range(self, job: JobId, start: int, end: int) -> frozenset[int]:
        self._ctx_of(job)
        if end <= start:
            return frozenset()
        return frozenset(range(start // self._block_size, (end - 1) // self._block_size + 1))

    def transcript(self, job: JobId) -> tuple[Token, ...]:
        return tuple(self._ctx_of(job).tokens)

    # -- outside the Protocol -------------------------------------------------

    def lines(self, job: JobId) -> tuple[str, ...]:
        """The syscall lines this job has produced, for display only."""
        return tuple(self._ctx_of(job).parser.lines)

    def close(self) -> None:
        for job in list(self._contexts):
            self.destroy_context(job)
        llama_cpp.llama_batch_free(self._batch)
        llama_cpp.llama_free(self._ctx)
