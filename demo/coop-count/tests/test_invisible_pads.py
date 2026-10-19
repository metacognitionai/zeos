# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Block padding keeps its slot in the kernel's token sequence but never reaches the
model: a pad occupies no llama id. Needs the weights; skips without them."""

from __future__ import annotations

from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import Token, tokens_from_text

from zeos_coop_count import model as model_mod
from zeos_coop_count.machine import PAD_TOKEN, LlamaMachine

JOB = JobId(1)
PROMPT = "say the numbers one after another"


def _machine(llama_model, machines) -> LlamaMachine:  # pyright: ignore[reportMissingParameterType]
    m = LlamaMachine(
        llama_model,
        descriptors={"d": ("stdin", "stdout")},
        block_size=16,
        n_ctx=2048,
        n_seq_max=2,
        n_threads=model_mod.DEFAULT_THREADS,
    )
    machines.append(m)
    m.create_context(JOB, "d")
    return m


def test_pads_never_reach_the_model(llama_model, machines) -> None:  # pyright: ignore[reportMissingParameterType]
    m = _machine(llama_model, machines)
    ctx = m._ctx_of(JOB)  # pyright: ignore[reportPrivateUsage]
    start, end = m.inject(JOB, tokens_from_text(PROMPT))
    assert end % m.block_size != 0
    ids_before = list(ctx.ids)

    padded = m.pad_to_block(JOB)
    assert padded == m.block_size - end % m.block_size
    assert m.stats(JOB).resident_tokens == end + padded
    assert m.transcript(JOB)[end:] == (PAD_TOKEN,) * padded
    assert ctx.kv_offset(len(ctx.tokens)) == len(ids_before)
    assert ctx.ids == ids_before
    window = m.raw(JOB)
    assert window.model_tokens == len(ids_before)
    assert all(word.pieces == () and word.framing == () for word in window.words[end:])

    stub = Token("<STUB", TokenKind.CONTROL)
    spliced = m.splice(JOB, start + 1, start + 3, (stub, PAD_TOKEN, PAD_TOKEN))
    assert spliced.tokens_in == 3
    assert ctx.spans[start + 1] > 0
    assert ctx.spans[start + 2 : start + 4] == [0, 0]
    assert m.transcript(JOB)[start + 2 : start + 4] == (PAD_TOKEN, PAD_TOKEN)
    assert len(ctx.ids) == sum(ctx.spans)
    window = m.raw(JOB)
    assert window.model_tokens == len(ctx.ids)
    assert all(word.pieces == () for word in window.words[start + 2 : start + 4])
    assert "pad" not in "".join(piece for word in window.words for piece in word.pieces)

    # The model's next step runs on exactly those ids, pads and all.
    result = m.decode(JOB, allow_control=False)
    assert m.raw(JOB).kv_resident == len(ctx.ids) - len(result.tokens)


def test_the_turn_opens_on_a_trailing_pad(llama_model, machines) -> None:  # pyright: ignore[reportMissingParameterType]
    """The decode that opens the model's turn frames the last kernel token; when that
    token is a pad the marker still follows the last word's ids with nothing between."""
    m = _machine(llama_model, machines)
    ctx = m._ctx_of(JOB)  # pyright: ignore[reportPrivateUsage]
    _, end = m.inject(JOB, tokens_from_text(PROMPT))
    padded = m.pad_to_block(JOB)
    assert padded > 0
    ids_before = list(ctx.ids)

    result = m.decode(JOB, allow_control=False)
    turn = llama_model.tokenize(LlamaMachine._CHATML_TURN, add_bos=False, parse_special=True)  # pyright: ignore[reportPrivateUsage]
    assert ctx.turn_index == end + padded - 1
    assert ctx.ids[: len(ids_before) + len(turn)] == ids_before + turn
    assert ctx.ids[ctx.kv_offset(end) : ctx.kv_offset(end + padded)] == turn
    assert ctx.spans[end : end + padded] == [0] * (padded - 1) + [len(turn)]

    window = m.raw(JOB)
    assert all(word.pieces == () for word in window.words[end : end + padded])
    marker = window.words[end + padded].framing if result.tokens else window.trailing
    assert "".join(marker) == LlamaMachine._CHATML_TURN  # pyright: ignore[reportPrivateUsage]


def test_an_arrival_starts_on_a_new_line_whatever_the_padding(llama_model, machines) -> None:  # pyright: ignore[reportMissingParameterType]
    """What the model reads of an arrival does not depend on the padding before it: the
    pads stay invisible and the arrival starts a new line of the prompt, or opens the
    sender's turn when it interrupts the job's own."""
    m = _machine(llama_model, machines)
    ctx = m._ctx_of(JOB)  # pyright: ignore[reportPrivateUsage]
    _, end = m.inject(JOB, tokens_from_text(PROMPT))
    padded = m.pad_to_block(JOB)
    start, _ = m.inject(JOB, tokens_from_text("<STATUS count> 7 </STATUS>"))
    assert ctx.spans[end:start] == [0] * padded
    window = m.raw(JOB)
    assert all(word.pieces == () for word in window.words[end:start])
    assert window.words[start].framing == ()
    assert "".join(window.words[start].pieces).startswith("\n<")

    m.decode(JOB, allow_control=False)
    assert ctx.turn_open
    m.pad_to_block(JOB)
    start, _ = m.inject(JOB, tokens_from_text("go"))
    word = m.raw(JOB).words[start]
    assert "".join(word.framing) == LlamaMachine._CHATML_ARRIVE  # pyright: ignore[reportPrivateUsage]
    assert "".join(word.pieces) == "go"
