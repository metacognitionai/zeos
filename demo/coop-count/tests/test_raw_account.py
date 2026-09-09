# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The llama backend's account of a window: pieces per kernel word, the chat framing
folded around them, and the cache mark. Needs the weights; skips without them."""

from __future__ import annotations

from zeos.core.ids import JobId
from zeos.machine.base import TracesRaw, tokens_from_text

from zeos_coop_count import model as model_mod
from zeos_coop_count.machine import LlamaMachine

JOB = JobId(1)


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


def test_the_account_covers_every_kernel_word_and_every_llama_id(llama_model, machines) -> None:  # pyright: ignore[reportMissingParameterType]
    m = _machine(llama_model, machines)
    m.inject(JOB, tokens_from_text("say the numbers one after another"))
    m.decode(JOB, allow_control=False)
    assert isinstance(m, TracesRaw)

    window = m.raw(JOB)
    assert len(window.words) == m.stats(JOB).resident_tokens
    assert window.model_tokens == len(m._ctx_of(JOB).ids)  # pyright: ignore[reportPrivateUsage]
    assert all(word.pieces for word in window.words)


def test_framing_is_told_apart_from_the_words(llama_model, machines) -> None:  # pyright: ignore[reportMissingParameterType]
    """The chat turn markers are folded into neighbouring words' spans, and the
    account has to pull them back out: the opener ahead of the first word, the turn
    marker ahead of the first decoded word."""
    m = _machine(llama_model, machines)
    m.inject(JOB, tokens_from_text("say the numbers one after another"))
    opened = m.raw(JOB)
    assert "".join(opened.words[0].framing).startswith("<|im_start|>")
    assert opened.trailing == ()

    m.decode(JOB, allow_control=False)
    spoken = m.raw(JOB)
    first_output = spoken.words[len(opened.words)]
    assert "<|im_start|>" in "".join(first_output.framing)
    assert "assistant" in "".join(first_output.framing)
    assert "".join(spoken.words[0].pieces) == "say"


def test_a_splice_moves_the_cache_mark_back(llama_model, machines) -> None:  # pyright: ignore[reportMissingParameterType]
    """On this hybrid model a cut anywhere empties the cache, which is the cost the
    account exists to make visible."""
    m = _machine(llama_model, machines)
    m.inject(JOB, tokens_from_text(" ".join(str(i) for i in range(40))))
    m.decode(JOB, allow_control=False)
    before = m.raw(JOB)
    # The token just sampled has no forward pass behind it yet; everything else has.
    assert before.kv_resident == before.model_tokens - 1

    m.splice(JOB, 10, 20, tokens_from_text("X"))
    after = m.raw(JOB)
    untouched = sum(len(w.pieces) + len(w.framing) for w in after.words[:10])
    assert after.kv_resident <= untouched, "the cache cannot outlive the cut"
    assert len(after.words) == len(before.words) - 9
