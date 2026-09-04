# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The move machine with a local llama.cpp model in the model seat.

See ``move_machine`` for the protocol; this file is only the asking. Where
``demo/coop-count`` builds a whole grammar-constrained syscall ABI around llama.cpp,
this stays deliberately naive -- the same free-text move protocol the Claude seat
uses, answered by a 4B quantised Qwen on this machine's CPU. Same prompt, much
smaller model: where the two seats diverge is the demo.

The weights live in this demo's own ``models/`` (``fetch_model.py`` puts them
there); ``ZEOS_COUNTER_MODEL`` points anywhere else.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from llama_cpp import Llama
from move_machine import SYSTEM, MoveMachine

from zeos.core.ids import JobId

__all__ = ["QwenMachine"]

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "Qwen3.5-4B-Q4_K_M.gguf"

#: Qwen3.5 thinks out loud between <think> tags before its answer; the move is
#: what remains. Stripped by text because llama.cpp's chat completion returns the
#: whole turn -- there is no display knob as on the API.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

#: Pinned: the thread count changes how the CPU backend partitions its reductions,
#: so logits differ in their last bits and greedy argmax occasionally flips -- it is
#: an experiment setting, not a performance knob.
THREADS = 32


class QwenMachine(MoveMachine):
    """``MoveMachine`` asking a local GGUF model for each move."""

    def __init__(self, *, model_path: Path | None = None, block_size: int = 16) -> None:
        super().__init__(block_size=block_size)
        path = model_path or Path(os.environ.get("ZEOS_COUNTER_MODEL") or DEFAULT_MODEL)
        if not path.exists():
            raise SystemExit(
                f"no model at {path}; run 'uv run python demo/counter/fetch_model.py' "
                "or point ZEOS_COUNTER_MODEL at a GGUF file"
            )
        self._llm = Llama(
            model_path=str(path), n_ctx=8192, n_threads=THREADS, seed=0, verbose=False
        )

    def _ask(self, job: JobId, prompt: str) -> str:
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            # Greedy: with the thread count pinned above, reruns repeat.
            temperature=0.0,
            # Room to think; the move itself is a few tokens.
            max_tokens=2048,
        )
        text = response["choices"][0]["message"]["content"] or ""  # type: ignore[index]
        return _THINK.sub("", text).strip()
