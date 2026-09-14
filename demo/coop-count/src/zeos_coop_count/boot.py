# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Building the llama backend for a case, ready to hand to ``zeos.driver.build_kernel``."""

from __future__ import annotations

from zeos.descriptor.loader import CaseBundle
from zeos.machine.seat import seat_maps

from zeos_coop_count.machine import DEFAULT_BLOCK_SIZE, LlamaMachine, LlamaModel

__all__ = ["llama_machine"]


def llama_machine(
    bundle: CaseBundle,
    model: LlamaModel,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_ctx: int = 8192,
    n_threads: int = 8,
    **machine_kwargs: object,
) -> LlamaMachine:
    """A llama seat sized for the case: one sequence per descriptor, and a grammar each."""
    descriptors, valued = seat_maps(bundle.descriptors, bundle.pipes)
    return LlamaMachine(
        model,
        descriptors=descriptors,
        valued=valued,
        block_size=block_size,
        n_ctx=n_ctx,
        n_seq_max=max(8, len(bundle.descriptors) + 2),
        n_threads=n_threads,
        # This case has no compartments, so every mask narrowing here comes from paging.
        enforce_mask=False,
        **machine_kwargs,  # pyright: ignore[reportArgumentType]
    )
