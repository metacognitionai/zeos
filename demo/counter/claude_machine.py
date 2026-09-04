# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The move machine with the Claude API in the model seat.

See ``move_machine`` for the protocol; this file is only the asking.
"""

from __future__ import annotations

from anthropic import Anthropic
from move_machine import SYSTEM, MoveMachine

from zeos.core.ids import JobId

__all__ = ["ClaudeMachine"]


class ClaudeMachine(MoveMachine):
    """``MoveMachine`` asking the Claude API for each move."""

    def __init__(self, *, model: str = "claude-opus-5", block_size: int = 16) -> None:
        super().__init__(block_size=block_size)
        # A long run makes one API call per boundary, so a transient 529 landing
        # somewhere in the middle is routine; the default two retries give up on it.
        self._client = Anthropic(max_retries=6)
        self._model = model

    def _ask(self, job: JobId, prompt: str) -> str:
        response = self._client.beta.messages.create(
            model=self._model,
            # Room for adaptive thinking; the visible move itself is a few tokens.
            max_tokens=4096,
            # Medium, not low: a move is chosen against rules of turn-taking, and
            # at low effort continuation bias wins over them -- a job mid-count
            # keeps counting instead of yielding the machine.
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused to produce a move for job {job}")
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise RuntimeError(
                f"no text in the response for job {job} (stop_reason={response.stop_reason})"
            )
        return text.strip()
