# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Who is answering.

`RandomPlayer` is the baseline and the smoke test: no key, no server, no
network. `OpenAIPlayer` and `ClaudePlayer` are the same `PromptPlayer` with a
different backend, so a comparison between them measures the model rather than
the harness; the fourth kind, in `zeos/`, reaches one of these two from inside a
descriptor tree the kernel schedules, so what answers is a job, not a loop.
"""

from .base import EFFORTS, PromptPlayer, SDKMissing, ThinkingNotDisabled
from .claude import ClaudePlayer
from .openai_compat import OpenAIPlayer
from .random_player import RandomPlayer

__all__ = [
    "EFFORTS",
    "ClaudePlayer",
    "OpenAIPlayer",
    "PromptPlayer",
    "RandomPlayer",
    "SDKMissing",
    "ThinkingNotDisabled",
]
