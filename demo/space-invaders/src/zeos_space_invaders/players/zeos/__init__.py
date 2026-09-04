# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The scheduled player: a two-job descriptor tree the ZEOS kernel runs.

`cases/space-invaders/` is its program -- the pipes, the vector table, the world
objects and the two descriptors -- and needs no code change to edit. `player.py`
couples that to a live game: the native reflex, the sensors, and the device
adapter. The machine the kernel decodes through is `api_machine.APIMachineBase`,
with `api_openai.py` and `api_claude.py` as the two wire formats.
"""

from .api_claude import ClaudeAPIMachine
from .api_machine import (
    ALIASES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_STALL_S,
    FORMATS,
    PAD_TOKEN,
    PARTIAL,
    VERBS,
    APIMachineBase,
    Native,
    Piece,
    parse_syscall,
)
from .api_openai import OpenAIAPIMachine, SDKMissing
from .player import (
    BLOCK_SIZE,
    CASE_ROOT,
    GAME_CONTROLS,
    GAME_STATE,
    GAME_THREATS,
    NEWLINE,
    NS_PER_TICK,
    REFLEX,
    REFLEX_HORIZON,
    RESUME,
    RESUME_END,
    Decision,
    ZeosDriver,
    build_kernel,
    build_machine,
    case_path,
    decode,
    dodge,
    encode,
    evade_behaviour,
    kernel_version,
    load_criteria,
    threat_reading,
)

__all__ = [
    "ALIASES",
    "BLOCK_SIZE",
    "CASE_ROOT",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_STALL_S",
    "FORMATS",
    "GAME_CONTROLS",
    "GAME_STATE",
    "GAME_THREATS",
    "NEWLINE",
    "NS_PER_TICK",
    "PAD_TOKEN",
    "PARTIAL",
    "REFLEX",
    "REFLEX_HORIZON",
    "RESUME",
    "RESUME_END",
    "VERBS",
    "APIMachineBase",
    "ClaudeAPIMachine",
    "Decision",
    "Native",
    "OpenAIAPIMachine",
    "Piece",
    "SDKMissing",
    "ZeosDriver",
    "build_kernel",
    "build_machine",
    "case_path",
    "decode",
    "dodge",
    "encode",
    "evade_behaviour",
    "kernel_version",
    "load_criteria",
    "parse_syscall",
    "threat_reading",
]
