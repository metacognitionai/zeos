# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Whether the world waits for the agent.

Choosing between the three runners is choosing what a run measures:
`run_episode` waits for the agent, `RealtimeRunner` never does, and
`ZeosRealtimeRunner` is the same clock with the zeos kernel deciding.
"""

from .realtime import RealtimeRunner
from .render import CLEAR
from .step import run_episode
from .zeos import ZeosRealtimeRunner

__all__ = ["CLEAR", "RealtimeRunner", "ZeosRealtimeRunner", "run_episode"]
