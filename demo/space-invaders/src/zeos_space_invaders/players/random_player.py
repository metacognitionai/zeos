# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A player that picks uniformly at random.

Same interface as the model players, so the loop, the logging and the summary
can be exercised without an API key. Also the baseline any model has to beat.
"""

import random
import time

from ..game import ACTIONS
from ..runlog import Decision


class RandomPlayer:
    thinking = False

    def __init__(self, seed=None, actions=ACTIONS, latency=0.0):
        self.rng = random.Random(seed)
        self.actions = tuple(actions)
        self.latency = latency  # seconds it "spends deciding", for realtime runs

    def choose(self, obs, info):
        action = self.rng.choice(self.actions)
        if self.latency:
            time.sleep(self.latency)
        return action, Decision(tick=info["ticks"], by="random", action=action)
