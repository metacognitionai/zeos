# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The step clock: one action, one tick, the world waits.

Deterministic, and therefore the mode a comparison runs in -- what it measures
is the decision, with latency taken out of the picture entirely.
"""

import time

from ..game import SpaceInvadersEnv
from .outcome import outcome_of, usage_of
from .render import board, played


def run_episode(
    player,
    seed=None,
    max_steps=500,
    run=None,
    verbose=True,
    render=False,
    delay=0.0,
    rules=None,
):
    env = SpaceInvadersEnv(seed=seed, max_steps=max_steps, rules=rules)
    obs, info = env.reset()
    started = time.monotonic()
    records, total_reward = [], 0.0
    terminated = truncated = False
    if run:
        run.frame(info)

    while not (terminated or truncated):
        action, decision = player.choose(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Said rather than implied, so a reader need not know which clock wrote
        # the run to join the two.
        decision.tick_applied = decision.tick
        decision.reward = reward
        records.append(decision)
        if run:
            run.decision(decision)
            run.frame(info, over=terminated)
        if render:
            mark = "" if decision.parsed else "  <- unparseable, fell back"
            print(
                board(env.game, played(env.game, action, flag=mark)),
                flush=True,
            )
            if delay:
                time.sleep(delay)
        elif verbose:
            mark = "" if decision.parsed else "  <- unparseable, fell back"
            print(
                f"  step {decision.tick:>3}  {action:<5}  "
                f"score {info['score']:>3}  lives {info['lives']}{mark}"
            )

    return {
        # `actions_per_tick` is 1.0 and `dropped_after_game_over` 0 by
        # construction, reported rather than omitted because a blank cell in
        # the index reads as "not measured".
        **outcome_of(env.game, records, time.monotonic() - started, over="truncated"),
        "reward": total_reward,
        # A reply that named no move.
        "unparseable": sum(1 for d in records if not d.parsed),
        "usage": usage_of(records),
    }
