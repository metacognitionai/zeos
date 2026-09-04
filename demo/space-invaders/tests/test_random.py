# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The random baseline: no key, no server, no cost.

It is the smoke test for everything around a player as well as the line a model
has to beat, so it has to produce the same record shape the model players do.
"""

from stubs import state

from zeos_space_invaders.game import ACTIONS
from zeos_space_invaders.players import RandomPlayer


def test_only_ever_plays_legal_actions():
    p = RandomPlayer(seed=0)
    for step in range(300):
        action, record = p.choose("BOARD", state(step=step))
        assert action in ACTIONS
        assert record.parsed is True and record.usage == {}


def draws(player, n=40):
    """`n` actions from one player drawn repeatedly, because a fresh `RandomPlayer`
    per draw only ever yields its constant first action."""
    return [player.choose("B", state(step=i))[0] for i in range(n)]


def test_is_reproducible():
    assert draws(RandomPlayer(seed=5)) == draws(RandomPlayer(seed=5))
    assert len(set(draws(RandomPlayer(seed=5)))) > 1, "not actually drawing"


def test_a_different_seed_plays_differently():
    assert draws(RandomPlayer(seed=1)) != draws(RandomPlayer(seed=2))


def test_the_record_says_who_chose_and_costs_nothing():
    _, record = RandomPlayer(seed=0).choose("BOARD", state(step=4))
    assert record.by == "random" and record.tick == 4
    assert record.usage == {} and record.prompt is None


def test_latency_is_only_spent_when_asked_for():
    """A realtime run pays it in ticks; the step clock must not pay it at all."""
    import time

    started = time.monotonic()
    RandomPlayer(seed=0, latency=0.05).choose("B", state())
    assert time.monotonic() - started >= 0.05
