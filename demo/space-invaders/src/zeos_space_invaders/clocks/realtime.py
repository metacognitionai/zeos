# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The wall clock: the world ticks on its own thread and never waits.

An action lands whenever the decision returns, so thinking is paid for in ticks.
A model that needs three seconds a move watches fifteen ticks go by between
decisions, and that is the point.
"""

import threading
import time

from ..game import Controls, Game, snapshot
from .outcome import outcome_of, usage_of
from .render import board, played


class RealtimeRunner:
    def __init__(
        self,
        player,
        seed=None,
        tick_seconds=0.2,
        max_ticks=2000,
        max_seconds=600,
        rules=None,
        render=False,
        actions_per_tick=None,
    ):
        self.player = player
        self.game = Game(seed=seed, rules=rules)
        # The stick, not `game.act`: how many moves one write makes is the
        # game's rule.
        self.controls = Controls(self.game, per_tick=actions_per_tick)
        self.tick_seconds = tick_seconds
        self.max_ticks = max_ticks
        self.max_seconds = max_seconds
        self.render = render
        self.lock = threading.Lock()  # guards every touch of `game`
        self.ticked = threading.Condition(self.lock)
        self.finished = threading.Event()
        #: What was played since the last tick, and nothing once it has ticked.
        self.last_action = None
        self.writer = None
        self.records = []

    def _clock(self):
        due = time.monotonic()
        while not self.finished.is_set():
            due += self.tick_seconds
            remaining = due - time.monotonic()
            if self.finished.wait(max(0.0, remaining)):
                return
            with self.lock:
                self.controls.tick()
                self.last_action = None
                over = self.game.over or self.game.ticks >= self.max_ticks
                # Written from this thread, which is why the writer locks; a
                # slow agent leaves dozens of ticks between two decisions, and
                # without a frame each the log has no timeline.
                if self.writer:
                    self.writer.frame(snapshot(self.game), over=self.game.over)
                if self.render:
                    self._draw()
                self.ticked.notify_all()
            if over:
                self.finished.set()
                return

    def _draw(self):
        """Called under the lock, from the clock thread, so frames never tear."""
        extra = f"  (x{self.controls.used} this tick)" if self.controls.used > 1 else ""
        print(
            board(self.game, played(self.game, self.last_action) + extra),
            flush=True,
        )

    def run(self, run=None):
        started = time.monotonic()
        self.writer = run
        if run:
            run.frame(snapshot(self.game))
        clock = threading.Thread(target=self._clock, daemon=True)
        clock.start()

        while not self.finished.is_set():
            with self.lock:
                board, info = self.game.render(), snapshot(self.game)
            ticks_before = info["ticks"]

            decided_at = time.monotonic()
            action, decision = self.player.choose(board, info)
            latency = time.monotonic() - decided_at

            with self.lock:
                # An LLM call is too expensive to discard, so a spent budget is
                # waited out; `full` rather than a retried `write`, because an
                # ended episode is the other refusal and is not worth waiting on.
                while (
                    self.controls.full
                    and not self.game.over
                    and not self.finished.is_set()
                ):
                    self.ticked.wait(timeout=2 * self.tick_seconds)
                # The game may have ended while the agent was thinking.
                applied = self.controls.write(action)
                if applied:
                    self.last_action = action
                    # Drawing solely from the clock thread would make a fast
                    # agent look laggy.
                    if self.render:
                        self._draw()
                ticks_after = self.game.ticks

            # `tick` and `tick_applied` differ whenever the agent is slow or
            # throttled; the gap is what a move costs in ticks.
            decision.tick = ticks_before
            decision.tick_applied = ticks_after
            decision.latency = round(latency, 4)
            decision.applied = applied
            self.records.append(decision)
            if run:
                run.decision(decision)

            if time.monotonic() - started > self.max_seconds:
                self.finished.set()

        self.finished.set()
        clock.join(timeout=2 * self.tick_seconds + 1)
        return self._summary(time.monotonic() - started)

    def _summary(self, elapsed):
        decisions = len(self.records)
        waited = [d.tick_applied - d.tick for d in self.records]
        latencies = [d.latency for d in self.records]
        return {
            **outcome_of(self.game, self.records, elapsed, over="timeout"),
            "mean_latency": round(sum(latencies) / decisions, 3) if decisions else 0.0,
            "mean_ticks_waited": round(sum(waited) / decisions, 2)
            if decisions
            else 0.0,
            # A reply that named no move; the kernel arm means something else by
            # this word -- see `zeos.py`.
            "unparseable": sum(1 for d in self.records if not d.parsed),
            "usage": usage_of(self.records),
        }
