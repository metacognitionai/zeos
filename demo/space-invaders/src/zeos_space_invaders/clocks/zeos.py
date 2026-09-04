# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The wall clock again, with the zeos kernel deciding.

A runner of its own because the other two ask a player for one action and
apply it, whereas `ZeosDriver.step` *is* the tick: it delivers sensor readings
into pipes, pumps the kernel, applies the actuator and publishes the world.
"""

import threading
import time

from ..game import ACTIONS, Controls, Game, snapshot
from .outcome import outcome_of
from .render import board, played


def _mean_gap(ticks):
    """Mean distance between consecutive entries, or 0.0 with fewer than two."""
    marks = [t for t in ticks if t is not None]
    if len(marks) < 2:
        return 0.0
    return round((marks[-1] - marks[0]) / (len(marks) - 1), 2)


class ZeosRealtimeRunner:
    """One episode driven by the zeos kernel instead of one prompt-response pair.

    Single-threaded, unlike `RealtimeRunner`: the machine runs each completion
    on its own thread and hands the kernel one piece at a time, so this loop
    never blocks on the model, only on the tick interval.
    """

    def __init__(
        self,
        driver,
        view,
        seed=None,
        tick_seconds=0.2,
        max_ticks=500,
        max_seconds=600,
        rules=None,
        render=False,
        actions_per_tick=None,
    ):
        self.driver = driver
        self.view = view
        self.game = Game(seed=seed, rules=rules)
        # The same stick the other clocks write to; the per-tick budget is the
        # game's rule whoever is driving.
        self.controls = Controls(self.game, per_tick=actions_per_tick)
        # The driver applies a write inside its own pump: applied here after
        # `step` returns, the ship has not moved when the resume diff is taken.
        driver.controls = self.controls
        self.tick_seconds = tick_seconds
        self.max_ticks = max_ticks
        self.max_seconds = max_seconds
        self.render = render
        self.records = []
        self.lock = threading.Lock()  # guards every touch of `game`
        self.ticked = threading.Condition(self.lock)
        self.finished = threading.Event()
        self.writer = None
        #: The clock thread's reading, waiting for the main thread to deliver it.
        #: One slot, not a queue: the board is a level-triggered sensor and only
        #: the newest reading is worth having.
        self._pending = None
        #: The move played since the last tick, cleared by the tick that draws it.
        self._last = None
        #: When the next tick falls due, so a kernel batch on the main thread is
        #: bounded by it rather than by a boundary count.
        self._due = None

    def _draw(self, decision):
        """Called under the lock, from the clock thread, so frames never tear."""
        print(
            board(
                self.game,
                played(
                    self.game,
                    decision.action if decision else None,
                    by=decision.by if decision else None,
                    flag="  PREEMPTED" if decision and decision.preempted else "",
                ),
            ),
            flush=True,
        )

    def _clock(self):
        """Pace the world, and hand the kernel this tick's reading.

        Its own thread, so the world ticks on time whether or not the kernel is
        busy. The reading is handed over rather than delivered because `deliver`
        mutates the kernel, and the kernel is the main thread's alone.
        """
        due = time.monotonic()
        while not self.finished.is_set():
            due += self.tick_seconds
            if self.finished.wait(max(0.0, due - time.monotonic())):
                return
            with self.lock:
                self.controls.tick()
                board, info = self.game.render(), snapshot(self.game)
                self._pending = (
                    self.view.state(board, info),
                    info,
                    # Rendered here because the driver knows nothing about views.
                    self.view.history(board, info),
                )
                self._due = due + self.tick_seconds
                over = self.game.over or self.game.ticks >= self.max_ticks
                if self.writer:
                    self.writer.frame(snapshot(self.game), over=self.game.over)
                if self.render:
                    self._draw(self._last)
                self._last = None
                self.ticked.notify_all()
            if over:
                self.finished.set()
                return

    def run(self, run=None):
        started = time.monotonic()
        self.writer = run
        if run:
            run.frame(snapshot(self.game))
        # Boot before the world starts ticking, and journal the boot as tick 0.
        # Booting inside the first batch stamped the kernel's first events with
        # whatever tick the clock had reached by the end of that batch, which at
        # a 1ms tick was sometimes 1.
        self.driver.start()
        if run:
            run.kernel(0, self.driver.journal())
        clock = threading.Thread(target=self._clock, daemon=True)
        clock.start()

        while not self.finished.is_set():
            # Delivered from here so the kernel is only ever touched from this
            # thread.
            with self.lock:
                pending, self._pending = self._pending, None
                due = self._due
            if pending is not None:
                self.driver.sense(*pending)

            batch = time.monotonic()
            decision = self.driver.run_kernel(deadline=due or batch + self.tick_seconds)
            with self.lock:
                tick = self.game.ticks
                if decision is not None:
                    decision.applied = self.driver.last_applied
                    if decision.tick is None:
                        decision.tick = tick
                    decision.tick_applied = tick
                    # Only the reflex reaches this: the pilot's latency is
                    # board-to-move, and only the driver sees the board arrive.
                    if decision.latency is None:
                        decision.latency = round(time.monotonic() - batch, 4)
                    self._last = decision
                    self.records.append(decision)
                # A batch that produced nothing and returned early means the
                # kernel is quiescent; re-asking at full speed would starve the
                # machine's producer threads, the ones waiting on the model.
                elif not self.finished.is_set():
                    self.ticked.wait(timeout=self.tick_seconds)
            if run:
                if decision is not None:
                    run.decision(decision)
                run.kernel(tick, self.driver.journal())

            if time.monotonic() - started > self.max_seconds:
                self.finished.set()

        self.finished.set()
        clock.join(timeout=2 * self.tick_seconds)
        return self._summary(time.monotonic() - started)

    def _summary(self, elapsed):
        game, records = self.game, self.records
        moves = [d for d in records if d.by == "pilot"]
        driver, machine = self.driver, self.driver.machine
        return {
            **outcome_of(game, records, elapsed, over="timeout"),
            # Over the pilot's moves only: the reflex is not an inference, and
            # averaging it in would flatter the run.
            "mean_latency": round(sum(d.latency for d in moves) / len(moves), 3)
            if moves
            else 0.0,
            # World ticks between one pilot move and the next, which is what the
            # other clocks get from wall-clock latency.
            "mean_ticks_waited": _mean_gap([d.tick_applied for d in moves]),
            # A board is delivered on a tick edge, so a turn answered in 1.4
            # ticks still costs 2; naming that rounding keeps it out of "the
            # kernel is slow".
            "tick_quantisation": round(
                _mean_gap([d.tick_applied for d in moves])
                - (sum(d.latency for d in moves) / len(moves) / self.tick_seconds),
                2,
            )
            if moves
            else 0.0,
            # The server's share of `mean_latency`: request out to first piece in.
            "mean_ttft": round(sum(machine.ttfts) / len(machine.ttfts), 3)
            if machine.ttfts
            else 0.0,
            # A schema makes this unproducible, so a non-zero count says the
            # server did not honour the schema it was given.
            "unparseable": sum(1 for d in moves if d.action not in ACTIONS),
            "pilot_moves": len(moves),
            "reflexes": driver.reflexes,
            # Read off the journal: the kernel's own claim that it took the
            # machine away from a running pilot.
            "preemptions": driver.preemptions,
            # Two numbers on purpose: `usage` is what the server reported, and a
            # cancelled completion reports none; `decoded_words` is what arrived
            # on the wire.
            "usage": dict(getattr(machine, "usage", {}) or {}),
            "decoded_words": machine.words,
            # What the reflex produced, which reached no model and cost nothing.
            "native_words": machine.native_words,
            # Reasoning: in `T` and charged to the budget, never sent back.
            "thinking_words": machine.thinking_words,
            # A cancellation is not a failure: it is what a new board *is*, seen
            # from the machine.
            "generations": machine.generations,
            "cancellations": machine.cancellations,
            # Decoded output of a cancelled completion that amounted to no
            # syscall: kept in `T`, not transmitted again.
            "voided": machine.voided,
            # Judged by zeos against `criteria.yaml`; part of the outcome, so
            # here and not in `meta.json`.
            "criteria": driver.verdicts(),
        }
