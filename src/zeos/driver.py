# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The driver: everything the kernel refuses to do.

The kernel reads no clock, touches no file, and performs no I/O. Something has to,
and this is it. The driver owns:

* **time** -- it decides what "now" is and tells the kernel via ``advance_time``;
* **device adapters** -- turning external events into pipe writes (core §4.3);
* **transports** -- polling for anything arriving from a peer node;
* **the journal file** -- the kernel emits events, the driver persists them;
* **the machine's trace** -- after each tick, a machine that can account for its
  own windows is asked to, and the answer goes to a file beside the journal.

Keeping this boundary sharp is what makes the whole thing replayable. In a test the
schedule is a list; in deployment it is a sensor feed; the kernel cannot tell the
difference, so a field incident replays as a test case.

Deliberately synchronous. M0 has no concurrency to manage -- one kernel, one machine,
a scripted event schedule -- and an async runtime here would buy nothing while making
the ordering of external events harder to reason about, which is precisely the thing
determinism depends on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zeos.core.events import Event
from zeos.core.ids import DescriptorName, ObjectName, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.loader import CaseBundle
from zeos.journal.writer import Journal
from zeos.machine.base import MachineBackend, Token, TracesRaw
from zeos.machine.scripted import ScriptedMachine
from zeos.trace import RawTrace
from zeos.transport.base import PipeTransport
from zeos.transport.local import LocalTransport
from zeos.world.store import WorldStore

__all__ = ["ScheduledEvent", "Driver", "load_schedule", "build_kernel"]


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """One external event: a device adapter writing to a pipe at a given time."""

    at_ns: int
    pipe: PipeName
    text: str

    @staticmethod
    def from_json(raw: dict[str, object], *, source: str, lineno: int) -> ScheduledEvent:
        try:
            return ScheduledEvent(
                at_ns=int(str(raw["at_ns"])),
                pipe=PipeName(str(raw["pipe"])),
                text=str(raw.get("text", "")),
            )
        except KeyError as exc:
            raise ValueError(
                f"{source}:{lineno}: schedule entry missing {exc.args[0]!r} "
                "(need 'at_ns' and 'pipe')"
            ) from exc


def load_schedule(path: Path) -> tuple[ScheduledEvent, ...]:
    """Read a JSONL event schedule.

    Sorted by time on load, and stably -- two events at the same instant keep their
    file order, so the schedule fully determines the run.
    """
    events: list[ScheduledEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{lineno}: schedule entry must be an object")
            events.append(
                ScheduledEvent.from_json(raw, source=str(path), lineno=lineno)  # pyright: ignore[reportUnknownArgumentType]
            )
    return tuple(sorted(events, key=lambda e: e.at_ns))


def build_kernel(
    bundle: CaseBundle,
    *,
    machine: MachineBackend | None = None,
    journal_sink: list[Event] | None = None,
    config: KernelConfig | None = None,
    block_size: int = 16,
) -> tuple[Kernel, PipeTransport]:
    """Assemble a kernel from a loaded case, on the case's own scripts unless a machine is given."""
    if machine is None:
        machine = ScriptedMachine(bundle.scripts, block_size=block_size)
    pipes = PipeTable(bundle.pipes)
    world = WorldStore()
    kernel = Kernel(
        descriptors=bundle.descriptors,
        machine=machine,
        pipes=pipes,
        vectors=VectorTable(bundle.vectors),
        world=world,
        resources=ResourceTable(bundle.resources),
        platforms=bundle.platforms,
        principals=bundle.principals,
        gates=bundle.gates,
        journal_sink=journal_sink,
        config=config or KernelConfig(case=bundle.name),
    )
    for obj, value in sorted(bundle.world.items()):
        world.set(ObjectName(obj), value, at=kernel.clock)
    return kernel, LocalTransport(pipes)


class Driver:
    """Runs a kernel against a schedule of external events."""

    #: Simulated wall-clock cost of one token boundary. The specs put a forward
    #: pass at "~ms" (core §5.2), and interrupt latency is budgeted against that,
    #: so this is the number that makes a deadline in a vector table mean anything
    #: in an M0 run. A simulation parameter like ``block_size`` -- replaced by
    #: measurement when a real machine arrives.
    DEFAULT_NS_PER_TICK = 1_000_000

    def __init__(
        self,
        kernel: Kernel,
        *,
        transport: PipeTransport | None = None,
        journal: Journal | None = None,
        trace: RawTrace | None = None,
        ns_per_tick: int = DEFAULT_NS_PER_TICK,
        on_drain: Callable[[PipeName, tuple[Token, ...]], None] | None = None,
    ) -> None:
        if trace is not None and not isinstance(kernel.machine, TracesRaw):
            raise TypeError(
                f"{type(kernel.machine).__name__} gives no account of its windows; "
                "a trace needs a machine that implements TracesRaw"
            )
        self.kernel = kernel
        self.transport = transport
        self.journal = journal
        self.trace = trace
        self.ns_per_tick = ns_per_tick
        self.on_drain = on_drain
        self._persisted = 0
        self._now_ns = 0

    def _flush(self) -> None:
        """Persist events the kernel has emitted since the last flush.

        Streaming rather than dumping at the end: a run killed mid-flight -- which is
        exactly what a thrash or starvation investigation looks like -- should still
        leave an analysable journal behind. Once per tick, because the machine's
        account is sampled here.
        """
        pending = self.kernel.events[self._persisted :]
        if self.journal is not None:
            self.journal.extend(pending)
        if self.trace is not None and isinstance(self.kernel.machine, TracesRaw):
            self.trace.sample(self.kernel.machine, pending, self._persisted)
        self._persisted += len(pending)

    def boot(self, descriptors: Sequence[DescriptorName]) -> None:
        self.kernel.start()
        for name in descriptors:
            self.kernel.spawn(name)
        self._flush()

    def run(self, schedule: Iterable[ScheduledEvent] = ()) -> int:
        """Run to quiescence, injecting scheduled events at their times.

        Ticks consume virtual time, so an event scheduled for 3ms arrives while a
        job is mid-flight rather than after everything has drained. Getting this
        wrong makes preemption structurally unobservable -- a driver that runs to
        quiescence before each event can only ever deliver interrupts to an idle
        kernel, which is the one case where the interrupt does not matter.
        """
        ticks = 0
        for event in sorted(schedule, key=lambda e: e.at_ns):
            ticks += self._run_until(event.at_ns)
            self.kernel.advance_time(max(event.at_ns, self._now_ns))
            self._now_ns = self.kernel.clock.virtual_ns
            self.kernel.deliver(event.pipe, event.text)
            self._flush()
        ticks += self._run_until(None)
        self._poll_transport()
        self._flush()
        return ticks

    def _run_until(self, deadline_ns: int | None) -> int:
        """Tick until quiescent, or until virtual time reaches ``deadline_ns``."""
        ticks = 0
        while ticks < self.kernel.config.max_ticks:
            if deadline_ns is not None and self._now_ns >= deadline_ns:
                break
            self.kernel.advance_time(self._now_ns)
            if not self.kernel.tick():
                break
            self._now_ns += self.ns_per_tick
            ticks += 1
            self._drain_sinks()
            self._flush()
        self._drain_sinks()
        self._flush()
        return ticks

    def _drain_sinks(self) -> None:
        """Take what jobs wrote for the world out of every sink, the mirror of ``deliver``."""
        for pipe in self.kernel.pipes.all():
            if pipe.spec.sink and pipe.available:
                tokens = self.kernel.drain(pipe.name)
                if self.on_drain is not None:
                    self.on_drain(pipe.name, tokens)

    def _poll_transport(self) -> None:
        """Drain anything arriving from a peer node.

        A no-op with ``LocalTransport``. It is called anyway so that the call site
        exists and is exercised -- the seam should not be discovered to be missing
        on the day a real transport arrives.
        """
        if self.transport is None:
            return
        for frame in self.transport.poll():
            self.kernel.deliver(frame.pipe, " ".join(t.text for t in frame.tokens))
