# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Two jobs, counting to five on repeat -- the smallest thing the kernel will run.

Drives the kernel directly, one ``tick()`` per loop iteration, so there is nothing
between a breakpoint in the loop below and the kernel itself. Each tick prints the
journal events that tick produced, and under every decode, the tokens it appended.

    uv run python demo/counter/run.py
    uv run python demo/counter/run.py --ticks 8
    uv run python demo/counter/run.py --machine claude

A "lap" is one whole job: it counts to five, spawns a fresh instance of itself, and
exits. The default budget is two laps -- one each for the two jobs the case boots --
because the interesting moment is the handover between them.

The machine seat is swappable. ``--machine scripted`` (the default) replays the
case's script and needs nothing beyond zeos; ``--machine claude`` puts a live model
in the seat (``claude_machine.ClaudeMachine``, needing ``anthropic`` and an API
key); ``--machine claude-code`` puts the same live model in the seat by shelling
out to a local, already-authenticated ``claude --print`` (``claude_code_machine.
ClaudeCodeMachine``), so it needs a `claude` login instead of a separate API key.
Either way each tick's content -- including when to spawn and exit -- is the
model's choice, and the run is no longer deterministic. The kernel cannot tell the
difference, which is the point.

Worth a breakpoint:

    Kernel.tick             core/kernel.py     one token boundary, start to finish
    Scheduler.best_ready    core/scheduler.py  which job is chosen, and why
    Kernel._handle_request  core/kernel.py     where 'spawn' and 'exit' are serviced
    Kernel._complete        core/kernel.py     the end of a lap
    ClaudeMachine.decode    claude_machine.py  the model answering for the script
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from zeos.core.events import Decoded, Event
from zeos.core.ids import ObjectName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.loader import load_case
from zeos.journal.writer import Journal
from zeos.machine.base import MachineBackend
from zeos.machine.scripted import ScriptedMachine
from zeos.monitor.state import render_event
from zeos.world.store import WorldStore

DEMO = Path(__file__).resolve().parent

#: Eight boundaries per lap: the dispatch, five numbers, the spawn, the exit.
#: The scripted machine hits this exactly; a live model may take more.
TICKS_PER_LAP = 8

#: Virtual wall-clock cost of one boundary. Time is the driver's job, never the
#: kernel's -- and in this script the driver is the loop in ``main``. The real
#: seconds a live API call takes never reach the kernel either.
NS_PER_TICK = 1_000_000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticks",
        type=int,
        default=2 * TICKS_PER_LAP,
        help="token boundaries to run (default: two laps)",
    )
    parser.add_argument(
        "--machine",
        choices=("scripted", "claude", "claude-code", "qwen"),
        default="scripted",
        help="what answers each decode: the case's script, the Claude API, a "
        "local 'claude --print' process, or a local llama.cpp Qwen",
    )
    parser.add_argument(
        "--journal",
        default=None,
        help="also write the journal here (JSONL), for zeos debug and zeos replay",
    )
    parser.add_argument(
        "--case",
        default="simple_count",
        help="which case directory in this demo to run (default: simple_count)",
    )
    args = parser.parse_args(argv)

    bundle = load_case(DEMO / args.case)
    machine: MachineBackend
    if args.machine == "claude":
        from claude_machine import ClaudeMachine

        machine = ClaudeMachine()
    elif args.machine == "claude-code":
        from claude_code_machine import ClaudeCodeMachine

        machine = ClaudeCodeMachine()
    elif args.machine == "qwen":
        from qwen_machine import QwenMachine

        machine = QwenMachine()
    else:
        machine = ScriptedMachine(bundle.scripts)

    # ``build_kernel`` hardwires the scripted machine, so the same parts are
    # assembled here around whichever machine was chosen.
    events: list[Event] = []
    world = WorldStore()
    kernel = Kernel(
        descriptors=bundle.descriptors,
        machine=machine,
        pipes=PipeTable(bundle.pipes),
        vectors=VectorTable(bundle.vectors),
        world=world,
        resources=ResourceTable(bundle.resources),
        platforms=bundle.platforms,
        principals=bundle.principals,
        gates=bundle.gates,
        journal_sink=events,
        config=KernelConfig(case=args.case, max_ticks=args.ticks),
    )
    for obj, value in sorted(bundle.world.items()):
        world.set(ObjectName(obj), value, at=kernel.clock)

    kernel.start()
    for name in bundle.boot:
        kernel.spawn(name)

    # Flushed tick by tick, as the Driver does, so an aborted run -- a live model
    # can hang or refuse -- still leaves an analysable journal behind.
    journal = Journal(Path(args.journal)) if args.journal else None

    seen = 0
    for tick in range(args.ticks):
        kernel.advance_time(tick * NS_PER_TICK)
        runnable = kernel.tick()
        for event in events[seen:]:
            print(f"  t{tick:<4} {render_event(event)}")
            if isinstance(event, Decoded):
                print(f'  t{tick:<4}   decoded "{" ".join(event.text)}"')
        if journal is not None:
            journal.extend(events[seen:])
        seen = len(events)
        if not runnable:
            print(f"  t{tick:<4} nothing runnable")
            break

    print(f"\n{len(events)} journal events over {args.ticks} boundaries")
    if journal is not None:
        journal.close()
        print(f"journal written to {args.journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
