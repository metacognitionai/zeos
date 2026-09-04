# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``zeos-count``: fetch a model, lint a case, and run it under the llama backend."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from types import FrameType

from collections.abc import Sequence

from zeos.core.events import Event, JobBlocked, JobWoken
from zeos.core.ids import DescriptorName, JobId, PipeName
from zeos.core.kernel import KernelConfig
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.loader import load_case
from zeos.descriptor.schema import DescriptorError
from zeos.driver import Driver, load_schedule
from zeos.journal.writer import Journal
from zeos.machine.base import MachineRequest, OpKind, render

from zeos_coop_count import model as model_mod
from zeos_coop_count.boot import build_kernel, seat_maps
from zeos_coop_count.keyboard import Console
from zeos_coop_count.machine import LlamaModel
from zeos_coop_count.seat import CommandSeat, CommandSource

__all__ = ["main"]


def _lint(bundle) -> list:  # pyright: ignore[reportMissingParameterType]
    return list(
        lint(
            bundle.descriptors,
            pipes=bundle.pipes,
            vectors=bundle.vectors,
            resources=bundle.resources,
            platforms=bundle.platforms,
            principals=bundle.principals,
            gates=bundle.gates,
        )
    )


def _cmd_fetch(args: argparse.Namespace) -> int:
    path = model_mod.fetch(args.repo, args.file)
    print(f"model at {path}")
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    bundle = load_case(Path(args.case))
    findings = _lint(bundle)
    for finding in findings:
        print(finding.render())
    errors = sum(1 for f in findings if f.severity is Severity.ERROR)
    print(
        f"{len(bundle.descriptors)} descriptors, {len(bundle.vectors)} vectors: "
        f"{errors} error(s), {len(findings) - errors} warning(s)"
    )
    return 1 if errors else 0


def _blocked_on(events: Sequence[Event], pipe: PipeName) -> bool:
    """Whether the last thing to happen on ``pipe``, as the journal tells it, was a job parking."""
    for event in reversed(events):
        if isinstance(event, JobBlocked) and event.pipe == pipe:
            return True
        if isinstance(event, JobWoken) and event.pipe == pipe:
            return False
    return False


class _Stop(Exception):
    """SIGINT or SIGTERM, turned into an exception the run loop can unwind through."""


def _install_signal_handlers() -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        raise _Stop(signal.Signals(signum).name)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _cmd_run(args: argparse.Namespace) -> int:
    bundle = load_case(Path(args.case))
    blocking = [f for f in _lint(bundle) if f.severity is Severity.ERROR]
    if blocking and not args.force:
        for finding in blocking:
            print(finding.render(), file=sys.stderr)
        print("refusing to run a tree that does not lint (--force to override)", file=sys.stderr)
        return 1

    model: LlamaModel | None = None
    if args.machine == "qwen":
        path = Path(args.model) if args.model else model_mod.default_path()
        if not path.is_file():
            print(f"no model at {path}; run 'zeos-count fetch-model'", file=sys.stderr)
            return 2

    started = time.time()
    if args.machine == "qwen":
        model = LlamaModel(path, n_gpu_layers=args.n_gpu_layers)
    events: list[Event] = []

    # A time only lands where it was meant to in a tape; a said number lands anywhere.
    said_at, typed = args.interrupt if args.interrupt else (None, None)
    armed = False

    # -- live output ---------------------------------------------------------
    # These two callbacks show what a job said, and nothing the kernel decided.
    names: dict[JobId, str] = {}
    kernel_box: list[object] = []

    def name_of(job: JobId) -> str:
        if job not in names:
            k = kernel_box[0]
            found = next((j for j in k.sched.jobs() if j.job_id == job), None)  # pyright: ignore[reportAttributeAccessIssue]
            names[job] = str(found.name) if found is not None else f"job {job}"
        return names[job]

    def resolve(job: JobId, alias: str | None) -> str:
        if alias is None:
            return "?"
        descriptor = bundle.descriptors.get(DescriptorName(name_of(job)))
        if descriptor is None:
            return alias
        return str(descriptor.pipes.resolve(alias) or alias)

    def on_command(job: JobId, line: str, request: MachineRequest) -> None:
        nonlocal armed
        if said_at is not None and not armed:
            head, _, rest = line.partition(" ")
            armed = head == "say" and rest.strip().isdigit() and int(rest.strip()) >= said_at
        who = f"{name_of(job):<10}"
        if request.op is OpKind.WRITE:
            print(
                f"{who} \u2500\u2500\u25b6 {resolve(job, request.pipe):<12} {render(request.payload)}",
                flush=True,
            )
        elif request.op is OpKind.READ:
            # Silent on purpose: whether a read waits is the kernel's call, and it has not
            # made it yet.
            pass
        elif request.op is OpKind.EXIT:
            print(f"{who} exit", flush=True)
        elif not args.quiet:
            print(f"{who} {line}", flush=True)

    def on_arrival(job: JobId, text: str) -> None:
        print(f"{name_of(job):<10} \u25c0\u2500\u2500 {text}", flush=True)

    def drain(seen: int) -> int:
        """Print the facts only the kernel knows, in the order the journal recorded them."""
        for event in events[seen:]:
            if isinstance(event, JobBlocked):
                print(f"{name_of(event.job):<10} ... waiting on {event.pipe}", flush=True)
        return len(events)

    if args.machine in ("scripted", "claude", "claude-code"):
        descriptors, valued = seat_maps(bundle)
        chosen = {"model": args.seat_model} if args.seat_model else {}
        source: CommandSource
        if args.machine == "scripted":
            from zeos_coop_count.scripted import TapeSource

            source = TapeSource(bundle.scripts)
        elif args.machine == "claude":
            from zeos_coop_count.claude import ClaudeSource

            source = ClaudeSource(descriptors=descriptors, valued=valued, **chosen)
        else:
            from zeos_coop_count.claude_code import ClaudeCodeSource

            source = ClaudeCodeSource(descriptors=descriptors, valued=valued, **chosen)
        seat = CommandSeat(
            source=source,
            block_size=args.block_size,
            on_command=on_command,
            on_arrival=on_arrival,
        )
        kernel, transport, machine = build_kernel(
            bundle,
            machine=seat,
            journal_sink=events,
            config=KernelConfig(seed=args.seed, case=bundle.name, max_ticks=args.max_ticks),
        )
    else:
        kernel, transport, machine = build_kernel(
            bundle,
            model,
            journal_sink=events,
            config=KernelConfig(seed=args.seed, case=bundle.name, max_ticks=args.max_ticks),
            block_size=args.block_size,
            n_ctx=args.n_ctx,
            n_threads=args.threads,
            on_command=on_command,
            on_arrival=on_arrival,
        )
    kernel_box.append(kernel)

    journal = Journal(Path(args.journal) if args.journal else None)
    driver = Driver(kernel, transport=transport, journal=journal)
    driver.boot(bundle.boot)
    schedule = load_schedule(Path(args.events)) if args.events else ()

    _install_signal_handlers()
    ticks = 0
    now_ns = 0
    reason = "quiescent"
    pending = sorted(schedule, key=lambda e: e.at_ns)
    prompted = False
    pressed = False
    rendered = 0
    interrupt_pipe, number_pipe = PipeName("keys.interrupt"), PipeName("keys.number")
    wired = bundle.pipes and any(p.name == interrupt_pipe for p in bundle.pipes)

    with Console.attached() as console:
        if console is not None and wired:
            print("space interrupts and asks for a number; q quits.\n", flush=True)
        else:
            print(f"{bundle.name}: {len(bundle.boot)} jobs booted; ctrl-c to stop\n", flush=True)

        # ``Driver._run_until`` unrolled, because this run never ends on its own: it has to
        # flush the journal as it goes, deliver events at their times, and keep turning while
        # a job waits for a human.
        try:
            while ticks < args.max_ticks:
                while pending and pending[0].at_ns <= now_ns:
                    event = pending.pop(0)
                    kernel.deliver(event.pipe, event.text)

                # Between ticks: `on_command` runs mid-decode and the kernel is not re-entrant.
                if armed and typed is not None:
                    if not pressed:
                        kernel.deliver(interrupt_pipe, "attention")
                        pressed = True
                    else:
                        # Sooner than a person could type, so the handler never parks on it.
                        kernel.deliver(number_pipe, str(typed))
                        typed = None

                if console is not None:
                    key, number = console.poll()
                    if key == "quit":
                        raise _Stop("q")
                    if key == "interrupt" and not console.collecting:
                        if _blocked_on(events, number_pipe):
                            # A handler is already parked, and the vector would queue a second
                            # fire to ask for a number nobody wanted to give.
                            print("\n(already waiting for a number)", flush=True)
                            console.begin_number()
                        else:
                            kernel.deliver(interrupt_pipe, "attention")
                            console.begin_number()
                    if number is not None:
                        kernel.deliver(number_pipe, number)
                        prompted = False

                    # While the handler is parked waiting for a number, the driver stops
                    # advancing the clock so the prompt does not scroll away under the output.
                    if console.collecting and _blocked_on(events, number_pipe):
                        if not prompted:
                            print("\ninterrupt: count from? ", end="", flush=True)
                            prompted = True
                        time.sleep(0.02)
                        continue

                kernel.advance_time(now_ns)
                ran = kernel.tick()
                rendered = drain(rendered)
                now_ns += Driver.DEFAULT_NS_PER_TICK
                if ran:
                    ticks += 1
                    if ticks % 200 == 0:
                        journal.extend(events[len(journal) :])
                elif pending or (console is not None and wired):
                    # Nothing runnable, but a scheduled event or a person is still expected.
                    time.sleep(0.02)
                else:
                    break
        except _Stop as stop:
            reason = str(stop).lower()
        except KeyboardInterrupt:
            reason = "interrupted"
        else:
            if ticks >= args.max_ticks:
                reason = f"reached --max-ticks {args.max_ticks}"

    journal.extend(events[len(journal) :])
    journal.close()

    print(
        f"\n{bundle.name}: stopped ({reason}) after {ticks} ticks, "
        f"{len(journal)} journal events, {time.time() - started:.1f}s"
    )
    for job in sorted(names):
        try:
            print(
                f"  {name_of(job):<10} {len(machine.lines(job))} commands, "
                f"{machine.stats(job).resident_tokens} resident tokens"
            )
        except KeyError:
            pass
    if args.journal:
        print(f"journal written to {args.journal}")

    machine.close()  # pyright: ignore[reportAttributeAccessIssue]
    if model is not None:
        model.free()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeos-count", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-model", help="download the GGUF weights")
    p_fetch.add_argument("--repo", default=model_mod.DEFAULT_REPO)
    p_fetch.add_argument("--file", default=model_mod.DEFAULT_FILE)
    p_fetch.set_defaults(func=_cmd_fetch)

    p_lint = sub.add_parser("lint", help="typecheck a case without running it")
    p_lint.add_argument("case")
    p_lint.set_defaults(func=_cmd_lint)

    p_run = sub.add_parser("run", help="run a case under the llama backend")
    p_run.add_argument("case")
    p_run.add_argument(
        "--machine",
        choices=("qwen", "scripted", "claude", "claude-code"),
        default="qwen",
        help="which seat answers each decode: local llama.cpp weights, the case's own "
        "tape (needs nothing, and replays byte-identically), the Claude API (needs "
        "ANTHROPIC_API_KEY), or a local 'claude --print' process (needs a claude login "
        "instead)",
    )
    p_run.add_argument("--model", default=None, help="path to a .gguf")
    p_run.add_argument(
        "--seat-model",
        default=None,
        help="which model answers in the claude and claude-code seats, overriding each "
        "seat's own default",
    )
    p_run.add_argument("--journal", default=None)
    p_run.add_argument("--events", default=None, help="JSONL schedule of external events")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--block-size", type=int, default=16)
    p_run.add_argument("--n-ctx", type=int, default=32768)
    p_run.add_argument("--n-gpu-layers", type=int, default=0)
    p_run.add_argument(
        "--threads",
        type=int,
        default=model_mod.DEFAULT_THREADS,
        help="thread count; changes sampling, so pin it when comparing runs",
    )
    p_run.add_argument(
        "--max-ticks",
        type=int,
        default=10_000_000,
        help="stop after this many token boundaries; the default is effectively never",
    )
    p_run.add_argument(
        "--interrupt",
        nargs=2,
        type=int,
        metavar=("SAID", "NUMBER"),
        default=None,
        help="press the key with nobody at the console: once a job has said SAID, "
        "deliver the interrupt, and type NUMBER one boundary later",
    )
    p_run.add_argument("--quiet", action="store_true", help="show only pipe traffic")
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DescriptorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
