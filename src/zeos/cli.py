# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``zeos`` -- lint, run, replay, and inspect.

Four verbs, matching the four things you do with a descriptor tree:

``lint``     typecheck the tree without running it -- the compiler
``run``      execute it against a schedule of external events
``replay``   re-read a journal and check it reproduces
``inspect``  summarise a journal: what preempted what, what faulted, what resumed
``debug``    draw the tree, and step a journal of it -- the wiring, and the run

``replay --assert-identical`` is the determinism gate. It compares **bytes**, not
parsed structures: a comparison tolerant of key reordering would miss exactly the
bug class the gate exists for, which is nondeterministic iteration order leaking
into kernel decisions.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from collections import Counter
from pathlib import Path

from zeos.core.events import (
    Event,
    FaultRaised,
    JobPreempted,
    JobResumed,
    VectorFired,
)
from zeos.core.kernel import KernelConfig
from zeos.descriptor.lint import LintOptions, Severity, lint
from zeos.descriptor.loader import load_case
from zeos.descriptor.schema import DescriptorError
from zeos.driver import Driver, build_kernel, load_schedule
from zeos.journal.writer import Journal, read_journal

__all__ = ["main"]


def _cmd_lint(args: argparse.Namespace) -> int:
    bundle = load_case(Path(args.case))
    findings = lint(
        bundle.descriptors,
        pipes=bundle.pipes,
        vectors=bundle.vectors,
        resources=bundle.resources,
        platforms=bundle.platforms,
        principals=bundle.principals,
        gates=bundle.gates,
        options=LintOptions(link_rtt_p99_ns=args.link_rtt_ns),
    )
    for finding in findings:
        print(finding.render())

    errors = sum(1 for f in findings if f.severity is Severity.ERROR)
    warnings = len(findings) - errors
    print(
        f"{len(bundle.descriptors)} descriptors, {len(bundle.vectors)} vectors: "
        f"{errors} error(s), {warnings} warning(s)"
    )
    return 1 if errors else 0


def _cmd_run(args: argparse.Namespace) -> int:
    bundle = load_case(Path(args.case))
    findings = lint(
        bundle.descriptors,
        pipes=bundle.pipes,
        vectors=bundle.vectors,
        resources=bundle.resources,
        platforms=bundle.platforms,
        principals=bundle.principals,
        gates=bundle.gates,
    )
    blocking = [f for f in findings if f.severity is Severity.ERROR]
    if blocking and not args.force:
        for finding in blocking:
            print(finding.render(), file=sys.stderr)
        print("refusing to run a tree that does not lint (--force to override)", file=sys.stderr)
        return 1

    events: list[Event] = []
    kernel, transport = build_kernel(
        bundle,
        journal_sink=events,
        config=KernelConfig(seed=args.seed, case=bundle.name),
        block_size=args.block_size,
    )
    journal = Journal(Path(args.journal) if args.journal else None)
    driver = Driver(kernel, transport=transport, journal=journal)
    driver.boot(bundle.boot)
    schedule = load_schedule(Path(args.events)) if args.events else ()
    ticks = driver.run(schedule)
    journal.close()

    print(f"{bundle.name}: {ticks} ticks, {len(journal)} journal events")
    if args.journal:
        print(f"journal written to {args.journal}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.journal)
    records = read_journal(path)
    rebuilt = Journal()
    rebuilt.extend(r.event for r in records)

    if [r.seq for r in records] != list(range(len(records))):
        print("journal sequence numbers are not contiguous", file=sys.stderr)
        return 1

    if args.assert_identical:
        original = path.read_bytes()
        if rebuilt.to_bytes() != original:
            print("replay is NOT byte-identical to the original journal", file=sys.stderr)
            return 1
        print(f"replay is byte-identical ({len(records)} events)")
    else:
        print(f"{len(records)} events replayed cleanly")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    records = read_journal(Path(args.journal))
    counts = Counter(type(r.event).__name__ for r in records)

    print(f"{len(records)} events")
    for name, count in sorted(counts.items()):
        print(f"  {count:>6}  {name}")

    preemptions = [r.event for r in records if isinstance(r.event, JobPreempted)]
    resumes = [r.event for r in records if isinstance(r.event, JobResumed)]
    faults = [r.event for r in records if isinstance(r.event, FaultRaised)]
    fired = [r.event for r in records if isinstance(r.event, VectorFired)]

    if fired:
        print("\ninterrupts:")
        for event in fired:
            print(f"  {event.vector} -> {event.handler} @ priority {event.priority}")
    if preemptions:
        print("\npreemptions:")
        for event in preemptions:
            print(
                f"  job {event.job} preempted by job {event.by_job} (priority {event.by_priority})"
            )
    if resumes:
        print("\nresumes:")
        for event in resumes:
            changed = ", ".join(f"{d.obj}: {d.before} -> {d.after}" for d in event.dirty)
            print(
                f"  job {event.job}: {event.resume_kind.value}"
                + (f" [{changed}]" if changed else "")
            )
    if faults:
        print("\nfaults:")
        for event in faults:
            print(f"  job {event.job}: {event.fault.value}: {event.detail}")
    return 0


def _cmd_debug(args: argparse.Namespace) -> int:
    from zeos.debugger.payload import build_payload
    from zeos.debugger.server import export, serve

    case = Path(args.case)

    def payload() -> object:
        bundle = load_case(case)
        findings = lint(
            bundle.descriptors,
            pipes=bundle.pipes,
            vectors=bundle.vectors,
            resources=bundle.resources,
            platforms=bundle.platforms,
            principals=bundle.principals,
            gates=bundle.gates,
        )
        records = read_journal(Path(args.journal)) if args.journal else None
        return build_payload(bundle, records=records, findings=findings, every=args.every)

    if args.out:
        out = export(payload(), Path(args.out))
        size = out.stat().st_size
        print(f"wrote {out} ({size / 1e6:.1f} MB)")
        if size > 5_000_000:
            # A page too big to open is a page nobody reads. Decimating the fold
            # costs only the ability to stop between two adjacent events.
            print(
                "that is large for a single page; re-run with --every N to keep one frame in N",
                file=sys.stderr,
            )
        return 0

    server = serve(payload, port=args.port)
    where = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"serving {case} at {where}\nctrl-c to stop")
    if not args.no_open:
        webbrowser.open(where)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from zeos.demo.problem import Problem
    from zeos.demo.runner import demo_paths, discover, run_demo
    from zeos.demo.solution import Solution

    demos_root = Path(args.demos)

    if args.demo_command == "list":
        found = discover(demos_root)
        if not found:
            print(f"no demonstrations found under {demos_root}")
            return 0
        for problem, solutions in found:
            print(f"{problem.name} -- {problem.title}")
            for solution in solutions:
                print(f"    {solution.describe()}")
            if not solutions:
                print("    (no solutions yet)")
        return 0

    if args.demo_command == "show":
        problem_dir, _ = demo_paths(demos_root, args.problem)
        problem = Problem.load(problem_dir)
        print(f"{problem.name} -- {problem.title}\n")
        if problem.description:
            print(problem.description + "\n")
        print(problem.contract.render())
        print(f"\n{len(problem.criteria)} criteria:")
        for entry in problem.criteria:
            print(f"    {entry.get('id')}  ({entry.get('kind')})")
        return 0

    # run / score
    problem_dir, solutions_dir = demo_paths(demos_root, args.problem)
    problem = Problem.load(problem_dir)
    names: list[str] = (
        [args.solution]
        if args.solution
        else sorted(d.name for d in solutions_dir.iterdir() if d.is_dir())
    )
    worst = 0
    for name in names:
        solution = Solution.load(solutions_dir / name)
        journal = Path(args.journal) if args.journal and len(names) == 1 else None
        report = run_demo(problem, solution, journal_path=journal)
        print(report.render())
        print()
        if not report.passed:
            worst = 1
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeos", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_lint = sub.add_parser("lint", help="typecheck a descriptor tree without running it")
    p_lint.add_argument("case", help="path to a case directory")
    p_lint.add_argument(
        "--link-rtt-ns",
        type=int,
        default=None,
        help="measured p99 link RTT; enables the placement check",
    )
    p_lint.set_defaults(func=_cmd_lint)

    p_run = sub.add_parser("run", help="run a descriptor tree")
    p_run.add_argument("case")
    p_run.add_argument("--events", default=None, help="JSONL schedule of external events")
    p_run.add_argument("--journal", default=None, help="where to write the journal")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--block-size", type=int, default=16)
    p_run.add_argument("--force", action="store_true", help="run despite lint errors")
    p_run.set_defaults(func=_cmd_run)

    p_replay = sub.add_parser("replay", help="re-read a journal and check it reproduces")
    p_replay.add_argument("journal")
    p_replay.add_argument("--assert-identical", action="store_true")
    p_replay.set_defaults(func=_cmd_replay)

    p_inspect = sub.add_parser("inspect", help="summarise a journal")
    p_inspect.add_argument("journal")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_debug = sub.add_parser("debug", help="draw a case, and step through a journal of it")
    p_debug.add_argument("case", help="path to a case directory")
    p_debug.add_argument("--journal", default=None, help="a journal to step through")
    p_debug.add_argument("-o", "--out", default=None, help="write one self-contained page")
    p_debug.add_argument("--port", type=int, default=8000)
    p_debug.add_argument(
        "--every",
        type=int,
        default=1,
        help="keep one frame in N; for a run too long to hold every frame",
    )
    p_debug.add_argument("--no-open", action="store_true", help="do not open a browser")
    p_debug.set_defaults(func=_cmd_debug)

    p_demo = sub.add_parser("demo", help="run demonstration problems against solutions")
    p_demo.add_argument(
        "--demos",
        default="demo",
        help="root holding one directory per demonstration",
    )
    demo_sub = p_demo.add_subparsers(dest="demo_command", required=True)

    demo_sub.add_parser("list", help="list problems and the solutions offered for each")

    d_show = demo_sub.add_parser("show", help="print a problem's contract and criteria")
    d_show.add_argument("problem")

    d_run = demo_sub.add_parser("run", help="run a solution against a problem and score it")
    d_run.add_argument("problem")
    d_run.add_argument("solution", nargs="?", default=None, help="omit to run every solution")
    d_run.add_argument("--journal", default=None)

    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DescriptorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
