# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Binding a solution to a problem, running it, and scoring it.

The three steps are deliberately separate:

1. **bind** -- check the solution against the problem's contract and assemble a
   kernel. Fails loudly and *before* the scenario, so a solution that cannot work
   says why rather than scoring zero mysteriously.
2. **run** -- drive the scenario. The problem owns the event schedule and the
   simulated cost of a token boundary; the solution owns nothing about time, so it
   cannot buy its own latency.
3. **score** -- evaluate criteria against the journal.

This is the reusable interface. Adding a problem means writing YAML and a scenario;
adding a solution means writing descriptors; neither requires touching this module,
and the same solution can be scored against a harder scenario without modification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from zeos.core.capabilities import Capability, Schema
from zeos.core.events import Event, JobSpawned
from zeos.core.ids import DescriptorName, JobId
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.demo.criteria import EvalContext, Verdict, evaluate_all, parse_criteria
from zeos.demo.problem import Problem, ProblemError
from zeos.demo.solution import Solution
from zeos.descriptor.lint import Finding, LintOptions, Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.driver import Driver, ScheduledEvent, load_schedule
from zeos.journal.writer import Journal
from zeos.machine.scripted import ScriptedMachine
from zeos.world.store import WorldStore

__all__ = ["Binding", "DemoReport", "bind", "run_demo", "discover"]


@dataclass(frozen=True, slots=True)
class Binding:
    """A problem and a solution, checked against each other and ready to run."""

    problem: Problem
    solution: Solution
    kernel: Kernel
    events: list[Event]
    journal: Journal
    driver: Driver
    findings: tuple[Finding, ...] = ()

    @property
    def blocked(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)


@dataclass(frozen=True, slots=True)
class DemoReport:
    problem: str
    solution: str
    verdicts: tuple[Verdict, ...] = ()
    findings: tuple[Finding, ...] = ()
    ticks: int = 0
    journal_events: int = 0
    ran: bool = True
    world: Mapping[str, str] = field(default_factory=dict[str, str])

    @property
    def passed(self) -> bool:
        return self.ran and bool(self.verdicts) and all(v.passed for v in self.verdicts)

    @property
    def score(self) -> tuple[int, int]:
        return sum(1 for v in self.verdicts if v.passed), len(self.verdicts)

    def render(self) -> str:
        lines = [f"{self.problem} / {self.solution}"]
        for finding in self.findings:
            lines.append(f"  {finding.render()}")
        if not self.ran:
            lines.append("  NOT RUN -- the solution does not satisfy the contract")
            return "\n".join(lines)
        for verdict in self.verdicts:
            lines.append(verdict.render())
        passed, total = self.score
        lines.append(
            f"  {passed}/{total} criteria met "
            f"({self.ticks} ticks, {self.journal_events} journal events)"
        )
        return "\n".join(lines)


def bind(
    problem: Problem,
    solution: Solution,
    *,
    journal_path: Path | None = None,
) -> Binding:
    """Check conformance and assemble a kernel. Does not run anything."""
    findings = list(problem.validate(solution))
    # The ordinary descriptor lint also applies: a solution can satisfy the contract
    # and still be internally incoherent.
    findings.extend(
        lint(
            solution.descriptors,
            pipes=problem.contract.pipe_specs(),
            vectors=solution.vectors,
            resources=problem.contract.resources,
            options=LintOptions(),
        )
    )

    events: list[Event] = []
    pipes = PipeTable(problem.contract.pipe_specs())
    world = WorldStore()
    kernel = Kernel(
        descriptors=_with_contract_schemas(problem, solution),
        machine=ScriptedMachine(solution.scripts, block_size=problem.block_size),
        pipes=pipes,
        vectors=VectorTable(solution.vectors),
        world=world,
        resources=ResourceTable(problem.contract.resources),
        journal_sink=events,
        config=KernelConfig(case=f"{problem.name}/{solution.name}"),
    )
    for obj, value in sorted(problem.contract.initial_world().items()):
        world.set(obj, value, at=kernel.clock)

    journal = Journal(journal_path)
    driver = Driver(kernel, journal=journal, ns_per_tick=problem.ns_per_tick)
    return Binding(
        problem=problem,
        solution=solution,
        kernel=kernel,
        events=events,
        journal=journal,
        driver=driver,
        findings=tuple(findings),
    )


def run_demo(
    problem: Problem,
    solution: Solution,
    *,
    journal_path: Path | None = None,
    schedule: Sequence[ScheduledEvent] | None = None,
) -> DemoReport:
    """Bind, run, and score."""
    binding = bind(problem, solution, journal_path=journal_path)
    if binding.blocked:
        binding.journal.close()
        return DemoReport(
            problem=problem.name,
            solution=solution.name,
            findings=binding.findings,
            ran=False,
        )

    events = (
        list(schedule)
        if schedule is not None
        else (
            list(load_schedule(problem.scenario_path)) if problem.scenario_path is not None else []
        )
    )

    binding.driver.boot(solution.boot)
    ticks = binding.driver.run(events)
    binding.journal.close()

    ctx = EvalContext(
        events=binding.events,
        world=binding.kernel.world,
        job_names=_job_names(binding.events),
    )
    verdicts = evaluate_all(parse_criteria(problem.criteria), ctx)
    return DemoReport(
        problem=problem.name,
        solution=solution.name,
        verdicts=verdicts,
        findings=binding.findings,
        ticks=ticks,
        journal_events=len(binding.journal),
        world={str(k): v for k, v in binding.kernel.world.snapshot().items()},
    )


def _with_contract_schemas(
    problem: Problem, solution: Solution
) -> Mapping[DescriptorName, Descriptor]:
    """Attach the problem's accepted-value schemas to the solution's capabilities.

    Which values an actuator accepts is a fact about the *plant*, not a claim a
    behaviour gets to make, so it is declared once in the contract and stamped onto
    every capability that targets that pipe. A solution therefore cannot widen its
    own effect channel -- it could not even express doing so -- and every actuator
    write is schema-checked without any descriptor author having to remember to ask
    for it.
    """
    result: dict[DescriptorName, Descriptor] = {}
    for name, descriptor in solution.descriptors.items():
        rewritten: list[Capability] = []
        changed = False
        for capability in descriptor.capabilities:
            contract = problem.contract.pipe(capability.pipe)
            if contract is not None and contract.accepts and capability.schema is None:
                rewritten.append(
                    replace(
                        capability,
                        schema=Schema.of_values(f"{contract.name}-values", contract.accepts),
                    )
                )
                changed = True
            else:
                rewritten.append(capability)
        result[name] = replace(descriptor, capabilities=tuple(rewritten)) if changed else descriptor
    return result


def _job_names(events: Sequence[Event]) -> Mapping[JobId, DescriptorName]:
    """Criteria are written against behaviour names; job ids are an allocation
    detail and must not leak into a problem definition."""
    return {e.job: e.descriptor for e in events if isinstance(e, JobSpawned)}


def discover(root: Path) -> tuple[tuple[Problem, tuple[Solution, ...]], ...]:
    """Find every demonstration under ``root``.

    A demonstration is **one self-contained directory**::

        <root>/<demo>/problem/              problem.yaml, criteria.yaml, scenario.jsonl
        <root>/<demo>/solutions/<name>/     descriptor trees attempting it

    Everything a demonstration is -- its problem, the solutions offered for it, and
    whatever research, scenario notes or tooling it carries -- lives together. An
    earlier layout split ``problems/<name>/`` from ``solutions/<name>/`` at the
    repository root and paired them by directory name, which meant that reading one
    demonstration meant reading two trees and that renaming it meant renaming both.

    **This does not weaken the problem/solution separation**, which was never about
    distance on disk. A solution may not ship ``pipes.yaml`` or ``world-state.yaml``
    and the loader refuses it if it does -- a solution that could redefine its own
    environment could pass by changing the question. That check is in code, and it is
    unaffected by which directory the files sit in.
    """
    found: list[tuple[Problem, tuple[Solution, ...]]] = []
    if not root.is_dir():
        return ()
    for demo_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        problem_dir = demo_dir / "problem"
        if not (problem_dir / "problem.yaml").is_file():
            continue
        problem = Problem.load(problem_dir)
        if problem.name != demo_dir.name:
            # The directory is what you type; the problem's own name is what the
            # criteria and journals record. Letting them drift means `zeos demo run
            # <what the listing showed>` fails, so say so rather than resolving it
            # silently in one direction.
            raise ProblemError(
                f"{demo_dir}: directory is {demo_dir.name!r} but problem.yaml "
                f"declares name {problem.name!r}; they must match"
            )
        solutions: list[Solution] = []
        solutions_dir = demo_dir / "solutions"
        if solutions_dir.is_dir():
            for solution_dir in sorted(s for s in solutions_dir.iterdir() if s.is_dir()):
                solutions.append(Solution.load(solution_dir))
        found.append((problem, tuple(solutions)))
    return tuple(found)


def demo_paths(root: Path, name: str) -> tuple[Path, Path]:
    """``(problem dir, solutions dir)`` for one demonstration, by directory name."""
    return (root / name / "problem", root / name / "solutions")
