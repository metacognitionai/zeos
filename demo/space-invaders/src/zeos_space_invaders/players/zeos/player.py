# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Play the game as a ZEOS descriptor tree instead of a prompt-response loop.

A world tick is faster than a forward pass, so the player is structurally late;
the fix is to stop the slow deliberation being the only thing that can act.
`pilot` (priority 60) reads a board and decides, served by the API machine as a
real streamed completion; `evade` (priority 5, pinned, unpreemptible) is bound
to `game.threats` through the vector table and writes a dodge as a native
behaviour, because a forward pass would arrive after the hit. The kernel decodes
and charges both the same way, which is what makes the comparison mean anything.

`ZeosDriver` is the device adapter -- sensor readings into pipes, pump the kernel,
apply the actuator, publish the world -- and the only part that touches a clock
or does I/O.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from zeos.core.ids import JobId, ObjectName, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.loader import CaseBundle, load_case
from zeos.journal.codec import encode_record
from zeos.machine.base import DecodeResult, MachineRequest, OpKind, tokens_from_text
from zeos.world.store import WorldStore

from ...game import ACTIONS, Controls, Rules, snapshot
from ...runlog import Decision
from ...utils.views import LeadView
from ..base import rules_prompt
from .api_machine import APIMachineBase, Native

#: Resolved from `__file__` so `agent` runs from any directory and out of a wheel.
CASE_ROOT = Path(__file__).resolve().parents[2] / "cases" / "space-invaders"
GAME_STATE = PipeName("game.state")
GAME_THREATS = PipeName("game.threats")
GAME_CONTROLS = PipeName("game.controls")

#: Served by local code rather than by the API; `handlers/evade.md` says so in prose.
REFLEX = "evade"

#: What the kernel injects into a resumed job whose inputs changed while it was
#: suspended (`zeos.core.kernel.render_resume_notice`); tests assert on it.
RESUME = "<RESUME>"
RESUME_END = "</RESUME>"

#: A line break as a word the pipe will carry, because `tokens_from_text` splits
#: on whitespace; `goals/pilot.md` tells the model what the word means.
NEWLINE = "<nl>"

#: The machine's, not the case's: `ns_per_tick` makes `deadline: 5ms` in
#: `vectors.yaml` mean five token boundaries, reproducible on any hardware.
BLOCK_SIZE = 16
NS_PER_TICK = 1_000_000

REFLEX_HORIZON = 2  # turns of warning that count as "about to land"


def encode(text: str) -> str:
    """A board, as words a pipe will carry. See `NEWLINE`."""
    return f" {NEWLINE} ".join(text.split("\n"))


def decode(text: str) -> str:
    """The inverse, for a reader that wants the board back as a board."""
    return "\n".join(part.strip() for part in text.split(NEWLINE))


# --- the reflex --------------------------------------------------------------


def dodge(threat: str) -> str:
    """Which way to go, from a threat reading that names the clear sides."""
    clear = threat.rsplit("clear:", 1)[-1] if "clear:" in threat else ""
    if "stay" in clear:
        # There is no wait in this game, so standing still is written as `shoot`.
        return "shoot"
    if "left" in clear and "right" in clear:
        column = int(threat.split("in col ", 1)[1].split(";", 1)[0])
        return "left" if column % 2 == 0 else "right"
    if "left" in clear:
        return "left"
    if "right" in clear:
        return "right"
    return "left"


def evade_behaviour(native: Native) -> DecodeResult:
    """The reflex, as one decode step and then an exit.

    The threat arrives with the dispatch -- a vector's payload is consumed when it
    fires and injected once the body is in place -- so reading stdin would block on
    a drained pipe. The first step's `arrived` is body and reading together, hence
    the test for `clear:`; with no reading the handler exits rather than guess.
    """
    if native.step == 0 and "clear:" in native.arrived:
        return DecodeResult(
            tokens=tokens_from_text("dodging"),
            request=MachineRequest(
                op=OpKind.WRITE,
                pipe=GAME_CONTROLS,
                payload=tokens_from_text(dodge(native.arrived)),
            ),
        )
    return DecodeResult(tokens=(), request=MachineRequest(op=OpKind.EXIT))


# --- the sensors -------------------------------------------------------------


def threat_reading(info: dict) -> str | None:
    """The threat sensor, which fires when fire is about to land on or beside us.

    Beside us as well, because the pilot's move lands ticks after the board it was
    chosen against, so a bomb landing one column over threatens a move in flight;
    the reading then says the own column is clear so the handler can stand still.
    Fall speed and width come off the snapshot's rules, not module defaults.
    """
    rules = info["rules"]
    player = info["player"]
    landing = [
        rules.turns_to_land(row) for row, col in info["dangers"] if col == player
    ]
    imminent = [turns for turns in landing if 0 < turns <= REFLEX_HORIZON]
    beside = sorted(
        {
            col
            for row, col in info["dangers"]
            if abs(col - player) == 1 and rules.turns_to_land(row) == 1
        }
    )
    if not imminent and not beside:
        return None
    occupied = {
        col
        for row, col in info["dangers"]
        if rules.turns_to_land(row) <= REFLEX_HORIZON + 1
    }
    sides = []
    if not imminent:
        sides.append("stay")
    if player - 1 >= 0 and player - 1 not in occupied:
        sides.append("left")
    if player + 1 < rules.w and player + 1 not in occupied:
        sides.append("right")
    where = (
        f"fire lands in {min(imminent)} turns in col {player}"
        if imminent
        else f"fire lands in 1 turn in col {' and '.join(map(str, beside))}, beside you"
    )
    return f"{where}; clear: {' '.join(sides) if sides else 'neither'}"


# --- assembling the machine and the kernel -----------------------------------


def build_machine(backend: str = "openai", **kwargs: object) -> APIMachineBase:
    """The machine the kernel will decode through, with the reflex registered."""
    # The machine narrows its schema to exactly the game's words, so a reply naming
    # anything else is unrepresentable rather than badly behaved.
    kwargs.setdefault("actions", ACTIONS)
    if backend == "openai":
        from .api_openai import OpenAIAPIMachine

        machine: APIMachineBase = OpenAIAPIMachine(block_size=BLOCK_SIZE, **kwargs)  # type: ignore[arg-type]
    elif backend == "claude":
        from .api_claude import ClaudeAPIMachine

        machine = ClaudeAPIMachine(block_size=BLOCK_SIZE, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown backend {backend!r}; expected openai or claude")
    machine.register_behaviour(REFLEX, evade_behaviour)
    return machine


def load_criteria(root: Path = CASE_ROOT) -> tuple[dict, ...]:
    """The case's success criteria, if it states any; not part of running."""
    import yaml

    path = root / "criteria.yaml"
    if not path.is_file():
        return ()
    raw = yaml.safe_load(path.read_text("utf-8")) or []
    return tuple(raw)


def build_kernel(
    machine: APIMachineBase | None = None,
    root: Path = CASE_ROOT,
    *,
    view: object | None = None,
    rules: Rules | None = None,
) -> tuple[Kernel, CaseBundle]:
    """Load the case off disk and assemble a kernel over it.

    The booted goals get the prompt loop's rules text appended, because the body
    on disk cannot know the board or view this run will play.
    """
    bundle = load_case(root)
    game = rules_prompt(view or LeadView(), rules or Rules())
    descriptors = {
        name: replace(desc, body=f"{desc.body}\n\n---\n\n{game}")
        if name in bundle.boot
        else desc
        for name, desc in bundle.descriptors.items()
    }
    bundle = replace(bundle, descriptors=descriptors)
    world = WorldStore()
    kernel = Kernel(
        descriptors=bundle.descriptors,
        machine=machine or build_machine(),
        pipes=PipeTable(bundle.pipes),
        vectors=VectorTable(bundle.vectors),
        world=world,
        resources=ResourceTable(bundle.resources),
        principals=bundle.principals,
        gates=bundle.gates,
        journal_sink=[],
        config=KernelConfig(case=bundle.name),
    )
    for obj, value in sorted(bundle.world.items()):
        world.set(ObjectName(obj), value, at=kernel.clock)
    return kernel, bundle


def case_path(root: Path = CASE_ROOT) -> str:
    """Where this run's case is, in the form you can paste into `zeos debug`.

    Repo-relative when the case is inside a checkout, because `meta.json` already
    records the commit and `zeos debug` runs from that root; out of a wheel there
    is no root, so the absolute path is the only true answer.
    """
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            return root.relative_to(parent).as_posix()
    return root.as_posix()


def kernel_version() -> str:
    """Which zeos wrote a run's journal.

    A VCS install records the commit in `direct_url.json`; a `path = "../.."`
    install records only a directory, so the commit is read out of the checkout.
    """
    from importlib.metadata import Distribution, PackageNotFoundError

    try:
        dist = Distribution.from_name("zeos")
    except PackageNotFoundError:  # pragma: no cover - zeos is a hard dependency
        return "unknown"
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    commit = info.get("vcs_info", {}).get("commit_id")
    if commit:
        return commit[:12]
    url = info.get("url", "")
    if url.startswith("file://"):
        import subprocess
        from urllib.parse import urlparse

        root = Path(urlparse(url).path)
        try:
            rev = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - not a checkout
            rev = ""
        if rev:
            # A working tree is not a commit: an editable install runs whatever is
            # on disk.
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            return f"{rev}-dirty" if dirty else rev
    return dist.version or "unknown"


# --- coupling the kernel to a running game -----------------------------------


@dataclass
class ZeosDriver:
    """Runs the kernel alongside a live game.

    The only part that touches a clock or does I/O, which is what keeps the kernel
    replayable; the order things are delivered in is load bearing.
    """

    kernel: Kernel = field(default=None)  # type: ignore[assignment]
    machine: APIMachineBase = field(default=None)  # type: ignore[assignment]
    #: Which served backend to build when none is handed in.
    backend: str = "openai"
    #: The game the pilot is told the rules of when the kernel is built here;
    #: `None` is the default board and the `lead` view.
    view: object | None = None
    rules: Rules | None = None
    max_ticks: int = 64
    started: bool = False
    #: Descriptors the solution boots, spawned by `start`; `boot.yaml` names them.
    boot: tuple[str, ...] = ()
    #: Virtual nanoseconds per token boundary. See `NS_PER_TICK`.
    ns_per_tick: int = NS_PER_TICK
    _now_ns: int = 0
    #: The case's success criteria, raw, judged after the run by `zeos.demo.criteria`.
    criteria: tuple = ()
    #: The runner's own stick, so a write reaches the ship the moment it lands and
    #: a job resumed in the same tick sees the dodge.
    controls: Controls | None = None
    #: Whether the last stick write actually moved the ship; the budget or an
    #: ended episode can refuse one.
    last_applied: bool = False
    _applied_at: int = 0
    _seen_events: int = 0
    reflexes: int = 0
    _tick: int = 0
    #: The tick the readable board was delivered on: a pilot move belongs to the
    #: board it was chosen against, not to the tick it lands on.
    _board_tick: int = 0
    _board_note: object = None
    _note: object = None
    #: `journal()`'s own cursor: `_seen_events` is reset per tick after `start()`
    #: and would swallow the events the kernel emits while coming up.
    _journalled: int = 0

    def __post_init__(self) -> None:
        if self.kernel is None:
            self.machine = self.machine or build_machine(self.backend)
            self.kernel, bundle = build_kernel(
                self.machine, view=self.view, rules=self.rules
            )
            self.boot = tuple(str(name) for name in bundle.boot)
            self.criteria = load_criteria()
        # A caller that assembles its own kernel is a shape the dataclass
        # advertises, and `close`, `usage` and the summary go through the machine.
        self.machine = self.machine or getattr(self.kernel, "machine", None)

    def close(self) -> None:
        if self.machine is not None:
            self.machine.close()

    def start(self) -> None:
        """Bring the kernel up and spawn what the solution boots.

        The pilot is booted rather than dispatched by a board vector: a vector
        consumes its source payload and injects it, so a vector-spawned pilot would
        block forever on a pipe the kernel had already drained.
        """
        self.kernel.start()
        for name in self.boot:
            self.kernel.spawn(name)
        self.started = True

    # -- pumping ------------------------------------------------------------

    def _pump(self, until_running: bool = False, deadline: float | None = None) -> int:
        """Run the kernel forward, bounded.

        `deadline` is when the world's next tick is due; `max_ticks` bounds token
        boundaries and is only right for the synchronous `step`. The batch also
        ends when a write lands, and `until_running` stops once some job is
        executing, because an interrupt handed to a drained kernel cannot preempt.
        """
        ticks = 0
        while True:
            # At least one boundary, whatever the bound says: a deadline that has
            # already passed would otherwise make a batch that runs nothing.
            if ticks and not (
                ticks < self.max_ticks
                if deadline is None
                else time.monotonic() < deadline
            ):
                break
            # The kernel never reads a clock; `advance_time` takes an absolute time.
            self.kernel.advance_time(self._now_ns)
            if not self.kernel.tick():
                break
            self._now_ns += self.ns_per_tick
            ticks += 1
            if self._apply_controls():
                break
            if until_running and self.kernel.sched.running is not None:
                break
        return ticks

    def _apply_controls(self) -> bool:
        """Put a stick write into the game the moment it lands, not at tick end.

        Returns whether one did, which ends a batch in `_pump`. The kernel resumes
        the pilot in the same breath the handler exits, so a dodge applied at tick
        end has not moved the ship when `dirty_for` runs.
        """
        if self.controls is None:
            return False
        author = self._author(since=self._applied_at)
        self._applied_at = len(self.kernel.events)
        if author is None:
            return False
        action = self.kernel.world.get(ObjectName("game.action"), "shoot")
        self.last_applied = self.controls.write(action)
        self._publish_world(snapshot(self.controls.game))
        return True

    def _publish_world(self, info: dict) -> None:
        """Publish the facts a suspended job will be told changed under it.

        Every tick, unconditionally, because `WorldStore.set` drops an idempotent write.
        """
        rows = [row for row, _ in info["monsters"].values()]
        for obj, value in (
            ("ship.column", str(info["player"])),
            ("block.front_row", str(max(rows)) if rows else "0"),
            ("missile.in_flight", "no" if info["can_shoot"] else "yes"),
        ):
            self.kernel.world.set(ObjectName(obj), value, at=self.kernel.clock)

    # -- one world tick ------------------------------------------------------

    def step(self, board: str, info: dict, note: object = None) -> Decision | None:
        """One world tick: sense, schedule, and read back the move, if any.

        None when no job wrote the stick this tick, which is most of them; that is
        not a decision and the caller must not play one.
        """
        if not self.started:
            self.start()
        self._seen_events = len(self.kernel.events)
        self._tick, self._note = info["ticks"], note

        # Before anything runs: `info` is the runner's snapshot from the top of the
        # tick and is stale the moment a job moves the ship.
        self._publish_world(info)

        # Pump before delivering, and only until something is running: an
        # interrupt delivered to a drained kernel is dispatched, never a preemption.
        ticks = self._pump(until_running=True)
        self.sense(board, info, note)
        ticks += self._pump()
        return self.collect(ticks)

    # -- the two halves the threaded runner drives separately -----------------

    def sense(self, board: str, info: dict, note: object = None) -> None:
        """Hand the kernel this tick's reading, the only device-side write.

        A threat and a board are never delivered on the same tick: the stick is
        last-write-wins, so a move chosen against a board the pilot never resolved
        could land after the dodge and undo it.
        """
        if not self.started:
            self.start()
        self._tick, self._note = info["ticks"], note
        self._publish_world(info)
        threat = threat_reading(info)
        if threat is not None:
            self.kernel.deliver(GAME_THREATS, threat)
            self.reflexes += 1
            return
        # Keep only the newest board: the pilot is slower than the world, and left
        # queued it would wake and read the oldest.
        stale = self.kernel.pipes.get(GAME_STATE)
        if stale.readable:
            stale.read()
        self.kernel.deliver(GAME_STATE, encode(board))
        # The stamp a pass gets belongs to the board it is about.
        self._board_tick, self._board_note = self._tick, self._note

    def collect(self, ticks: int) -> Decision | None:
        """The move a job made since `_seen_events`, if one did.

        Split out of `step` because the threaded runner pumps continuously and
        wants to know the moment a write lands, not at the end of a tick.
        """
        # A preempted job's reply was composed against a world that has moved on,
        # and a resume with an empty diff injects nothing to tell the machine so.
        self._invalidate_preempted()
        author = self._author()
        if author is None:
            return None
        action = self.kernel.world.get(ObjectName("game.action"), "shoot")
        return self._decision(author, action, ticks)

    def run_kernel(self, deadline: float | None = None) -> Decision | None:
        """One batch of boundaries, and the move it produced if any.

        The kernel is touched from the threaded runner's thread and no other; the
        clock thread hands its reading over through the pending slot.
        """
        if not self.started:
            self.start()
        self._seen_events = len(self.kernel.events)
        return self.collect(self._pump(deadline=deadline))

    def _decision(self, by: str, action: str, ticks: int) -> Decision:
        """This tick's record.

        `preempted` is read off the journal rather than kept as our own flag: the
        kernel saying it took the machine away is the claim.
        """
        return Decision(
            by=by,
            action=action,
            tick=self._board_tick if by == "pilot" else self._tick,
            # Request-to-syscall as the machine measured it, the same span the
            # unscheduled arm reports; the reflex reached no model and has none.
            latency=self.machine.last_roundtrip if by == "pilot" else None,
            preempted=self._preempted_since(self._seen_events),
            kernel_ticks=ticks,
        )

    def _invalidate_preempted(self) -> None:
        """Drop the completion of every job the kernel preempted this tick.

        Only the driver can: it reads the journal, and neither the kernel nor the
        machine tells the other about a preemption.
        """
        if self.machine is None:
            return
        for event in self.kernel.events[self._seen_events :]:
            if getattr(type(event), "KIND", "") == "job.preempted":
                self.machine.invalidate(event.job)

    def _preempted_since(self, start: int) -> bool:
        return any(
            getattr(type(event), "KIND", "") == "job.preempted"
            for event in self.kernel.events[start:]
        )

    def _author(self, since: int | None = None) -> str | None:
        """Which job moved the stick this tick, or None if none did.

        Read off the kernel's own log, because the reflex and the pilot can choose
        the same action; None is a real answer, and replaying the world value would
        emit a move nobody chose.
        """
        start = self._seen_events if since is None else since
        wrote: JobId | None = None
        for event in self.kernel.events[start:]:
            if type(event).__name__ == "PipeWritten" and str(
                getattr(event, "pipe", "")
            ) == str(GAME_CONTROLS):
                job = getattr(event, "job", None)
                if job is not None:
                    wrote = job
        if wrote is None:
            return None
        # Named only once a write is found; almost no token boundary carries one.
        names = {
            job.job_id: str(job.descriptor.name) for job in self.kernel.sched.jobs()
        }
        return names.get(wrote, "pilot")

    # -- what the run leaves behind ------------------------------------------

    @property
    def preemptions(self) -> int:
        """Preemptions the kernel journalled, the only evidence there is."""
        return sum(
            1
            for event in self.kernel.events
            if getattr(type(event), "KIND", "") == "job.preempted"
        )

    @property
    def usage(self) -> dict:
        """What the served model reported, summed over the run."""
        return dict(getattr(self.machine, "usage", {}) or {})

    def verdicts(self) -> list[dict]:
        """Judge the run against the case's criteria, in zeos's own terms.

        Nothing in `zeos.demo.criteria` knows this game exists, so a capability
        that stops holding reports as one failing criterion.
        """
        from zeos.core.events import JobSpawned
        from zeos.demo.criteria import EvalContext, evaluate_all, parse_criteria

        if not self.criteria:
            return []
        events = self.kernel.events
        ctx = EvalContext(
            events=events,
            world=self.kernel.world,
            job_names={
                e.job: e.descriptor for e in events if isinstance(e, JobSpawned)
            },
        )
        return [
            {
                "id": v.criterion_id,
                "kind": v.kind,
                "passed": v.passed,
                "detail": v.detail,
                "because": v.because,
            }
            for v in evaluate_all(parse_criteria(self.criteria), ctx)
        ]

    def journal(self) -> list[dict]:
        """The kernel events of the tick just stepped, as zeos journal records.

        `encode_record` rather than a shape of our own, so the file the writer
        produces is a zeos journal that `zeos inspect` reads unchanged.
        """
        events = self.kernel.events[self._journalled :]
        first = self._journalled
        self._journalled = len(self.kernel.events)
        return [encode_record(first + n, event) for n, event in enumerate(events)]
