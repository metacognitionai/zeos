# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The ZEOS-scheduled player: contract, reflex, and what happens to a slow reply.

No GPU and no served model: the pilot's completions come from `stubs.stub_machine`,
shaped like a chat-completions stream, and the reflex is a native behaviour that
never reaches a client. The tests are about what the kernel does with the machine.
"""

import json
import time
from pathlib import Path

import pytest
from stubs import BRISK, SLOW, TICK, stub_machine
from zeos.core.ids import JobId
from zeos.machine.base import tokens_from_text

from zeos_space_invaders.clocks import ZeosRealtimeRunner
from zeos_space_invaders.game import ACTIONS, Controls, Game, H, Rules, snapshot
from zeos_space_invaders.players.zeos import (
    CASE_ROOT,
    GAME_STATE,
    REFLEX,
    RESUME,
    Native,
    ZeosDriver,
    build_kernel,
    build_machine,
    case_path,
    dodge,
    encode,
    evade_behaviour,
    threat_reading,
)
from zeos_space_invaders.utils import VIEWS


@pytest.fixture
def quiet_game():
    game = Game(seed=1)
    game.rng.random = lambda: 1.0  # no unplanned monster fire
    game.player = 4
    return game


def drive(*moves, game=None, delay=0.0, **kwargs):
    """A driver over a scripted machine, optionally coupled to a game so that a
    dodge moves the ship and a resume is dirty."""
    driver = ZeosDriver(machine=stub_machine(*moves, delay=delay), **kwargs)
    if game is not None:
        driver.controls = Controls(game, per_tick=1)
    return driver


def wrote_this_tick(driver, name):
    """Whether `name` wrote the stick since the tick began, asked of the journal
    because `Decision.by` names only the last writer."""
    names = {job.job_id: str(job.descriptor.name) for job in driver.kernel.sched.jobs()}
    return any(
        type(event).__name__ == "PipeWritten"
        and str(getattr(event, "pipe", "")) == "game.controls"
        and names.get(getattr(event, "job", None)) == name
        for event in driver.kernel.events[driver._seen_events :]
    )


def run_ticks(driver, game, count, dangers=None):
    """Step a driver `count` times, collecting the decisions it produced."""
    out = []
    for _ in range(count):
        if dangers is not None:
            game.dangers = list(dangers)
        out.append(driver.step(game.render(), snapshot(game)))
    return out


# --- the case ----------------------------------------------------------------


def test_the_case_declares_everything_its_descriptors_bind():
    """`system/pipes.yaml` is the declaration, and the loader refuses a descriptor
    that binds or claims a pipe it does not name."""
    _kernel, bundle = build_kernel(stub_machine())
    assert bundle.name == "space-invaders"
    declared = {str(spec.name) for spec in bundle.pipes}
    for descriptor in bundle.descriptors.values():
        bound = (
            {str(p) for p in descriptor.pipes.all()}
            if hasattr(descriptor.pipes, "all")
            else set()
        )
        granted = {str(c.pipe) for c in descriptor.capabilities}
        assert (bound | granted) <= declared, (
            f"{descriptor.name} reaches for a pipe the case never declares: "
            f"{(bound | granted) - declared}"
        )


def test_every_pipe_the_case_declares_is_the_games():
    """A backend that decodes on demand needs no `model.tokens` shim to block on
    the next token."""
    _kernel, bundle = build_kernel(stub_machine())
    assert {str(spec.name) for spec in bundle.pipes} == {
        "game.state",
        "game.threats",
        "game.controls",
    }


def test_the_reflex_outranks_the_pilot():
    """The reflex's priority comes off its vector, because that is what the kernel
    dispatches it at; the pilot's off its descriptor, because it is booted."""
    _kernel, bundle = build_kernel(stub_machine())
    dispatched = {str(v.handler): v.priority for v in bundle.vectors}
    pilot = next(d for n, d in bundle.descriptors.items() if str(n) == "pilot")
    assert dispatched["evade"] < pilot.priority


def test_only_the_pilot_boots():
    """A handler that boots is a loop, not an interrupt."""
    _kernel, bundle = build_kernel(stub_machine())
    assert tuple(str(name) for name in bundle.boot) == ("pilot",)


def test_the_pilot_has_a_window_so_the_pager_has_work():
    _kernel, bundle = build_kernel(stub_machine())
    pilot = next(d for n, d in bundle.descriptors.items() if str(n) == "pilot")
    assert pilot.context.window > 0


def test_the_pilot_is_told_the_rules_of_the_board_it_flies():
    """The rules text is appended to the body at build time from the `Rules` and
    view of this run."""
    from zeos_space_invaders.game import Rules

    _kernel, bundle = build_kernel(stub_machine())
    pilot = next(d for n, d in bundle.descriptors.items() if str(n) == "pilot")
    assert "## How to choose your action" in pilot.body
    assert "aim point" in pilot.body
    evade = next(d for n, d in bundle.descriptors.items() if str(n) == "evade")
    assert "## How to choose your action" not in evade.body, "handlers fly on their own"

    small = Rules(w=5, h=6, monster_rows=1, monster_cols=2, monster_col_offset=1)
    _kernel, bundle = build_kernel(stub_machine(), rules=small)
    pilot = next(d for n, d in bundle.descriptors.items() if str(n) == "pilot")
    assert "5 columns" in pilot.body and "9 columns" not in pilot.body


# --- the threat sensor -------------------------------------------------------


def test_threat_is_silent_when_nothing_is_coming(quiet_game):
    assert threat_reading(snapshot(quiet_game)) is None


def test_threat_fires_for_our_column_and_the_ones_beside_it(quiet_game):
    quiet_game.dangers = [[H - 2, 0]]  # about to land, but far away
    assert threat_reading(snapshot(quiet_game)) is None
    quiet_game.dangers = [[H - 2, quiet_game.player]]
    assert threat_reading(snapshot(quiet_game)) is not None
    quiet_game.dangers = [[H - 2, quiet_game.player + 1]]
    assert threat_reading(snapshot(quiet_game)) is not None


def test_a_bomb_beside_us_is_a_threat_only_on_its_last_row(quiet_game):
    """The ship's own column is watched two ticks out, the neighbours one."""
    quiet_game.dangers = [[H - 3, quiet_game.player + 1]]
    assert threat_reading(snapshot(quiet_game)) is None
    quiet_game.dangers = [[H - 3, quiet_game.player]]
    assert threat_reading(snapshot(quiet_game)) is not None


def test_a_bomb_beside_us_reads_as_stay(quiet_game):
    """The handler standing still preempts the move already on its way, which is
    the whole protection."""
    quiet_game.dangers = [[H - 2, quiet_game.player - 1]]
    reading = threat_reading(snapshot(quiet_game))
    assert "beside you" in reading and "clear: stay" in reading
    assert dodge(reading) == "shoot"


def test_threat_reports_the_situation_not_the_answer(quiet_game):
    """The handler still decides; it just does not need a GPU to do it."""
    quiet_game.dangers = [[H - 2, 4]]
    reading = threat_reading(snapshot(quiet_game))
    assert "clear:" in reading
    assert reading.strip() not in ("left", "right", "shoot")


@pytest.mark.parametrize(
    "reading,expected",
    [
        ("fire lands in 1 turns in col 4; clear: left", "left"),
        ("fire lands in 1 turns in col 4; clear: right", "right"),
        ("fire lands in 1 turns in col 4; clear: neither", "left"),
        ("fire lands in 1 turn in col 3, beside you; clear: stay right", "shoot"),
    ],
)
def test_dodge_picks_a_clear_side(reading, expected):
    assert dodge(reading) == expected


@pytest.mark.parametrize(
    "reading,expected",
    [
        ("fire lands in 1 turns in col 4; clear: left right", "left"),
        ("fire lands in 1 turns in col 5; clear: left right", "right"),
    ],
)
def test_a_free_choice_is_broken_on_the_ships_column(reading, expected):
    """Both sides clear is most of what the reflex sees, and always answering it
    the same way walks the ship into the left wall."""
    assert dodge(reading) == expected


def test_two_free_dodges_running_cancel_out():
    """Which is the point of breaking the tie on the column rather than on a
    preference: the reflex leaves the ship where it found it."""
    column = 4
    first = dodge(f"fire lands in 1 turns in col {column}; clear: left right")
    column += -1 if first == "left" else 1
    second = dodge(f"fire lands in 1 turns in col {column}; clear: left right")
    assert {first, second} == {"left", "right"}


# --- the reflex as a native behaviour ----------------------------------------


def test_the_reflex_is_registered_and_the_pilot_is_not():
    """Getting this wrong is silent: the reflex would work, four world ticks
    late."""
    machine = build_machine("openai", client=object())
    assert set(machine._behaviours) == {REFLEX}


def test_the_reflex_writes_a_dodge_then_exits():
    """`arrived` on the first step is the descriptor body and the vector's payload
    together, which is why the test is for `clear:`."""
    first = evade_behaviour(
        Native(
            job=JobId(1),
            descriptor=REFLEX,
            step=0,
            arrived="get out of the way fire lands in 1 turns in col 4; clear: right",
        )
    )
    assert first.request.op.value == "write"
    assert [t.text for t in first.request.payload] == ["right"]

    second = evade_behaviour(
        Native(job=JobId(1), descriptor=REFLEX, step=1, arrived="")
    )
    assert second.request.op.value == "exit"


def test_the_reflex_refuses_to_steer_on_nothing():
    """`dodge` falls through to `left` when it finds no clear side, so a handler
    that wrote regardless would steer into the bomb."""
    result = evade_behaviour(
        Native(job=JobId(1), descriptor=REFLEX, step=0, arrived="get out of the way")
    )
    assert result.request.op.value == "exit"
    assert result.tokens == ()


def test_a_native_step_still_enters_the_transcript():
    """The kernel cannot tell a native behaviour from a served one: its output is
    charged, counted and recorded the same way."""
    machine = build_machine("openai", client=object())
    job = JobId(7)
    machine.create_context(job, REFLEX)
    machine.inject(job, tokens_from_text("fire lands in 1 turns in col 4; clear: left"))
    before = machine.stats(job).resident_tokens
    machine.decode(job, allow_control=False)
    assert machine.stats(job).resident_tokens > before


def test_the_reflex_moves_us_and_is_credited(quiet_game):
    quiet_game.dangers = [[H - 2, 4]]
    driver = drive("shoot", delay=0.01)
    driver.step(quiet_game.render(), snapshot(quiet_game))
    dodged = wrote_this_tick(driver, "evade")
    driver.close()
    assert dodged, "the reflex never reached the stick"


def test_the_reflex_needs_no_forward_pass():
    """Proved at the machine with `object()` as the client, so any attempt to
    reach `.chat` raises instead of quietly costing a request."""
    machine = build_machine("openai", client=object())
    job = JobId(3)
    machine.create_context(job, REFLEX)
    machine.inject(job, tokens_from_text("fire lands in 1 turns in col 4; clear: left"))
    first = machine.decode(job, allow_control=False)
    second = machine.decode(job, allow_control=False)
    assert first.request.op.value == "write"
    assert second.request.op.value == "exit"


# --- the pilot flies through the machine -------------------------------------


def test_the_pilot_flies_when_nothing_is_threatening(quiet_game):
    driver = drive("left")
    authors = {d.by for d in run_ticks(driver, quiet_game, 8) if d is not None}
    driver.close()
    assert authors <= {"pilot"}, "something other than the pilot moved the stick"
    assert driver.reflexes == 0


def test_the_pilots_move_reaches_the_stick_through_the_capability_check(quiet_game):
    """The machine produces a WRITE; the capability record is the only reason it
    lands."""
    driver = drive("right")
    decisions = [d for d in run_ticks(driver, quiet_game, 8) if d is not None]
    driver.close()
    assert decisions and decisions[0].action == "right"
    assert any(
        type(event).__name__ == "CapabilityChecked" for event in driver.kernel.events
    )


def test_a_tick_nobody_wrote_the_stick_on_is_not_a_move(quiet_game):
    """The stick is last-write-wins, and replaying the world value would be a
    move nobody chose."""
    driver = drive("shoot", delay=SLOW)
    decisions = run_ticks(driver, quiet_game, 4)
    driver.close()
    assert None in decisions, "every tick claimed a move while the pilot decoded"


def test_the_pilot_blocks_on_the_board_and_nothing_else(quiet_game):
    """Waiting for the model is a running job whose decode steps produce no
    tokens, which is why a reflex can displace it."""
    driver = drive("left")
    run_ticks(driver, quiet_game, 8)
    blocked = {
        str(getattr(event, "pipe", ""))
        for event in driver.kernel.events
        if type(event).__name__ == "JobBlocked"
    }
    driver.close()
    assert blocked == {str(GAME_STATE)}


# --- preemption, which is the claim ------------------------------------------


def test_the_reflex_preempts_a_pilot_that_is_decoding(quiet_game):
    """The pilot must be running when the interrupt lands, and the interrupt must
    not be delivered to a drained kernel."""
    driver = drive("shoot", delay=0.01, game=quiet_game)
    for tick in range(10):
        quiet_game.dangers = [[H - 2, quiet_game.player]] if tick >= 2 else []
        driver.step(quiet_game.render(), snapshot(quiet_game))
    driver.close()
    assert driver.preemptions > 0, "the kernel never took the machine off the pilot"


def test_a_preemption_is_read_off_the_journal_not_our_bookkeeping(quiet_game):
    """The kernel saying it took the machine away is the claim; the driver's I/O
    accounting is only a consequence of it."""
    driver = drive("shoot", delay=0.01, game=quiet_game)
    seen = False
    for tick in range(10):
        quiet_game.dangers = [[H - 2, quiet_game.player]] if tick >= 2 else []
        decision = driver.step(quiet_game.render(), snapshot(quiet_game))
        seen = seen or bool(decision and decision.preempted)
    driver.close()
    assert seen == (driver.preemptions > 0)


def test_no_preemption_leaves_a_live_completion_behind(quiet_game):
    """A resume whose diff is empty injects nothing, so a dodge that changed
    nothing in the pilot's read set would otherwise leave a completion composed
    against a pre-dodge board."""
    driver = drive("shoot", delay=0.01, game=quiet_game)
    live = []
    for tick in range(14):
        quiet_game.dangers = [[H - 2, quiet_game.player]] if tick >= 2 else []
        before = driver.preemptions
        driver.step(quiet_game.render(), snapshot(quiet_game))
        if driver.preemptions > before:
            pilot = [c for c in driver.machine._ctx.values() if c.descriptor == "pilot"]
            live.append(bool(pilot and pilot[0].gen is not None))
        driver.controls.tick()
    driver.close()
    assert live, "no preemption happened, so nothing was tested"
    assert not any(live), "a preempted pilot kept the reply it was composing"


def test_the_driver_takes_its_machine_off_a_kernel_it_was_handed():
    """A caller assembling its own kernel is a shape the dataclass advertises."""
    machine = stub_machine("left")
    kernel, _bundle = build_kernel(machine)
    driver = ZeosDriver(kernel=kernel)
    assert driver.machine is machine
    driver.close()


def test_the_dodge_lands_inside_the_vectors_deadline(quiet_game):
    """`deadline: 5ms` means five token boundaries, because the driver advances
    the kernel's virtual clock one `ns_per_tick` per boundary."""
    driver = drive("shoot", delay=0.01, game=quiet_game)
    for tick in range(8):
        quiet_game.dangers = [[H - 2, quiet_game.player]] if tick >= 2 else []
        driver.step(quiet_game.render(), snapshot(quiet_game))
    verdicts = {v["id"]: v for v in driver.verdicts()}
    driver.close()
    latency = verdicts["the-dodge-lands-inside-the-budget"]
    assert latency["passed"], latency["detail"]


def test_the_case_judges_itself():
    """Through the real runner, because two of the four criteria need the world
    to move for a resume to be dirty."""
    driver = ZeosDriver(machine=stub_machine("shoot", delay=SLOW))
    runner = ZeosRealtimeRunner(
        driver,
        VIEWS["lead"](),
        seed=7,
        # A turn outlasts a tick on purpose: two criteria need the pilot to be
        # mid-completion when the fire lands (see `stubs.SLOW`).
        tick_seconds=TICK,
        max_ticks=40,
        rules=Rules(fire_chance=0.9),
    )
    runner.run()
    verdicts = driver.verdicts()
    driver.close()
    assert verdicts, "the case states criteria and none were evaluated"
    failed = [v["id"] for v in verdicts if not v["passed"]]
    assert not failed, f"criteria the run did not meet: {failed}"


def test_the_recorded_case_path_is_the_one_that_pastes(tmp_path):
    """Repo-relative inside a checkout, so it pastes after `zeos debug`; absolute
    outside one, where there is no root to be relative to."""
    inside = case_path()
    assert not Path(inside).is_absolute()
    assert (Path(__file__).resolve().parents[3] / inside) == CASE_ROOT

    outside = tmp_path / "cases" / "flat"
    assert case_path(outside) == outside.as_posix()


# --- what the interrupted pilot is told --------------------------------------


def test_the_resume_notice_reaches_the_model_as_a_turn(quiet_game):
    """The kernel injects `<RESUME>` like any other content, so it becomes the
    next user turn and `goals/pilot.md` explains what it means."""
    driver = drive("shoot", delay=0.01, game=quiet_game)
    for tick in range(12):
        quiet_game.dangers = [[H - 2, quiet_game.player]] if tick >= 2 else []
        driver.step(quiet_game.render(), snapshot(quiet_game))
    sent = "\n".join(
        message["content"]
        for request in driver.machine.client_stub.requests
        for message in request["messages"]
    )
    driver.close()
    assert RESUME in sent, "the pilot was never told what moved under it"


def test_an_injection_makes_the_completion_in_flight_stale(quiet_game):
    """A board arriving is a prefix change, so the reply being composed was
    composed against a context that no longer exists."""
    machine = stub_machine("left", delay=SLOW)
    driver = ZeosDriver(machine=machine)
    run_ticks(driver, quiet_game, 6)
    driver.close()
    assert machine.cancellations > 0
    assert machine.generations > machine.cancellations - 1


# --- the kernel-level details that were bugs once ----------------------------


def test_only_the_newest_board_is_offered(quiet_game):
    """The vector policy says coalesce: a level-triggered sensor's newest reading
    is the only one worth having."""
    driver = drive("shoot", delay=SLOW)
    for tick in range(4):
        quiet_game.player = tick  # a visibly different board each time
        driver.step(quiet_game.render(), snapshot(quiet_game))
    waiting = [t.text for t in driver.kernel.pipes.get(GAME_STATE).peek()]
    newest = encode(quiet_game.render()).split()
    driver.close()
    # Compared as words, because that is what a pipe carries.
    assert waiting in ([], newest), "a board older than the newest was still queued"


def test_the_board_survives_the_pipe_as_rows():
    """`tokens_from_text` splits on whitespace, so the line breaks are carried as
    a word that `goals/pilot.md` explains to the model."""
    board = "..d..\n.....\n..A.."
    carried = encode(board)
    assert "<nl>" in carried
    assert carried.split() == ["..d..", "<nl>", ".....", "<nl>", "..A.."]


def test_a_finished_job_stops_being_remembered():
    machine = build_machine("openai", client=object())
    job = JobId(1)
    machine.create_context(job, REFLEX)
    assert job in machine._ctx
    machine.destroy_context(job)
    assert job not in machine._ctx


def test_padding_is_never_sent_to_the_model(quiet_game):
    driver = drive("left")
    run_ticks(driver, quiet_game, 8)
    sent = "\n".join(
        message["content"]
        for request in driver.machine.client_stub.requests
        for message in request["messages"]
    )
    driver.close()
    assert "<pad>" not in sent


def test_the_syscall_schema_asks_for_a_move_and_says_nothing_about_pipes():
    """The ABI is the machine's, not the model's: there is nothing in the document
    to get wrong except which of three moves to make."""
    machine = build_machine("openai", client=object())
    schema = machine.syscall_schema()["schema"]
    assert schema["properties"] == {"move": {"enum": list(ACTIONS)}}
    assert schema["required"] == ["move"]
    assert schema["additionalProperties"] is False
    # Their absence is the claim: the model is told about the game and nothing else.
    assert not {"op", "pipe", "steps"} & set(json.dumps(schema).split('"'))


# --- the horizon has to track the game actually being played -----------------


def test_the_horizon_comes_from_the_games_own_rules():
    """A reflex on a 5x5 board taking its horizon from a 9x8 one would report a
    bomb as further away than it is."""
    assert Rules(danger_rows=3).turns_to_land(4) == 1
    assert Rules(danger_rows=1).turns_to_land(4) == 3


def test_fast_fire_still_triggers_the_reflex_in_time(quiet_game):
    """At speed 3 a row-4 danger lands next tick, and a row-counting horizon
    misses it."""
    fast = Game(seed=1, rules=Rules(danger_rows=3))
    fast.rng.random = lambda: 1.0
    fast.player = 4
    fast.dangers = [[4, fast.player]]
    assert threat_reading(snapshot(fast)) is not None
    driver = drive("shoot", delay=0.01)
    driver.step(fast.render(), snapshot(fast))
    dodged = wrote_this_tick(driver, "evade")
    driver.close()
    assert dodged


def test_a_small_board_is_a_short_transcript():
    big = Game(seed=1)
    small = Game(seed=1, rules=Rules(w=5, h=5, monster_rows=1, monster_cols=2))
    assert len(encode(small.render()).split()) < len(encode(big.render()).split()) / 2


def test_the_reflex_reads_the_board_width_from_the_rules():
    """On a 3-wide board standing at column 2 there is no right, and a reflex
    reading the default width would offer one."""
    narrow = Game(seed=1, rules=Rules(w=3, h=5, monsters=((0, 1),)))
    narrow.rng.random = lambda: 1.0
    narrow.player = 2
    narrow.dangers = [[3, 2]]
    reading = threat_reading(snapshot(narrow))
    assert reading is not None and "right" not in reading


# --- the runner, end to end --------------------------------------------------


def test_a_scheduled_episode_reports_the_machines_numbers(quiet_game):
    """`usage` is what the server said it charged; `decoded_words` is what arrived
    and cannot be zero on a run that worked."""
    driver = ZeosDriver(machine=stub_machine("left", delay=BRISK))
    runner = ZeosRealtimeRunner(
        driver, VIEWS["lead"](), seed=7, tick_seconds=TICK, max_ticks=12
    )
    summary = runner.run()
    driver.close()
    assert summary["decoded_words"] > 0
    assert summary["generations"] > 0
    assert summary["pilot_moves"] > 0
    assert "criteria" in summary


def test_the_two_backends_differ_only_in_the_wire_format():
    from zeos_space_invaders.players.zeos import ClaudeAPIMachine, OpenAIAPIMachine

    openai = build_machine("openai", client=object())
    claude = build_machine("claude", client=object())
    assert isinstance(openai, OpenAIAPIMachine)
    assert isinstance(claude, ClaudeAPIMachine)
    for machine in (openai, claude):
        assert set(machine._behaviours) == {REFLEX}
        assert machine.syscall_format == "json"
        assert machine.block_size == openai.block_size


def test_an_unknown_backend_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown backend"):
        build_machine("gemini")


def test_the_driver_closes_the_machine_it_built():
    """A leaked daemon thread can hold a connection the server is billing for."""
    machine = stub_machine("left")
    driver = ZeosDriver(machine=machine)
    driver.step(*_first_tick())
    driver.close()
    time.sleep(0.05)
    assert machine.stranded == 0


def _first_tick():
    game = Game(seed=1)
    game.rng.random = lambda: 1.0
    return game.render(), snapshot(game)
