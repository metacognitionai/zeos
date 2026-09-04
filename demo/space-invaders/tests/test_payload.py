# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The projection the page is drawn from.

The browser draws what these functions return, so this is where the viewer's
content is checked. Runs are built by hand rather than played, because what is
under test is the join between frames and decisions.
"""

import pytest
from stubs import BRISK, TICK, stub_machine

from zeos_space_invaders.game import Game, snapshot
from zeos_space_invaders.runlog import Decision, RunReader, RunWriter
from zeos_space_invaders.web import payload


def world(tick, *, score=0, lives=3, monsters=2, player=4, over=False, won=False):
    """One tick of a world, in the shape `RunWriter.frame` takes."""
    return {
        "ticks": tick,
        "score": score,
        "lives": lives,
        "player": player,
        "monsters": {i: (0, i) for i in range(monsters)},
        "missile": None,
        "dangers": [],
        "can_shoot": True,
        "won": won,
        "over": over,
    }


def episode(path, frames, decisions=(), config=None):
    """A run directory with exactly the frames and decisions given."""
    with RunWriter(path, config or {"player": "model", "clock": "realtime"}) as run:
        for info in frames:
            run.frame(info, over=info.get("over", False))
        for decision in decisions:
            run.decision(decision)
        run.finish({"outcome": "lost", "score": 0, "decisions": len(decisions)})
    return RunReader(path)


# --- the index ---------------------------------------------------------------


def test_the_index_lists_newest_first(tmp_path):
    for name in ("20260101-000000-a", "20260301-000000-c", "20260201-000000-b"):
        episode(tmp_path / name, [world(0)])
    assert [row["path"][-1] for row in payload.index(tmp_path)] == ["c", "b", "a"]


def test_a_comparison_is_a_row_with_its_episodes_under_it(tmp_path):
    root = tmp_path / "20260101-000000-compare"
    for seed in (0, 1):
        episode(
            root / "episodes" / f"random-seed{seed}",
            [world(0)],
            config={"player": "random", "seed": seed},
        )
    RunWriter(root, {"kind": "compare", "players": ["random"], "seeds": 2}).close()
    (root / "table.json").write_text("{}")

    row = payload.index(tmp_path)[0]
    assert row["is_compare"] and row["finished"]
    assert [e["seed"] for e in row["episodes"]] == [0, 1]
    assert row["episodes"][0]["path"].endswith("compare/episodes/random-seed0")


def test_a_comparison_is_not_called_unfinished_for_having_no_summary(tmp_path):
    """It never has one -- its verdict is the table -- so the question differs."""
    root = tmp_path / "20260101-000000-compare"
    RunWriter(root, {"kind": "compare", "players": [], "seeds": 0}).close()
    assert payload.index(tmp_path)[0]["finished"] is False
    (root / "table.json").write_text("{}")
    assert payload.index(tmp_path)[0]["finished"] is True


def test_a_run_that_never_finished_says_so(tmp_path):
    with RunWriter(tmp_path / "20260101-000000-half", {"player": "openai"}) as run:
        run.frame(world(0))
    assert payload.index(tmp_path)[0]["finished"] is False


def test_one_unreadable_run_does_not_take_the_index_down(tmp_path):
    episode(tmp_path / "20260102-000000-fine", [world(0)])
    old = tmp_path / "20260101-000000-old"
    old.mkdir()
    (old / "meta.json").write_text('{"schema": 1, "run": "old"}')

    rows = payload.index(tmp_path)
    assert rows[0]["path"].endswith("fine")
    assert "schema 1" in rows[1]["unreadable"]


def test_a_directory_that_is_not_a_run_is_skipped(tmp_path):
    (tmp_path / "notes").mkdir()
    episode(tmp_path / "20260101-000000-a", [world(0)])
    assert len(payload.index(tmp_path)) == 1


def test_usage_reaches_the_index_as_numbers_a_column_can_sort(tmp_path):
    """A dict is not sortable, the two SDKs do not name these alike, and a total
    the vendor sent is preferred to a sum that would double-count it."""
    with RunWriter(tmp_path / "20260101-000000-a", {"player": "openai"}) as run:
        run.frame(world(0))
        run.finish(
            {
                "outcome": "lost",
                "score": 0,
                "usage": {"input_tokens": 100, "output_tokens": 7, "total_tokens": 107},
            }
        )
    row = payload.index(tmp_path)[0]
    assert row["tokens"] == 107 and row["output_tokens"] == 7
    assert "usage" not in row


def test_a_run_that_reported_no_usage_has_no_token_count_rather_than_zero(tmp_path):
    """Blank sorts to the end of the column; a zero ties with a genuinely free
    run."""
    episode(tmp_path / "20260101-000000-a", [world(0)])
    assert payload.index(tmp_path)[0]["tokens"] is None


def test_tokens_are_summed_when_the_vendor_sent_no_total(tmp_path):
    with RunWriter(tmp_path / "20260101-000000-a", {"player": "claude"}) as run:
        run.frame(world(0))
        run.finish(
            {
                "outcome": "lost",
                "score": 0,
                "usage": {"input_tokens": 40, "output_tokens": 2},
            }
        )
    assert payload.index(tmp_path)[0]["tokens"] == 42


def test_the_measured_actions_per_tick_is_not_confused_with_the_setting(tmp_path):
    """The name is a cap in `meta` and a rate in `summary`, and `row()` merges
    the two."""
    with RunWriter(
        tmp_path / "20260101-000000-half", {"player": "openai", "actions_per_tick": 1}
    ) as run:
        run.frame(world(0))
    assert payload.index(tmp_path)[0]["actions_per_tick"] is None

    with RunWriter(
        tmp_path / "20260102-000000-done", {"player": "openai", "actions_per_tick": 1}
    ) as run:
        run.frame(world(0))
        run.finish({"outcome": "lost", "score": 0, "actions_per_tick": 0.25})
    assert payload.index(tmp_path)[0]["actions_per_tick"] == 0.25


def test_the_stream_switch_is_two_words_and_an_empty_cell(tmp_path):
    """A boolean would filter as a number, and a run that never had the setting
    must read as empty rather than as one somebody turned off."""
    for stamp, stream in (("01", True), ("02", False), ("03", None)):
        episode(
            tmp_path / f"202601{stamp}-000000-run",
            [world(0)],
            config={"player": "zeos-openai", "stream": stream},
        )
    # Newest first, so the run with no setting at all comes back first.
    assert [row["stream"] for row in payload.index(tmp_path)] == [None, "off", "on"]


def test_the_rules_a_run_was_played_under_reach_the_table(tmp_path):
    """A run whose monster fire was turned up must not read like one that was
    not."""
    for stamp, fire in (("01", 0.15), ("02", 0.75), ("03", None)):
        episode(
            tmp_path / f"202601{stamp}-000000-run",
            [world(0)],
            config={"player": "openai", "fire_chance": fire},
        )
    # Newest first; a run from before the setting existed leaves the cell empty.
    assert [row["fire_chance"] for row in payload.index(tmp_path)] == [None, 0.75, 0.15]


def test_a_comparison_row_carries_the_means_over_its_episodes(tmp_path):
    """The head of a group holds numbers or it sinks to the end of every sortable
    column."""
    root = tmp_path / "20260101-000000-compare"
    for seed, score, outcome in ((0, 10, "won"), (1, 30, "lost")):
        with RunWriter(
            root / "episodes" / f"random-seed{seed}", {"player": "random", "seed": seed}
        ) as run:
            run.frame(world(0))
            run.finish({"outcome": outcome, "score": score, "ticks": 20 + 2 * seed})
    RunWriter(root, {"kind": "compare", "players": ["random", "openai"]}).close()

    row = payload.index(tmp_path)[0]
    # An episode answers "did it finish" itself, because the table asks each line.
    assert [e["finished"] for e in row["episodes"]] == [True, True]
    assert row["player"] == "random, openai"
    assert row["score"] == 20.0 and row["ticks"] == 21
    assert row["wins"] == 1 and row["episode_count"] == 2
    # And no average of things a comparison has no single one of.
    assert row.get("view") is None and row.get("seed") is None


def test_a_comparison_with_no_finished_episode_averages_nothing(tmp_path):
    root = tmp_path / "20260101-000000-compare"
    with RunWriter(root / "episodes" / "random-seed0", {"player": "random"}) as run:
        run.frame(world(0))
    RunWriter(root, {"kind": "compare", "players": ["random"]}).close()

    row = payload.index(tmp_path)[0]
    assert row["score"] is None and row["ticks"] is None
    assert row["wins"] == 0 and row["episode_count"] == 1


# --- the join ----------------------------------------------------------------


def test_a_decision_belongs_to_the_tick_its_action_landed_on(tmp_path):
    """Not the tick it was chosen on: that is where the *prompt* belongs."""
    run = episode(
        tmp_path / "r",
        [world(t) for t in range(6)],
        [Decision(by="pilot", action="left", tick=1, tick_applied=4)],
    )
    data = payload.episode(run)
    assert data["by_tick"][4] == [0]
    assert data["by_tick"][1] == []


def test_the_ticks_spent_waiting_point_at_what_is_being_waited_for(tmp_path):
    """Those ticks have a frame and no decision, and without this they read as
    nothing happening when the expensive thing is exactly what is happening."""
    run = episode(
        tmp_path / "r",
        [world(t) for t in range(6)],
        [Decision(by="pilot", action="left", tick=1, tick_applied=4)],
    )
    waiting = payload.episode(run)["in_flight"]
    assert waiting[:6] == [None, 0, 0, 0, None, None]


def test_a_decision_that_cost_nothing_waits_on_no_tick(tmp_path):
    run = episode(
        tmp_path / "r",
        [world(t) for t in range(3)],
        [Decision(by="random", action="left", tick=1, tick_applied=1)],
    )
    data = payload.episode(run)
    assert data["in_flight"] == [None, None, None]
    assert data["by_tick"][1] == [0]


def test_several_decisions_can_land_on_one_tick(tmp_path):
    """A human out-mashes the clock, and the log has to keep all of it."""
    run = episode(
        tmp_path / "r",
        [world(t) for t in range(2)],
        [
            Decision(by="human", action=a, tick=0, tick_applied=0)
            for a in ("left", "left", "shoot")
        ],
    )
    assert payload.episode(run)["by_tick"][0] == [0, 1, 2]


def test_the_prompt_is_offered_once_rather_than_on_every_line(tmp_path):
    run = episode(tmp_path / "r", [world(0)], config={"prompt": "RULES", "player": "x"})
    data = payload.episode(run)
    assert data["prompt"] == "RULES" and "prompt" not in data["meta"]


# --- marks -------------------------------------------------------------------


def test_marks_name_the_moments_the_design_makes_a_claim_about(tmp_path):
    run = episode(
        tmp_path / "r",
        [world(0), world(1, monsters=1), world(2, lives=2), world(3, over=True)],
        [Decision(by="evade", action="left", tick=2, tick_applied=2)],
    )
    marks = {m["tick"]: m["label"] for m in payload.episode(run)["marks"]}
    assert marks[1] == "kill"
    assert marks[2] == "life lost", "a life lost outranks the reflex on the same tick"
    assert marks[3] == "over"


def test_an_unreadable_reply_is_worth_jumping_to(tmp_path):
    run = episode(
        tmp_path / "r",
        [world(t) for t in range(3)],
        [Decision(by="model", action="left", tick=1, tick_applied=1, parsed=False)],
    )
    assert payload.episode(run)["marks"] == [{"tick": 1, "label": "unreadable reply"}]


def test_an_episode_with_nothing_in_it_has_no_marks(tmp_path):
    run = episode(tmp_path / "r", [world(0), world(1)])
    assert payload.episode(run)["marks"] == []


# --- a comparison ------------------------------------------------------------


def test_a_comparison_averages_over_the_seeds_a_player_played(tmp_path):
    root = tmp_path / "cmp"
    for seed, score in ((0, 10), (1, 30)):
        with RunWriter(
            root / "episodes" / f"random-seed{seed}", {"player": "random", "seed": seed}
        ) as run:
            run.frame(world(0))
            run.finish(
                {
                    "outcome": "lost",
                    "score": score,
                    "decisions": 5,
                    "unparseable": 0,
                    "per_decision": 0.1,
                    "ticks": 5,
                }
            )
    RunWriter(root, {"kind": "compare"}).close()

    line = payload.comparison(RunReader(root))["players"][0]
    assert line["player"] == "random" and line["episodes"] == 2
    assert line["score"] == 20.0 and line["score_sd"] == 10.0
    assert line["wins"] == 0


def test_a_comparisons_episodes_carry_their_own_aimed_share(tmp_path):
    """Per episode and not on the index, which opens nothing but `meta.json` and
    `summary.json`."""
    root = tmp_path / "cmp"
    with RunWriter(
        root / "episodes" / "random-seed0", {"player": "random", "seed": 0}
    ) as run:
        run.frame(world(0, player=1, monsters=3))  # monsters sit in cols 0..2
        run.frame(world(1, player=7, monsters=3))  # nothing in column 7
        run.decision(Decision(by="random", action="shoot", tick=0))
        run.decision(Decision(by="random", action="shoot", tick=1))
        run.decision(Decision(by="random", action="left", tick=1))
        run.finish(
            {
                "outcome": "lost",
                "score": 0,
                "decisions": 3,
                "unparseable": 0,
                "per_decision": 0.1,
                "ticks": 2,
            }
        )
    RunWriter(root, {"kind": "compare"}).close()

    data = payload.comparison(RunReader(root))
    assert data["episodes"][0]["aimed"] == 50.0  # one of two shots was aimed
    assert "usage" not in data["episodes"][0]
    assert data["players"][0]["tokens"] is None


# --- the kernel lane ---------------------------------------------------------


@pytest.fixture
def scheduled(tmp_path):
    """A short zeos run, kernel journal and all."""
    from zeos_space_invaders.clocks import ZeosRealtimeRunner
    from zeos_space_invaders.players.zeos import ZeosDriver
    from zeos_space_invaders.utils import VIEWS

    driver = ZeosDriver(machine=stub_machine("shoot", delay=BRISK))
    runner = ZeosRealtimeRunner(
        driver, VIEWS["lead"](), seed=7, tick_seconds=TICK, max_ticks=10
    )
    meta = {"player": "zeos-openai", "clock": "zeos"}
    with RunWriter(tmp_path / "zeos", meta) as run:
        run.finish(runner.run(run=run))
    driver.close()
    return RunReader(tmp_path / "zeos")


def test_the_kernel_lane_has_one_frame_per_world_tick(scheduled):
    """Not one per token boundary: there are ten of those to a tick and the
    scrubber does not move in them."""
    data = payload.episode(scheduled)
    assert len(data["kernel"]) == len(data["frames"])
    assert [view["tick"] for view in data["kernel"]] == list(range(len(data["frames"])))


def test_the_kernel_lane_says_what_the_pilot_is_blocked_on(tmp_path):
    """Driven by hand with no board on offer, so the pilot is caught blocked
    whatever the body costs to load."""
    import time

    from zeos_space_invaders.players.zeos import ZeosDriver

    driver = ZeosDriver(machine=stub_machine("shoot", delay=BRISK))
    for _ in range(20):
        driver.run_kernel(deadline=time.monotonic() + 0.1)
    meta = {"player": "zeos-openai", "clock": "zeos"}
    with RunWriter(tmp_path / "zeos", meta) as run:
        run.frame(snapshot(Game(seed=7)))
        run.kernel(0, driver.journal())
        run.finish({})
    driver.close()
    lane = payload.episode(RunReader(tmp_path / "zeos"))["kernel"]
    blocked = [
        job
        for view in lane
        for job in view["jobs"]
        if job["name"] == "pilot" and job["state"] == "blocked"
    ]
    assert blocked, "the pilot never waited"
    # The next board is the pilot's only blocking read; waiting for the model is
    # a running job whose decode steps produce no tokens.
    assert {job["blocked_on"] for job in blocked} == {"game.state"}


def test_the_lane_carries_forward_through_a_tick_the_kernel_slept_in(scheduled):
    data = payload.episode(scheduled)
    assert all(view is not None for view in data["kernel"])


def test_the_ticker_says_when_it_had_to_drop_events(tmp_path, monkeypatch, scheduled):
    """A ticker that stops at the cap looks like a tick that did that little."""
    monkeypatch.setattr(payload, "EVENT_LIMIT", 2)
    events = payload.episode(scheduled)["kernel_events"][0]
    assert len(events) == 3 and events[-1]["text"].startswith("+")


def test_a_run_with_no_kernel_has_no_kernel_lane(tmp_path):
    run = episode(tmp_path / "r", [world(0)])
    data = payload.episode(run)
    assert data["kernel"] is None and data["kernel_events"] is None
