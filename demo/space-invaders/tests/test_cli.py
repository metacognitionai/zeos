# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""`agent`: the flags it refuses, the directory it opens, and what it prints.

The refusals matter as much as the runs: a silently ignored flag leaves you
believing you applied a correction you did not.
"""

import json
from pathlib import Path

import pytest

from zeos_space_invaders import cli
from zeos_space_invaders.players.zeos import CASE_ROOT
from zeos_space_invaders.runlog import RunReader

#: A finished step-clock summary, for the printing tests.
SUMMARY = {
    "outcome": "lost",
    "score": 30,
    "lives": 0,
    "monsters_left": 4,
    "ticks": 40,
    "decisions": 40,
    "seconds": 3.0,
    "unparseable": 0,
    "usage": {},
}


def agent(*argv):
    return cli.main(list(argv))


def test_a_step_clock_run_writes_a_complete_directory(tmp_path, capsys):
    out = tmp_path / "r"
    agent(
        "--player",
        "random",
        "--clock",
        "step",
        "--seed",
        "1",
        "--max-steps",
        "6",
        "--out",
        str(out),
        "--quiet",
    )

    episode = RunReader(out)
    assert episode.meta["player"] == "random" and episode.meta["clock"] == "step"
    assert episode.meta["seed"] == 1 and "commit" in episode.meta
    assert episode.summary["decisions"] == 6
    assert len(list(episode.frames())) == 7
    assert "summary written to" in capsys.readouterr().out


def test_a_realtime_run_reports_what_being_late_cost(tmp_path, capsys):
    out = tmp_path / "r"
    agent(
        "--player",
        "random",
        "--clock",
        "realtime",
        "--seed",
        "1",
        "--tick",
        "0.01",
        "--max-steps",
        "8",
        "--latency",
        "0.02",
        "--out",
        str(out),
    )

    printed = capsys.readouterr().out
    assert "ticks per move" in printed and "arrived too late" in printed
    assert RunReader(out).meta["tick_seconds"] == 0.01


def test_the_default_directory_is_named_after_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent(
        "--player",
        "random",
        "--clock",
        "step",
        "--seed",
        "3",
        "--max-steps",
        "2",
        "--quiet",
    )
    made = list((tmp_path / "runs").iterdir())
    assert len(made) == 1 and made[0].name.endswith("-random-seed3")


def settings_file(tmp_path, board=None, play=None):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"board": board or {}, "play": play or {}}))
    return str(path)


def test_a_settings_file_supplies_the_flags_the_run_is_played_under(tmp_path):
    """The two files the documentation runs reach `agent` this way."""
    out = tmp_path / "r"
    agent(
        "--settings",
        settings_file(
            tmp_path,
            board={"width": 12, "height": 16, "march_group": 1},
            play={"seed": 3},
        ),
        "--player",
        "random",
        "--clock",
        "step",
        "--max-steps",
        "4",
        "--out",
        str(out),
        "--quiet",
    )

    meta = RunReader(out).meta
    assert (meta["width"], meta["height"]) == (12, 16)
    assert meta["march_group"] == 1 and meta["seed"] == 3


def test_a_flag_on_the_command_line_wins_over_the_file(tmp_path):
    """The same precedence `.env` has: the file is a starting point, not a lock."""
    out = tmp_path / "r"
    agent(
        "--settings",
        settings_file(tmp_path, board={"width": 12}, play={"seed": 3}),
        "--width",
        "20",
        "--player",
        "random",
        "--clock",
        "step",
        "--max-steps",
        "4",
        "--out",
        str(out),
        "--quiet",
    )

    meta = RunReader(out).meta
    assert meta["width"] == 20 and meta["seed"] == 3


def test_a_null_in_the_file_leaves_the_flag_alone(tmp_path):
    """`null` is how a file says "whatever the default is", as `.env.example` does."""
    out = tmp_path / "r"
    agent(
        "--settings",
        settings_file(tmp_path, play={"model": None, "seed": 3}),
        "--player",
        "random",
        "--clock",
        "step",
        "--max-steps",
        "4",
        "--out",
        str(out),
        "--quiet",
    )

    assert RunReader(out).meta["model"] is None


def test_the_prompt_is_recorded_whole_not_by_name():
    """`--prompt` takes a file, and a file can change underneath a run."""
    from stubs import claude_player

    args = cli.build_parser().parse_args(["--player", "claude"])
    config = cli.describe(args, claude_player(prompt="PLAY WELL"), scheduled=False)
    assert config["prompt"] == "PLAY WELL"


@pytest.fixture
def no_sdk(monkeypatch):
    """The vendor SDK absent, whether or not it is installed here."""
    import zeos_space_invaders.players.base as base

    real = base.importlib.import_module

    def missing(module, *args, **kwargs):
        if module in ("anthropic", "openai"):
            raise ModuleNotFoundError(f"No module named {module!r}", name=module)
        return real(module, *args, **kwargs)

    monkeypatch.setattr(base.importlib, "import_module", missing)


def test_a_run_that_never_started_leaves_no_directory(tmp_path, no_sdk):
    """`summary.json` missing means "died halfway"; an empty dir would lie."""
    out = tmp_path / "r"
    with pytest.raises(SystemExit):
        agent("--player", "claude", "--out", str(out))
    assert not out.exists()


def test_a_missing_sdk_exits_with_the_command_that_fixes_it(no_sdk):
    with pytest.raises(SystemExit) as exit:
        agent("--player", "claude")
    assert "uv sync --extra claude" in str(exit.value)


def test_the_scheduled_players_refuse_the_step_clock():
    with pytest.raises(SystemExit) as exit:
        agent("--player", "zeos-openai", "--clock", "step")
    assert "needs --clock realtime" in str(exit.value)


def test_only_the_prompt_loop_players_take_no_stream(stub_backends, tmp_path):
    """`--no-stream` changes the request on a prompt loop and cannot apply to the
    zeos players, whose machine always streams through a producer thread."""
    with pytest.raises(SystemExit) as exit:
        agent("--player", "zeos-openai", "--no-stream")
    assert "the machine always streams" in str(exit.value)

    out = tmp_path / "unstreamed"
    agent(
        "--player",
        "openai",
        "--no-stream",
        "--clock",
        "step",
        "--max-steps",
        "2",
        "--out",
        str(out),
    )
    assert json.loads((out / "meta.json").read_text())["stream"] is False


def test_streaming_is_on_by_default_and_recorded_as_asked_for(stub_backends, tmp_path):
    """It is part of the request, so it belongs in meta.json and is None for a
    player that cannot use it."""
    out = tmp_path / "streamed"
    agent(
        "--player", "zeos-openai", "--tick", "0", "--max-steps", "3", "--out", str(out)
    )
    meta = json.loads((out / "meta.json").read_text())
    # `True` records what the wire did; `syscall` is the structured default.
    assert meta["stream"] is True and meta["partial"] == "syscall"

    plain = tmp_path / "plain"
    agent(
        "--player", "openai", "--clock", "step", "--max-steps", "2", "--out", str(plain)
    )
    plain_meta = json.loads((plain / "meta.json").read_text())
    assert plain_meta["stream"] is True and plain_meta["partial"] is None

    nowhere = tmp_path / "random"
    agent(
        "--player",
        "random",
        "--clock",
        "step",
        "--max-steps",
        "2",
        "--out",
        str(nowhere),
    )
    assert json.loads((nowhere / "meta.json").read_text())["stream"] is None


@pytest.mark.parametrize("cap", ["1", "2"])
def test_the_scheduled_player_takes_a_cap_like_every_other_player(
    cap, stub_backends, tmp_path
):
    """The budget belongs to the game's stick, so there is nothing
    player-specific to refuse."""
    out = tmp_path / f"cap{cap}"
    agent(
        "--player",
        "zeos-openai",
        "--effort",
        "none",
        "--view",
        "lead",
        "--seed",
        "7",
        "--tick",
        "0.001",
        "--max-steps",
        "6",
        "--actions-per-tick",
        cap,
        "--out",
        str(out),
    )
    assert RunReader(out).meta["actions_per_tick"] == int(cap)


# --- the banner, and the two backends behind it -----------------------------


@pytest.fixture
def stub_backends(monkeypatch):
    """Every path to a network replaced: the prompt-loop players' vendor client
    and the one the scheduled player's machine holds of its own."""
    from stubs import claude_player, openai_player, stub_machine

    #: What each factory was handed, so a test can assert that `--model` and
    #: `--base-url` reached the machine as well as the player.
    seen: dict[str, dict] = {}

    def player(kind, factory):
        def build(**kw):
            seen[kind] = kw
            return factory(view=kw.get("view"))

        return build

    def machine(backend="openai", **kw):
        seen["machine"] = {"backend": backend, **kw}
        return stub_machine()

    monkeypatch.setattr(cli, "OpenAIPlayer", player("openai", openai_player))
    monkeypatch.setattr(cli, "ClaudePlayer", player("claude", claude_player))
    monkeypatch.setattr(cli, "build_machine", machine)
    return seen


def test_the_banner_says_where_an_openai_run_is_pointing(stub_backends, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gpu-box:8000/v1")
    args = cli.build_parser().parse_args(
        ["--player", "openai", "--view", "cues", "--effort", "none"]
    )
    _player, banner = cli.build_player(args, cli.rules_of(args))
    assert "http://gpu-box:8000/v1" in banner
    assert "view cues" in banner and "effort none" in banner
    assert "zeos kernel" not in banner


def test_the_banner_names_the_kernel_when_the_run_is_scheduled(stub_backends):
    args = cli.build_parser().parse_args(["--player", "zeos-claude"])
    _player, banner = cli.build_player(args, cli.rules_of(args))
    assert "zeos kernel: pilot @60, evade @5" in banner
    assert "effort default" in banner, "an omitted effort is the server's own"


def test_the_model_and_endpoint_reach_the_machine_not_only_the_player(
    stub_backends,
):
    """On the scheduled arm the machine issues the request, and a run that names
    a model it never contacted is worse than one that names none."""
    args = cli.build_parser().parse_args(
        [
            "--player",
            "zeos-openai",
            "--model",
            "served-model",
            "--base-url",
            "http://box:1/v1",
        ]
    )
    cli.build_player(args, cli.rules_of(args))
    cli.build_machine(
        args.player.removeprefix("zeos-"), model=args.model, base_url=args.base_url
    )
    assert stub_backends["machine"] == {
        "backend": "openai",
        "model": "served-model",
        "base_url": "http://box:1/v1",
    }
    assert stub_backends["openai"]["model"] == "served-model"


def test_a_missing_sdk_on_the_scheduled_arm_is_caught_like_any_other(monkeypatch):
    """One `SDKMissing` class, so the CLI's one `except` covers both halves."""
    from zeos_space_invaders.players import SDKMissing
    from zeos_space_invaders.players.zeos import api_claude, api_openai

    assert api_openai.SDKMissing is SDKMissing
    assert api_claude.SDKMissing is SDKMissing


def test_a_scheduled_run_goes_through_the_kernel_end_to_end(
    stub_backends, tmp_path, capsys
):
    """The whole path: cli -> driver -> kernel -> runner."""
    out = tmp_path / "r"
    agent(
        "--player",
        "zeos-openai",
        "--effort",
        "none",
        "--view",
        "lead",
        "--seed",
        "7",
        "--tick",
        "0.001",
        "--max-steps",
        "8",
        "--out",
        str(out),
    )

    episode = RunReader(out)
    # `realtime`, not a third clock: the world does not wait, and `player` says
    # who drives.
    assert episode.meta["clock"] == "realtime"
    assert episode.meta["zeos"], "no record of which kernel wrote kernel.jsonl"
    # `zeos debug` draws the wiring from the case, so a run has to name it.
    case = Path(episode.meta["case"])
    assert not case.is_absolute(), "a repo-relative path is the one that pastes"
    repo = Path(__file__).resolve().parents[3]
    assert (repo / case) == CASE_ROOT, "the recorded path is not the case that ran"
    # Not one per tick: a tick the pilot spends blocked produces no move.
    assert episode.summary["decisions"] <= episode.summary["ticks"]
    assert {d["by"] for d in episode.decisions()} <= {"pilot", "evade"}
    first = next(iter(episode.kernel()))
    assert first["kind"] == "kernel.started" and first["tick"] == 0
    assert "ticks per move" in capsys.readouterr().out


# --- the failures that get a message rather than a traceback ----------------


def raising(exc):
    """A player that fails the way a real backend fails, on turn one."""
    from stubs import openai_player

    def build(**kw):
        player = openai_player(view=kw.get("view"))
        player.choose = _raise(exc)
        return player

    return build


def _raise(exc):
    def choose(*args, **kw):
        raise exc

    return choose


def test_a_server_that_reasoned_anyway_stops_the_run_on_turn_one(monkeypatch):
    from zeos_space_invaders.players import ThinkingNotDisabled

    monkeypatch.setattr(cli, "OpenAIPlayer", raising(ThinkingNotDisabled("it did")))
    with pytest.raises(SystemExit) as exit:
        agent("--player", "openai", "--effort", "none", "--clock", "step")
    assert "stopped on the first turn: it did" in str(exit.value)


def test_an_unreachable_server_names_the_endpoint_it_tried(monkeypatch):
    monkeypatch.setattr(cli, "OpenAIPlayer", raising(ConnectionError()))
    with pytest.raises(SystemExit) as exit:
        agent(
            "--player", "openai", "--clock", "step", "--base-url", "http://nowhere:9/v1"
        )
    assert "could not reach http://nowhere:9/v1" in str(exit.value)


def test_no_credential_at_all_says_how_to_provide_one(monkeypatch):
    """The SDK raises a bare TypeError when it resolves nothing."""
    monkeypatch.setattr(
        cli, "OpenAIPlayer", raising(TypeError("Could not resolve authentication"))
    )
    with pytest.raises(SystemExit) as exit:
        agent("--player", "openai", "--clock", "step")
    assert "ANTHROPIC_API_KEY" in str(exit.value)


def test_an_unrelated_type_error_keeps_its_traceback(monkeypatch):
    monkeypatch.setattr(cli, "OpenAIPlayer", raising(TypeError("bad argument")))
    with pytest.raises(TypeError, match="bad argument"):
        agent("--player", "openai", "--clock", "step")


def test_the_stand_in_errors_are_inert_without_the_sdk():
    """Nothing can raise the Claude SDK's classes when it is not installed."""
    for stand_in in cli.anthropic_errors():
        assert issubclass(stand_in, Exception)


# --- what gets printed at the end -------------------------------------------


def report_for(player, summary):
    from zeos_space_invaders.clocks.outcome import announce

    announce(summary, clock="step", player=player, path=Path("runs/x"))


def test_tokens_are_reported_when_the_run_billed_any(capsys):
    report_for(
        "openai",
        dict(SUMMARY, usage={"input_tokens": 1_000_000, "output_tokens": 500_000}),
    )
    printed = capsys.readouterr().out
    assert "tokens: input_tokens 1000000, output_tokens 500000" in printed
    assert "$" not in printed, "only the Claude arms get a price"


@pytest.mark.parametrize("player", ["claude", "zeos-claude"])
def test_the_claude_arms_print_what_the_run_cost(player, capsys):
    """Scheduled or not, it bills the same API."""
    report_for(
        player,
        dict(SUMMARY, usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000}),
    )
    # 1M in at $2.00 + 1M out at $10.00
    assert "~$12.00 at base rates" in capsys.readouterr().out


def test_a_free_run_prints_no_token_line(capsys):
    report_for("random", SUMMARY)
    assert "tokens:" not in capsys.readouterr().out


def test_what_a_run_records_about_itself(tmp_path):
    args = cli.build_parser().parse_args(
        ["--player", "random", "--seed", "5", "--view", "cues"]
    )
    from zeos_space_invaders.players import RandomPlayer

    config = cli.describe(args, RandomPlayer(seed=5), scheduled=False)
    assert config["player"] == "random" and config["seed"] == 5
    # A null here is a fact, not a gap: a random player has none of these.
    assert config["model"] is None and config["view"] is None
    assert config["prompt"] is None and config["clock"] == "realtime"
    assert json.dumps(config), "the meta has to be JSON"
