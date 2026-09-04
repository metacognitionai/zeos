# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Run one episode of Space Invaders with a player at the controls.

    agent --player random                         # no key, no server, no cost
    agent --player openai --effort none           # any OpenAI Responses endpoint
    agent --player claude --effort high           # the Claude API
    agent --player zeos-openai --effort none       # the same endpoint, scheduled
    agent --player zeos-claude --effort none       # the Claude API, scheduled

Endpoint, key and model id come from `.env` (see `.env.example`); every flag
below overrides it. Every run writes a directory under `runs/` -- what was run,
every frame of the world, and every decision with its prompt and reasoning --
so a run can be replayed and compared after the fact. See `zeos_space_invaders.runlog`.
"""

import argparse
import sys
from pathlib import Path

from .clocks import RealtimeRunner, ZeosRealtimeRunner, run_episode
from .clocks.outcome import announce
from .game import FIRE_CHANCE, MAX_STEPS
from .players import (
    EFFORTS,
    ClaudePlayer,
    OpenAIPlayer,
    RandomPlayer,
    SDKMissing,
    ThinkingNotDisabled,
)
from .players.claude import DEFAULT_MODEL as CLAUDE_MODEL
from .players.claude import default_base_url as claude_base_url
from .players.openai_compat import (
    DEFAULT_BASE_URL,
    default_base_url,
    default_model,
)
from .players.openai_compat import DEFAULT_MODEL as OPENAI_MODEL
from .players.zeos import ZeosDriver, build_machine, case_path, kernel_version
from .runlog import RunWriter
from .utils import VIEWS, add_rules_flags, load_env, rules_of, settings_flags

#: `--player` names, and which backend each reaches. The `zeos-` pair run the same
#: two vendors through the kernel; across a pair the game, the seed and the runlog
#: are the same, and scheduling is not the only variable. See AGENTS.md.
PLAYERS = ("random", "openai", "claude", "zeos-openai", "zeos-claude")


class _NeverRaised(Exception):
    """Stands in for a Claude SDK error when the SDK is not installed."""


def anthropic_errors():
    """The three Claude SDK failures that deserve a message of their own.

    The SDK is an optional install, so its classes cannot be named at import
    time; nothing can raise them when it is absent either, so the stand-in
    simply leaves those `except` clauses inert.
    """
    try:
        import anthropic
    except ModuleNotFoundError:
        return _NeverRaised, _NeverRaised, _NeverRaised
    return (
        anthropic.AuthenticationError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
    )


NO_CREDENTIALS = (
    "no credentials found: set ANTHROPIC_API_KEY in .env or the environment, "
    "or install the `ant` CLI and run `ant auth login`"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="a JSON file of these same flags, underscores for hyphens, in a "
        "`board` object and a `play` object, read before the command line so a "
        "flag given here wins over the file. `settings_default.json` and "
        "`settings_ablation.json` are the two the documentation runs",
    )
    parser.add_argument(
        "--player",
        choices=list(PLAYERS),
        default="random",
        help="`random` is the no-cost baseline and the smoke test; `openai` is "
        "any endpoint serving the OpenAI Responses API, local or hosted; "
        "`claude` is the Claude API. The `zeos-` pair reach the same two "
        "backends through the zeos kernel, as a deliberating job a reflex can "
        "preempt (default: random)",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORTS,
        default=None,
        help="how hard the model reasons before answering. `none` means no "
        "reasoning at all, which is checked rather than trusted. Omit to "
        "leave the server's own default alone",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"default {OPENAI_MODEL} ($OPENAI_MODEL) for openai, "
        f"{CLAUDE_MODEL} ($ANTHROPIC_MODEL) for claude",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"$OPENAI_BASE_URL, else {DEFAULT_BASE_URL}. For claude, "
        "$ANTHROPIC_BASE_URL, else the official endpoint",
    )
    parser.add_argument(
        "--view",
        choices=sorted(VIEWS),
        default="lead",
        help="how the state is shown -- the single biggest lever on whether a "
        "model can play. `lead` (the default) is coordinates, cues and the aim "
        "point; `grid` is the ascii board and is the experiment's control, which "
        "scores below the random baseline",
    )
    parser.add_argument(
        "--seed", type=int, help="fixes the episode, for a fair comparison"
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="how many previous boards the model is shown",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="override the system prompt; by default it is built from the view "
        "plus the shared rules",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="run directory (default: runs/<stamp>-<player>-<view>-<effort>-seed<n>)",
    )
    parser.add_argument(
        "--clock",
        choices=["realtime", "step"],
        default="realtime",
        help="`realtime`: the world ticks on a wall clock and does not wait for "
        "the agent. `step`: one action = one tick, the deterministic mode",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=0.2,
        help="seconds per world tick in realtime mode (0.2 = human speed)",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=0.2,
        help="seconds the random player spends 'deciding'; a model player's "
        "latency is however long its request takes",
    )
    parser.add_argument(
        "--actions-per-tick",
        type=int,
        default=None,
        help="cap actions landing in one tick. 1 makes --tick a true "
        "slow-motion dial; omit for human-style unlimited input",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask the model for its reply a piece at a time (prompt-loop players "
        "only). On by default. `--no-stream` asks for one lump instead, which is "
        "a different request on the wire and is there to compare against. The "
        "zeos players always stream: the machine reads a completion through a "
        "producer thread, and a job with nothing arriving cannot be preempted",
    )
    parser.add_argument(
        "--fire-chance",
        type=float,
        default=None,
        help=f"chance a monster drops a bomb on any tick (default {FIRE_CHANCE}). "
        "Raise it to put the game under pressure: at the default something is "
        "about to hit you under once an episode, which is too rare to tell an "
        "interrupt's effect from noise",
    )
    add_rules_flags(
        parser,
        "What game to play. Every one of these is recorded in `meta.json`, so a "
        "run played under turned-down rules can never be mistaken for one played "
        "under the real ones. Shrinking the board is the largest single lever on "
        "whether a kernel journal can be read by a person: a 9x8 board is close to "
        "eighty elements into the pilot's context per tick, and a 5x5 board is "
        "thirty. Monster positions can be set exactly, but only from Python -- see "
        "`game.rules.Rules(monsters=...)`.",
    )
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw the board as it is played, the same way `play` draws it. On "
        "by default: watching a model play should look like playing. "
        "`--no-render` prints one line per step instead, which is what you want "
        "when the output is being piped or read afterwards",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="seconds between frames when rendering (0 for no pause)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_player(args, rules):
    """The player, and the one-line banner describing it.

    `rules` because the system prompt states the board's geometry, the monster
    count, the lives and the odds.
    """
    if args.player == "random":
        latency = args.latency if args.clock == "realtime" else 0.0
        player = RandomPlayer(seed=args.seed, latency=latency)
        return player, f"player random | clock {args.clock} | seed {args.seed}"

    prompt = args.prompt.read_text() if args.prompt else None
    effort = args.effort or "default"
    # `zeos-openai` and `zeos-claude` build the same player their unscheduled
    # namesake does; the kernel is wrapped around it in main().
    backend = args.player.removeprefix("zeos-")
    if backend == "openai":
        base_url = args.base_url or default_base_url()
        player = OpenAIPlayer(
            prompt=prompt,
            base_url=base_url,
            model=args.model or default_model(),
            effort=args.effort,
            history=args.history,
            view=VIEWS[args.view](),
            stream=args.stream,
            rules=rules,
        )
        where = f"{player.model} at {base_url}"
    else:
        player = ClaudePlayer(
            prompt=prompt,
            model=args.model,
            effort=args.effort,
            history=args.history,
            base_url=args.base_url,
            view=VIEWS[args.view](),
            stream=args.stream,
            rules=rules,
        )
        endpoint = args.base_url or claude_base_url() or "the official API"
        where = f"{player.model} at {endpoint}"
    if args.player.startswith("zeos-"):
        # The endpoint alone does not say what is different about this arm.
        where += " | zeos kernel: pilot @60, evade @5 (pinned)"
    return player, (
        f"{where} | view {args.view} | effort {effort} | clock {args.clock} "
        f"| seed {args.seed} | history {args.history}"
    )


def main(argv=None):
    load_env()
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(settings_flags(argv, ("board", "play")) + argv)
    scheduled = args.player.startswith("zeos-")

    if scheduled and args.clock == "step":
        # The step clock waits for the agent, which removes the only thing these
        # two players exist to show.
        sys.exit(
            f"--player {args.player} needs --clock realtime: the step clock "
            "waits for the agent, so nothing is ever late and there is nothing "
            "to preempt"
        )

    if not args.stream and scheduled:
        # Not a silent no-op: `APIMachineBase` has no such option, because
        # reading a completion through a producer thread is what makes a decode
        # step preemptible at all.
        sys.exit(
            f"--no-stream cannot apply to {args.player}: the machine always "
            "streams, because the pieces are what the pilot spends token "
            "boundaries on and a job with nothing arriving cannot be preempted. "
            "Use it on --player openai or --player claude"
        )

    effort = args.effort or "default"
    mode = (
        "random" if args.player == "random" else f"{args.player}-{args.view}-{effort}"
    )

    # Built once and handed to the clock, the player and the metadata, so the
    # game a run was played under cannot drift between them.
    rules = rules_of(args)
    try:
        player, banner = build_player(args, rules)
        # `--model`, `--base-url` and `--effort` reach the machine as well as the
        # player: on this arm the machine issues the request, and the two arms
        # of a comparison have to ask the endpoint for the same thing.
        driver = (
            ZeosDriver(
                machine=build_machine(
                    args.player.removeprefix("zeos-"),
                    model=args.model or None,
                    base_url=args.base_url or None,
                    effort=args.effort,
                ),
                # The same rules text the prompt loop's system prompt carries.
                view=player.view,
                rules=rules,
            )
            if scheduled
            else None
        )
    except SDKMissing as exc:
        # The message already names the command that fixes it, and a traceback
        # would bury it; the machine raises its own `SDKMissing`, hence the
        # construction inside the `try`.
        sys.exit(str(exc))
    auth_failed, rate_limited, unreachable = anthropic_errors()
    config = describe(args, player, scheduled, driver)
    run = (
        RunWriter(args.out, config)
        if args.out
        else RunWriter.create(f"{mode}-seed{args.seed}", config)
    )
    print(banner)
    print(f"logging to {run.path}\n")
    try:
        with run:
            if driver is not None:
                runner = ZeosRealtimeRunner(
                    driver,
                    player.view,
                    seed=args.seed,
                    tick_seconds=args.tick,
                    max_ticks=args.max_steps,
                    max_seconds=args.max_seconds,
                    render=args.render,
                    actions_per_tick=args.actions_per_tick,
                    rules=rules,
                )
                summary = run.finish(runner.run(run=run))
            elif args.clock == "realtime":
                runner = RealtimeRunner(
                    player,
                    seed=args.seed,
                    tick_seconds=args.tick,
                    max_ticks=args.max_steps,
                    max_seconds=args.max_seconds,
                    render=args.render,
                    actions_per_tick=args.actions_per_tick,
                    rules=rules,
                )
                summary = run.finish(runner.run(run=run))
            else:
                summary = run.finish(
                    run_episode(
                        player,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        run=run,
                        verbose=not args.quiet,
                        render=args.render,
                        delay=args.delay,
                        rules=rules,
                    )
                )
    except ThinkingNotDisabled as exc:
        sys.exit(f"stopped on the first turn: {exc}")
    except ConnectionError:
        sys.exit(
            f"could not reach {args.base_url or default_base_url()}: "
            "is the model server running?"
        )
    except auth_failed as exc:
        sys.exit(f"credentials rejected by the API (401): {exc.message}")
    except TypeError as exc:
        # The SDK raises a bare TypeError when it resolves no credential at all.
        if "authentication" not in str(exc).lower():
            raise
        sys.exit(NO_CREDENTIALS)
    except rate_limited as exc:
        sys.exit(f"rate limited: {exc}")
    except unreachable as exc:
        sys.exit(f"could not reach the API: {exc}")
    finally:
        # The driver owns a thread pool with a pass possibly still in it, which
        # would keep the process alive past the last tick.
        if driver is not None:
            driver.close()

    announce(summary, clock=run.meta["clock"], player=args.player, path=run.path)


def describe(args, player, scheduled, driver=None):
    """Everything needed to run this again, written before the first tick.

    The prompt goes in whole rather than by name, because `--prompt` takes a
    file that can change underneath a run.
    """
    client = getattr(player, "client", None)
    return {
        "player": args.player,
        # `realtime` for the scheduled players too: who is driving is `player`,
        # and recording it twice would invent a third pacing nobody can ask for.
        "clock": args.clock,
        # A kernel journal is only as reproducible as the zeos revision that
        # wrote it.
        "zeos": kernel_version() if scheduled else None,
        # Bindings, read and write sets and capabilities are declared and never
        # journalled, so `zeos debug` cannot draw the wiring without the case.
        "case": case_path() if scheduled else None,
        "model": None if args.player == "random" else player.model,
        "base_url": str(getattr(client, "base_url", "")) or None,
        "view": None if args.player == "random" else args.view,
        "effort": args.effort,
        "seed": args.seed,
        "history": None if args.player == "random" else args.history,
        "max_steps": args.max_steps,
        "max_seconds": args.max_seconds,
        "tick_seconds": args.tick if args.clock == "realtime" else None,
        "actions_per_tick": args.actions_per_tick,
        # Always the numbers that were played, flat, so a run under turned-down
        # rules cannot pass for one under the real ones; `fire_chance` keeps its
        # own key and column.
        **rules_of(args).fields(),
        # What the wire did: `True` for the scheduled players, whose machine
        # streams by construction, and `None` for `random`, which contacts
        # nothing.
        "stream": None
        if args.player == "random"
        else (True if scheduled else args.stream),
        # What the machine does with the tail of a cancelled completion: the
        # machine's policy rather than a switch, because under a syscall schema
        # a half-answer is a fragment of a call that never happened.
        "partial": getattr(driver.machine, "partial", None) if driver else None,
        "prompt": getattr(player, "prompt", None),
    }


if __name__ == "__main__":
    main()
