# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Run several players over the same seeds and print one table.

    compare --seeds 5 --base-url http://localhost:8000/v1

Uses the step clock, so the comparison is about decisions rather than latency.

A comparison is a run of runs. Each episode gets a full run directory of its
own, identical to what `agent` writes, under `episodes/` -- so anything that can
read one episode can read any episode of a comparison, and the table beside them
is a convenience rather than the only way back to the numbers.
"""

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

from .clocks import RealtimeRunner, ZeosRealtimeRunner, run_episode
from .game import FIRE_CHANCE, MAX_STEPS
from .players import OpenAIPlayer, RandomPlayer
from .players.openai_compat import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    default_base_url,
    default_model,
)
from .players.zeos import ZeosDriver, build_machine, case_path
from .runlog import RunReader, RunWriter, aimed, meta_for, stamp
from .utils import VIEWS, load_env, rules_of


def settings(kind):
    """The view and effort a player name asks for, as the player receives them.

    `default` reaches the player as None, so the meta records `null` exactly as
    `agent` does and one condition does not split into two rows when compared.
    """
    kind = kind.removeprefix("zeos-")
    if kind == "random":
        return None, None
    _, view, effort = kind.split("-", 2)
    return view, (None if effort == "default" else effort)


def build(kind, seed, args, rules):
    """`random`, `openai-<view>-<effort>`, or the same prefixed `zeos-`.

    `rules` because the system prompt states the board, and two arms of a
    comparison must be told the same game. The endpoint and model are resolved
    by `main` at parse time, so every episode contacts what `agent` would from
    the same shell.
    """
    if kind == "random":
        # A latency only under a clock that does not wait; without one the
        # baseline lands tens of thousands of actions in an episode.
        return RandomPlayer(
            seed=seed, latency=args.latency if args.clock == "realtime" else 0.0
        )
    view, effort = settings(kind)
    return OpenAIPlayer(
        prompt=args.prompt.read_text() if args.prompt else None,
        base_url=args.base_url,
        model=args.model,
        effort=effort,
        history=args.history,
        view=VIEWS[view](),
        rules=rules,
    )


def _play(player, driver, seed, args, episode, rules):
    """One episode, on whichever clock was asked for.

    The three arms of `agent`, in the same order and with the same arguments,
    or the episode would not be comparable with anything else in `runs/`.
    """
    if driver is not None:
        runner = ZeosRealtimeRunner(
            driver,
            player.view,
            seed=seed,
            tick_seconds=args.tick,
            max_ticks=args.max_steps,
            actions_per_tick=args.actions_per_tick,
            rules=rules,
        )
        return runner.run(run=episode)
    if args.clock == "realtime":
        runner = RealtimeRunner(
            player,
            seed=seed,
            tick_seconds=args.tick,
            max_ticks=args.max_steps,
            actions_per_tick=args.actions_per_tick,
            rules=rules,
        )
        return runner.run(run=episode)
    return run_episode(
        player,
        seed=seed,
        max_steps=args.max_steps,
        run=episode,
        verbose=False,
        rules=rules,
    )


def main(argv=None):
    load_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--players",
        nargs="+",
        default=[
            "random",
            "openai-grid-none",
            "openai-coords-none",
            "openai-cues-none",
        ],
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"default: $OPENAI_BASE_URL, else {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model", default=None, help=f"default: $OPENAI_MODEL, else {DEFAULT_MODEL}"
    )
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--fire-chance",
        type=float,
        default=None,
        help=f"chance a monster drops a bomb on any tick (default {FIRE_CHANCE}); "
        "raise it to put every arm under the same pressure",
    )
    parser.add_argument(
        "--clock",
        choices=["step", "realtime"],
        default="step",
        help="`step` waits for the agent and is the deterministic default: a "
        "comparison is about decisions. `realtime` does not wait, which is what "
        "the zeos players need",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=0.2,
        help="seconds per world tick under --clock realtime",
    )
    parser.add_argument(
        "--actions-per-tick",
        type=int,
        default=None,
        help="cap actions landing in one tick; the stick's budget",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=0.2,
        help="seconds the random baseline spends 'deciding' under --clock "
        "realtime; a model's latency is however long its request takes",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="override the system prompt; by default each player "
        "builds its own from its view plus the shared rules",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="comparison directory (default: runs/<stamp>-compare)",
    )
    args = parser.parse_args(argv)
    args.base_url = args.base_url or default_base_url()
    args.model = args.model or default_model()

    # Checked before any episode starts, so a typo costs nothing.
    for kind in args.players:
        if kind.startswith("zeos-") and args.clock == "step":
            sys.exit(
                f"{kind} needs --clock realtime: the step clock waits for the "
                "agent, so nothing is ever late and there is nothing to "
                "preempt -- which is the whole thing these players demonstrate."
            )

    # Built once, exactly as `agent` builds it: one game for every episode.
    rules = rules_of(args)

    root = args.out or Path("runs") / f"{stamp()}-compare"
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(
        json.dumps(
            meta_for(
                root.name,
                {
                    "kind": "compare",
                    "clock": "step",
                    "players": list(args.players),
                    "seeds": args.seeds,
                    "model": args.model,
                    "base_url": args.base_url,
                    "history": args.history,
                    "max_steps": args.max_steps,
                },
            ),
            indent=2,
        )
        + "\n"
    )

    def one(kind, seed):
        started = time.monotonic()
        scheduled = kind.startswith("zeos-")
        # The machine needs the effort too: both arms have to ask the endpoint
        # for the same thing.
        view, effort = settings(kind)
        player = build(kind, seed, args, rules)
        driver = (
            ZeosDriver(
                machine=build_machine(
                    kind.removeprefix("zeos-"),
                    # The machine issues the request on this arm, so it needs
                    # the endpoint too, or the table would call a difference of
                    # endpoint scheduling.
                    model=args.model,
                    base_url=args.base_url,
                    effort=effort,
                ),
                view=player.view,
                rules=rules,
            )
            if scheduled
            else None
        )
        episode = RunWriter(
            root / "episodes" / f"{kind}-seed{seed}",
            {
                "player": kind,
                "clock": args.clock,
                "model": getattr(player, "model", None),
                "base_url": args.base_url if kind != "random" else None,
                "view": view,
                "effort": effort,
                "seed": seed,
                "history": None if kind == "random" else args.history,
                "max_steps": args.max_steps,
                "tick_seconds": args.tick if args.clock == "realtime" else None,
                "actions_per_tick": args.actions_per_tick,
                # The whole game, flat, exactly as `agent` records it.
                **rules.fields(),
                # `zeos debug` needs the descriptor tree to draw the wiring
                # around this episode's `kernel.jsonl`.
                "case": case_path() if scheduled else None,
                "prompt": getattr(player, "prompt", None),
            },
        )
        try:
            with episode:
                summary = _play(player, driver, seed, args, episode, rules)
                summary["wall"] = round(time.monotonic() - started, 1)
                summary["per_decision"] = (
                    round(summary["wall"] / summary["decisions"], 2)
                    if summary["decisions"]
                    else 0
                )
                episode.finish(summary)
        finally:
            # The driver owns a thread pool with a pass possibly still in it.
            if driver is not None:
                driver.close()
        print(
            f"  {kind} seed {seed}: {summary['outcome']} score {summary['score']} "
            f"in {summary['decisions']} steps ({summary['per_decision']}s/decision)",
            flush=True,
        )
        # Read back rather than returned: `row()` is settings and verdict
        # joined, so a row of `table.json` says which player and seed it was.
        return RunReader(episode.path).row()

    # One episode at a time: concurrent seeds would share the endpoint, and
    # under `--clock realtime` the latency is the measurement.
    table = {
        kind: [one(kind, seed) for seed in range(args.seeds)] for kind in args.players
    }

    print(
        f"\n{'player':<26}{'score':>14}{'steps':>8}{'wins':>7}"
        f"{'aimed':>8}{'unparsed':>10}{'s/dec':>8}"
    )
    for kind, rows in table.items():
        scores = [r["score"] for r in rows]
        shots = aimed(
            RunReader(root / "episodes" / f"{kind}-seed{seed}")
            for seed in range(len(rows))
        )
        print(
            f"{kind:<26}{st.mean(scores):>8.1f} ±{st.pstdev(scores):>4.1f}"
            f"{st.mean(r['decisions'] for r in rows):>8.0f}"
            f"{sum(r['outcome'] == 'won' for r in rows):>4}/{len(rows)}"
            f"{shots:>7.0f}%"
            f"{sum(r['unparseable'] for r in rows):>10}"
            f"{st.mean(r['per_decision'] for r in rows):>8.2f}"
        )

    (root / "table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {root}")


if __name__ == "__main__":
    main()
