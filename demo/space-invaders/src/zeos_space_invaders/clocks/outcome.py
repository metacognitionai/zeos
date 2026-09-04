# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The half of a summary that is the same whatever clock produced it.

What is *not* here is deliberate: `unparseable`, where `usage` comes from and
the word for a run cut short each mean something different on each runner, so
each runner still says those itself.
"""


def usage_of(records):
    """Every integer in every decision's usage, summed by key.

    Integers only: a server may report a nested breakdown beside the totals, and
    adding a dict to a dict is not a sum.
    """
    total = {}
    for decision in records:
        for key, value in decision.usage.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
    return total


def outcome_of(game, records, elapsed, *, over):
    """The keys every clock answers the same way.

    `over` is this runner's word for a run cut short rather than finished --
    `timeout` off a wall clock, `truncated` off a step budget -- because a
    reader of a table has to be able to tell the two apart.
    """
    decisions = len(records)
    return {
        # A guard cutting the run short is not a loss.
        "outcome": "won" if game.won else ("lost" if game.over else over),
        "score": game.score,
        "lives": game.lives,
        # A hit while standing in the bomb's column is a dodge not made; a hit
        # while stepping into it is a move chosen against a board that had
        # moved on.
        "hits_standing": game.hits_standing,
        "hits_walking_in": game.hits_walking_in,
        "monsters_left": len(game.monsters),
        "ticks": game.ticks,
        "decisions": decisions,
        "actions_per_tick": round(decisions / game.ticks, 3) if game.ticks else 0.0,
        "dropped_after_game_over": sum(1 for d in records if not d.applied),
        "seconds": round(elapsed, 1),
    }


#: Sonnet 5 rates, USD per million tokens.
PRICE_IN, PRICE_OUT = 2.00, 10.00


def announce(summary, *, clock, player=None, path=None):
    """Print a finished run the same way whoever produced it.

    `player` and `path` are optional because a human has no `--player` and may
    be playing with `--no-record`.
    """
    usage = summary["usage"]
    print(
        f"\n{summary['outcome']}  |  score {summary['score']}  "
        f"|  {summary['monsters_left']} monsters left  |  lives {summary['lives']}"
    )
    # Keyed on what the summary carries, not on the clock: a human has no
    # latency to report, and the step clock nothing to say about lateness.
    if "mean_latency" in summary:
        print(
            f"{summary['ticks']} ticks in {summary['seconds']}s | "
            f"{summary['decisions']} decisions "
            f"({summary['actions_per_tick']} per tick) | "
            f"mean latency {summary['mean_latency']}s "
            f"= {summary['mean_ticks_waited']} ticks per move"
        )
        print(
            f"{summary['unparseable']} unparseable replies, "
            f"{summary['dropped_after_game_over']} actions arrived too late"
        )
    elif clock == "step":
        print(
            f"{summary['decisions']} steps in {summary['seconds']}s, "
            f"{summary['unparseable']} unparseable replies"
        )
    else:
        print(
            f"{summary['ticks']} ticks in {summary['seconds']}s | "
            f"{summary['decisions']} actions "
            f"({summary['actions_per_tick']} per tick)"
        )
    if "preemptions" in summary:
        # `preemptions` is the kernel's own count, off `kernel.jsonl`; the rest
        # is our bookkeeping.
        print(
            f"{summary['pilot_moves']} pilot moves, "
            f"{summary['reflexes']} reflexes, "
            f"{summary['preemptions']} kernel preemptions"
        )
        print(
            f"{summary['generations']} completions "
            f"({summary['cancellations']} cancelled by a context change, "
            f"{summary['voided']} elements voided), "
            f"{summary['decoded_words']} decoded, "
            f"{summary['thinking_words']} thinking"
        )
    if usage:
        print(
            "tokens: "
            + ", ".join(
                f"{k} {v}" for k, v in sorted(usage.items()) if isinstance(v, int)
            )
        )
    # Both Claude arms: the scheduled one bills the same API.
    if usage and player and player.removeprefix("zeos-") == "claude":
        cost = (
            usage.get("input_tokens", 0) * PRICE_IN
            + usage.get("output_tokens", 0) * PRICE_OUT
        ) / 1_000_000
        print(
            f"~${cost:.2f} at base rates "
            "(cache reads bill lower, so the real figure is under this)"
        )
    if path is not None:
        print(f"summary written to {path / 'summary.json'}")
