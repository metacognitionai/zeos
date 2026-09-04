# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A run directory, projected into what the page draws.

Everything here is a pure function of what is on disk: the page is drawn from
these return values, so a test on them is a test of the page's content.
`by_tick` is keyed by where an action landed, not where it was chosen, and
`in_flight` covers the ticks in between, which have a frame and no decision.
"""

from __future__ import annotations

import statistics as st
from itertools import pairwise

from zeos.journal.codec import decode_record
from zeos.monitor.state import Monitor, render_event

from ..runlog import RunReader, aimed

#: Kernel events one tick may contribute to the ticker; a tick that spends
#: thirty token boundaries would otherwise push the board off the screen.
EVENT_LIMIT = 24

#: Jobs shown at once. The reflex spawns one job per threat, and the live ones
#: are the ones that matter.
JOB_LIMIT = 8


# --- the index ---------------------------------------------------------------


def index(root):
    """Every run under `root`, newest first.

    Sorted by name, which sorts by time: a run directory is stamped
    `YYYYmmdd-HHMMSS-...` precisely so this needs no file dates.
    """
    rows = []
    for path in sorted(root.iterdir(), reverse=True):
        if not (path / "meta.json").is_file():
            continue
        try:
            reader = RunReader(path)
        except ValueError as exc:
            # A run from an older layout must not take the page down with it.
            rows.append({"path": path.name, "run": path.name, "unreadable": str(exc)})
            continue
        rows.append(_index_row(reader, path.name))
    return rows


def _index_row(reader, name):
    row = {**reader.row(), "path": name}
    row.update(_tokens(row.pop("usage", None)))
    # `row()` merges the cap in `meta` with the rate in `summary`; read the
    # measurement on purpose, so an unfinished run leaves the cell empty.
    row["actions_per_tick"] = (reader.summary or {}).get("actions_per_tick")
    row["criteria"] = _criteria(row.pop("criteria", None))
    for switch in ("stream",):
        row[switch] = _switch(row.get(switch))
    if reader.meta.get("kind") == "compare":
        # A comparison has no summary of its own -- its verdict is the table.
        row["is_compare"] = True
        row["finished"] = (reader.path / "table.json").is_file()
        row["episodes"] = [_episode_row(episode, name) for episode in reader.episodes()]
        row.update(_rollup(row["episodes"], reader.meta))
    else:
        row["is_compare"] = False
        row["finished"] = reader.summary is not None
    return row


def _episode_row(episode, name):
    """One episode of a comparison, as a line of the same table.

    It answers "did it finish" off its own summary, because the question is
    asked of every line and a line that cannot answer reads as unfinished.
    """
    row = {**episode.row(), "path": f"{name}/episodes/{episode.path.name}"}
    row.update(_tokens(row.pop("usage", None)))
    for switch in ("stream",):
        row[switch] = _switch(row.get(switch))
    row["criteria"] = _criteria(row.pop("criteria", None))
    row["is_compare"] = False
    row["finished"] = episode.summary is not None
    return row


def _criteria(verdicts):
    """ "passed/total" for the problem's own criteria, or None when there are none.

    Derived here because `viewer.js` holds no arithmetic and has no tests; the
    list itself stays in the summary, for the run screen to say which failed.
    """
    if not verdicts:
        return None
    return f"{sum(1 for v in verdicts if v.get('passed'))}/{len(verdicts)}"


def _tokens(usage):
    """Token counts as two sortable numbers rather than a dict.

    `total_tokens` when the vendor sent one, otherwise the sum of what it did
    send: the two SDKs name these fields differently, and summing everything
    would double-count a total. None rather than 0 for nothing spent, so a run
    that reported no usage sorts to the end instead of tying with a free one.
    """
    numbers = {k: v for k, v in (usage or {}).items() if isinstance(v, int | float)}
    total = numbers.get("total_tokens")
    if total is None:
        total = sum(v for k, v in numbers.items() if not k.startswith("total"))
    return {"tokens": total or None, "output_tokens": numbers.get("output_tokens")}


def _switch(value):
    """A run's on/off setting as something a column can be filtered by.

    Two words get the list filter the other enumerations have, where a boolean
    would filter as a number. `None` stays `None`: an empty cell says the
    setting does not apply, where "off" would claim someone turned it off.
    """
    return None if value is None else ("on" if value else "off")


def _rollup(episodes, meta):
    """A comparison's own line, averaged over the episodes under it.

    Real numbers because a spanning cell has nothing to sort by; `is_compare`
    still tells the page to draw the row as a heading. `view`, `effort` and
    `seed` are left alone: a comparison has no single one of any of them.
    """
    scored = [e for e in episodes if e.get("score") is not None]
    ticked = [e for e in episodes if e.get("ticks") is not None]
    return {
        "player": ", ".join(meta.get("players") or []) or None,
        "score": round(st.mean(e["score"] for e in scored), 1) if scored else None,
        "ticks": round(st.mean(e["ticks"] for e in ticked)) if ticked else None,
        "wins": sum(e.get("outcome") == "won" for e in episodes),
        "episode_count": len(episodes),
    }


# --- a comparison ------------------------------------------------------------


def comparison(reader):
    """The table `compare` prints, recomputed from the episodes themselves.

    `table.json` is a convenience, not the source: the episodes are plain run
    directories and the numbers come back out of them.
    """
    episodes = list(reader.episodes())
    rows = [
        # `aimed` here and not on the index: it reads every decision and frame,
        # and the index opens nothing but `meta.json` and `summary.json`.
        {**_episode_row(episode, reader.path.name), "aimed": round(aimed([episode]), 1)}
        for episode in episodes
    ]
    return {
        "meta": reader.meta,
        "episodes": rows,
        "players": _players(episodes, rows),
    }


def _players(episodes, rows):
    """One line per player, over the seeds it played."""
    order, grouped = [], {}
    for episode, row in zip(episodes, rows, strict=True):
        player = row.get("player") or "?"
        if player not in grouped:
            order.append(player)
            grouped[player] = []
        grouped[player].append((episode, row))

    lines = []
    for player in order:
        pairs = grouped[player]
        scores = [row.get("score", 0) for _, row in pairs]
        lines.append(
            {
                "player": player,
                "episodes": len(pairs),
                "score": round(st.mean(scores), 1),
                "score_sd": round(st.pstdev(scores), 1),
                "decisions": round(
                    st.mean(row.get("decisions", 0) for _, row in pairs)
                ),
                "wins": sum(row.get("outcome") == "won" for _, row in pairs),
                "aimed": round(aimed(episode for episode, _ in pairs)),
                "unparseable": sum(row.get("unparseable", 0) for _, row in pairs),
                "per_decision": round(
                    st.mean(row.get("per_decision", 0) for _, row in pairs), 2
                ),
                # Tokens rather than `mean_latency`: a comparison runs the step
                # clock, which does not record it, and tokens say what a
                # scoreline cost.
                "tokens": sum(row.get("tokens") or 0 for _, row in pairs) or None,
            }
        )
    return lines


# --- one episode -------------------------------------------------------------


def episode(reader):
    """Everything the playback screen draws, indexed by world tick."""
    meta = {k: v for k, v in reader.meta.items() if k != "prompt"}
    frames = list(reader.frames())
    decisions = list(reader.decisions())
    span = frames[-1]["tick"] + 1 if frames else 0

    landed, waiting = _join(decisions, span)
    kernel, events = _kernel_lane(reader, span)
    return {
        "meta": meta,
        "prompt": reader.meta.get("prompt"),
        "summary": reader.summary,
        "frames": frames,
        "decisions": decisions,
        "by_tick": landed,
        "in_flight": waiting,
        "kernel": kernel,
        "kernel_events": events,
        "marks": _marks(frames, decisions, span),
    }


def _join(decisions, span):
    """Decisions by the tick they landed on, and the ticks spent waiting."""
    landed = [[] for _ in range(span)]
    waiting = [None] * span
    for index_, decision in enumerate(decisions):
        asked = decision["tick"]
        applied = decision.get("tick_applied", asked)
        if 0 <= applied < span:
            landed[applied].append(index_)
        for tick in range(max(asked, 0), min(applied, span)):
            waiting[tick] = index_
    return landed, waiting


# --- the marks a scrubber can jump to ----------------------------------------

#: What earns a mark, most interesting first -- one per tick, so a tick where a
#: reflex saved a life is labelled by the life, not by the reflex.
_MARKS = ("life lost", "reflex", "kill", "unreadable reply", "pass abandoned", "over")


def _marks(frames, decisions, span):
    found = {}

    def mark(tick, label):
        # The sentinel is past the end of the table: a tick with nothing on it
        # yet has to lose to every label, including the last.
        held = _MARKS.index(found[tick]) if tick in found else len(_MARKS)
        if 0 <= tick < span and _MARKS.index(label) < held:
            found[tick] = label

    for before, after in pairwise(frames):
        if after["lives"] < before["lives"]:
            mark(after["tick"], "life lost")
        elif len(after["monsters"]) < len(before["monsters"]):
            mark(after["tick"], "kill")
    for decision in decisions:
        tick = decision.get("tick_applied", decision["tick"])
        if decision["by"] == "evade":
            mark(tick, "reflex")
        if not decision.get("parsed", True):
            mark(tick, "unreadable reply")
        if decision.get("preempted"):
            mark(tick, "pass abandoned")
    if frames and frames[-1]["over"]:
        mark(frames[-1]["tick"], "over")
    return [{"tick": tick, "label": found[tick]} for tick in sorted(found)]


# --- the kernel lane ---------------------------------------------------------


def _kernel_lane(reader, span):
    """One `SystemView` per world tick, plus that tick's events, rendered.

    Per world tick rather than per token boundary, and only the fields the pane
    draws: the whole `SystemView` is kilobytes a tick, and this is 200 bytes at
    the resolution the scrubber actually moves in.
    """
    records = list(reader.kernel())
    if not records or not span:
        return None, None

    monitor, events = Monitor(), [[] for _ in range(span)]
    views, current, dropped = {}, None, [0] * span
    for record in records:
        tick = record.get("tick", 0)
        if current is not None and tick != current:
            views[current] = monitor.snapshot()
        current = tick
        seq, event = decode_record({k: v for k, v in record.items() if k != "tick"})
        monitor.apply(event, seq=seq)
        if 0 <= tick < span:
            if len(events[tick]) < EVENT_LIMIT:
                events[tick].append(
                    {"seq": seq, "kind": record["kind"], "text": render_event(event)}
                )
            else:
                dropped[tick] += 1
    if current is not None:
        views[current] = monitor.snapshot()

    for tick, extra in enumerate(dropped):
        if extra:
            # Said rather than silently truncated: a ticker that stops at
            # twenty-four looks like a tick that only did twenty-four things.
            events[tick].append({"seq": None, "kind": "…", "text": f"+{extra} more"})

    lane, last = [], None
    for tick in range(span):
        # Carried forward: a tick the kernel had nothing to do in still has the
        # state the previous tick left behind.
        last = _view(tick, views[tick]) if tick in views else last
        lane.append({**last, "tick": tick} if last else None)
    return lane, events


def _view(tick, view):
    alive = [job for job in view.jobs if job.alive]
    done = [job for job in reversed(view.jobs) if not job.alive]
    return {
        "tick": tick,
        "running": str(view.running) if view.running is not None else None,
        "jobs": [_job(job) for job in alive + done[: max(0, JOB_LIMIT - len(alive))]],
        "tokens": view.counters.tokens,
        "passes": view.counters.forward_passes,
        "preemptions": view.counters.preemptions,
        "last": view.last_kind,
    }


def _job(job):
    return {
        "job": str(job.job),
        "name": str(job.descriptor),
        "state": job.state.value,
        "priority": job.priority,
        "blocked_on": str(job.blocked_on) if job.blocked_on else None,
        "tokens": job.tokens,
        "alive": job.alive,
    }
