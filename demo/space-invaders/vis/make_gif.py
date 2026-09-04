# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Draw a run directory as an animated GIF.

    uv run --with pillow python make_gif.py runs/<id> out.gif

Reads the two files a run writes -- `events.jsonl` for the world and
`kernel.jsonl` for what the kernel was doing -- and paints one frame per world
tick: the board, the score, the two jobs, and a timeline of the episode with a
playhead on it. It never imports the game, so it can draw a run someone sent
you.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"

BG = (13, 17, 23)
PANEL = (22, 27, 34)
GRID = (30, 38, 48)
INK = (201, 209, 217)
DIM = (110, 122, 136)
GREEN = (126, 231, 135)
BLUE = (88, 166, 255)
RED = (255, 123, 114)
AMBER = (233, 196, 106)
WHITE = (240, 246, 252)

CELL = 32
PAD = 20
HEADER = 44
PANEL_W = 336
#: The panel is at least this tall, and matches the board when the board is
#: taller. What it says is anchored to the two ends rather than stretched, so a
#: tall board leaves a gap in the middle rather than a column of empty box.
PANEL_H = 240
LANE_H = 74

INVADER = (
    "..X..X..",
    ".XXXXXX.",
    "XX.XX.XX",
    "XXXXXXXX",
    ".X.XX.X.",
    "X......X",
    ".X....X.",
    "........",
)
SHIP = (
    "........",
    "...XX...",
    "...XX...",
    "..XXXX..",
    ".XXXXXX.",
    ".XXXXXX.",
    "XXXXXXXX",
    "........",
)
BOMB = (
    "...XX...",
    "..XX....",
    "...XX...",
    "....XX..",
    "...XX...",
    "..XX....",
    "...XX...",
    "........",
)


def read_run(path):
    """The world, the decisions and the kernel, all keyed by world tick."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    frames, decisions = {}, {}
    for line in (path / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "frame":
            frames[event["tick"]] = event
        else:
            decisions[event["tick_applied"]] = event
    kernel = []
    journal = path / "kernel.jsonl"
    if journal.is_file():
        kernel = [json.loads(line) for line in journal.read_text().splitlines()]
    return meta, frames, decisions, kernel


def kernel_lanes(kernel, ticks):
    """Which job held the machine on each tick, and where it was taken away.

    The journal names a job by number and only its `job.spawned` says which
    descriptor that number is, so the two are joined here before the state
    changes are folded forward: a job stays running until an event says it
    stopped, and ticks with no event inherit the tick before.
    """
    names, running, events = {}, {}, {}
    for record in kernel:
        tick, kind = record["tick"], record["kind"]
        if kind == "job.spawned":
            names[record["job"]] = record["descriptor"]
        elif kind == "job.state":
            job = names.get(record["job"], str(record["job"]))
            running.setdefault(tick, {})[job] = record["to_state"]
        elif kind in ("job.preempted", "vector.fired"):
            events.setdefault(tick, []).append(kind)
    lanes, state = {}, {}
    for tick in ticks:
        state = dict(state)
        state.update(running.get(tick, {}))
        lanes[tick] = (dict(state), events.get(tick, []))
    return lanes


def sprite(draw, pattern, x, y, colour, size=CELL):
    """One 8x8 bitmap, scaled to fill a board cell."""
    unit = size / 8
    for row, line in enumerate(pattern):
        for col, mark in enumerate(line):
            if mark == "X":
                left, top = x + col * unit, y + row * unit
                draw.rectangle(
                    (left, top, left + unit - 1, top + unit - 1), fill=colour
                )


def draw_board(draw, frame, meta, x, y, cell=CELL):
    w, h = meta["width"], meta["height"]
    draw.rectangle((x - 6, y - 6, x + w * cell + 5, y + h * cell + 5), fill=PANEL)
    for col in range(w + 1):
        draw.line((x + col * cell, y, x + col * cell, y + h * cell), fill=GRID)
    for row in range(h + 1):
        draw.line((x, y + row * cell, x + w * cell, y + row * cell), fill=GRID)
    for row, col in frame["monsters"].values():
        sprite(draw, INVADER, x + col * cell, y + row * cell, GREEN, cell)
    for row, col in frame["dangers"]:
        sprite(draw, BOMB, x + col * cell, y + row * cell, RED, cell)
    if frame["missile"]:
        row, col = frame["missile"]
        left = x + col * cell + cell / 2 - cell / 16
        top = y + row * cell + cell / 5
        draw.rectangle((left, top, left + cell / 8, top + cell * 0.6), fill=WHITE)
    sprite(draw, SHIP, x + frame["player"] * cell, y + (h - 1) * cell, BLUE, cell)


def draw_panel(
    draw, x, y, width, height, frame, decision, jobs, fonts, tally, meta, scheduled
):
    small, bold, tiny = fonts
    draw.rectangle((x, y, x + width, y + height), fill=PANEL)
    line = y + 14

    draw.text((x + 16, line), "SCORE", font=tiny, fill=DIM)
    draw.text((x + 100, line), f"{frame['score']:>4}", font=bold, fill=WHITE)
    draw.text((x + 180, line), "LIVES", font=tiny, fill=DIM)
    for life in range(meta["lives"]):
        colour = RED if life < frame["lives"] else GRID
        left = x + 250 + life * 22
        draw.ellipse((left, line + 1, left + 13, line + 14), fill=colour)
    line += 34

    draw.line((x + 16, line, x + width - 16, line), fill=GRID)
    line += 12
    draw.text((x + 16, line), "KERNEL" if scheduled else "PLAYER", font=tiny, fill=DIM)
    line += 22
    if scheduled:
        for name, priority, note in (("evade", 5, "pinned"), ("pilot", 60, "")):
            state = jobs.get(name, "idle")
            colour = {"running": RED if name == "evade" else BLUE}.get(
                state, AMBER if state == "suspended" else DIM
            )
            draw.ellipse((x + 16, line + 4, x + 26, line + 14), fill=colour)
            draw.text((x + 36, line), f"{name} @{priority}", font=small, fill=INK)
            draw.text((x + 170, line), note, font=tiny, fill=DIM)
            draw.text((x + 240, line), state, font=small, fill=colour)
            line += 24
    else:
        draw.ellipse((x + 16, line + 4, x + 26, line + 14), fill=BLUE)
        draw.text((x + 36, line), meta["player"], font=small, fill=INK)
        draw.text((x + 170, line), "prompt loop", font=tiny, fill=DIM)
        line += 24
        draw.text((x + 16, line), "no reflex, no scheduler", font=tiny, fill=DIM)
        line += 24

    line += 8
    draw.line((x + 16, line, x + width - 16, line), fill=GRID)
    line += 12
    if frame["over"]:
        if frame["won"]:
            verdict = "cleared the board"
        elif frame["lives"] <= 0:
            verdict = "no lives left"
        else:
            verdict = "the block reached the ship"
        draw.text((x + 16, line), "GAME OVER", font=bold, fill=AMBER)
        line += 22
        draw.text((x + 16, line), verdict, font=small, fill=DIM)
    elif decision:
        who = decision["by"]
        colour = RED if who == "evade" else BLUE
        draw.text(
            (x + 16, line), f"{who} -> {decision['action']}", font=bold, fill=colour
        )
    else:
        draw.text((x + 16, line), "thinking...", font=small, fill=DIM)
    line += 26
    draw.text(
        (x + 16, line),
        f"moves {tally['pilot'] + tally['model']}   dodges {tally['evade']}   "
        f"preempted {tally['preempted']}"
        if scheduled
        else f"moves {tally['pilot'] + tally['model']}",
        font=tiny,
        fill=DIM,
    )

    monsters = meta["monster_rows"] * meta["monster_cols"]
    settings = (
        # A run with no model behind it -- `random`, or a human at the keyboard --
        # records `model: null`, so the player names it instead.
        meta["model"] or meta["player"],
        f"view {meta['view']}   effort {meta['effort'] or '-'}   seed {meta['seed']}",
        f"board {meta['width']}x{meta['height']}   {monsters} monsters",
        f"tick {meta['tick_seconds']}s   fire {meta['fire_chance']}",
    )
    line = y + height - 20 - 18 * len(settings)
    draw.line((x + 16, line - 12, x + width - 16, line - 12), fill=GRID)
    for text in settings:
        draw.text((x + 16, line), text, font=tiny, fill=DIM)
        line += 18


def draw_timeline(draw, x, y, width, ticks, lanes, decisions, now, fonts, lanes_shown):
    _, _, tiny = fonts
    draw.rectangle((x, y, x + width, y + LANE_H), fill=PANEL)
    span = max(len(ticks) - 1, 1)
    step = (width - 40) / span
    for row, (name, colour) in enumerate(lanes_shown):
        top = y + 16 + row * 22
        draw.text((x + 12, top), name, font=tiny, fill=DIM)
        draw.line((x + 70, top + 7, x + width - 12, top + 7), fill=GRID)
        for tick in ticks:
            jobs, _ = lanes.get(tick, ({}, []))
            if jobs.get(name) != "running":
                continue
            left = x + 70 + (tick / span) * (width - 90)
            draw.rectangle((left, top + 3, left + max(step, 2), top + 11), fill=colour)
        for tick, decision in decisions.items():
            if decision["by"] != name:
                continue
            left = x + 70 + (tick / span) * (width - 90)
            draw.ellipse((left - 3, top + 2, left + 5, top + 12), fill=WHITE)
    playhead = x + 70 + (now / span) * (width - 90)
    draw.line((playhead, y + 8, playhead, y + LANE_H - 8), fill=AMBER)


def render(run, out, scale=1, duration=200, tail=1200, cell=CELL):
    meta, frames, decisions, kernel = read_run(run)
    ticks = sorted(frames)
    lanes = kernel_lanes(kernel, ticks)
    fonts = (
        ImageFont.truetype(FONT, 15),
        ImageFont.truetype(FONT_BOLD, 17),
        ImageFont.truetype(FONT, 12),
    )
    title = ImageFont.truetype(FONT_BOLD, 18)

    board_w = meta["width"] * cell
    board_h = meta["height"] * cell
    width = PAD + board_w + 24 + PANEL_W + PAD
    height = HEADER + board_h + 16 + LANE_H + PAD

    scheduled = bool(kernel)
    tally = {"pilot": 0, "evade": 0, "model": 0, "preempted": 0}
    images = []
    for tick in ticks:
        frame = frames[tick]
        decision = decisions.get(tick)
        jobs, marks = lanes.get(tick, ({}, []))
        if decision:
            tally[decision["by"]] = tally.get(decision["by"], 0) + 1
        if "job.preempted" in marks:
            tally["preempted"] += 1

        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        draw.text((PAD, 14), "ZEOS", font=title, fill=GREEN)
        draw.text((PAD + 62, 16), "space invaders", font=fonts[0], fill=DIM)
        draw.text(
            (width - PAD - 130, 16),
            f"tick {tick:>3}/{ticks[-1]}",
            font=fonts[0],
            fill=DIM,
        )
        draw_board(draw, frame, meta, PAD, HEADER, cell)
        draw_panel(
            draw,
            PAD + board_w + 24,
            HEADER - 6,
            PANEL_W,
            max(PANEL_H, board_h + 12),
            frame,
            decision,
            jobs,
            fonts,
            tally,
            meta,
            scheduled,
        )
        draw_timeline(
            draw,
            PAD,
            HEADER + board_h + 16,
            width - 2 * PAD,
            ticks,
            lanes,
            decisions,
            tick,
            fonts,
            (("evade", RED), ("pilot", BLUE)) if scheduled else (("model", BLUE),),
        )
        if "job.preempted" in marks:
            draw.rectangle((0, 0, width - 1, height - 1), outline=RED, width=3)
            draw.text((PAD + board_w + 24, 16), "PREEMPTED", font=fonts[1], fill=RED)
        if scale != 1:
            image = image.resize((width * scale, height * scale), Image.NEAREST)
        images.append(image)

    delays = [duration] * len(images)
    delays[-1] = tail
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=delays,
        loop=0,
        optimize=True,
    )
    print(f"{out}  {len(images)} frames  {Path(out).stat().st_size / 1024:.0f} kB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--duration", type=int, default=200, help="ms per world tick")
    parser.add_argument(
        "--cell", type=int, default=CELL, help="pixels per board square"
    )
    args = parser.parse_args()
    render(
        args.run,
        args.out,
        scale=args.scale,
        duration=args.duration,
        cell=args.cell,
    )


if __name__ == "__main__":
    main()
