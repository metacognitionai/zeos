# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Draw two runs side by side, boards only, played through once.

    uv run --with pillow python compare_gif.py \
        runs/zeos-claude-full-game runs/claude-prompt-loop out.gif

One frame per world tick of the longer run. A run that has already ended keeps
its last board, dimmed, so the difference you are looking at is one board still
moving while the other has stopped. The GIF carries no loop extension, so it
plays once and holds on the last frame.
"""

import argparse
from pathlib import Path

from make_gif import BG, CELL, DIM, GRID, INK, RED, draw_board, read_run
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
PAD = 20
GAP = 28
LABEL_H = 34
#: How much of the ground to mix into a board whose game is over.
FADE = 0.62


def render(runs, out, cell=CELL, duration=80, every=1):
    loaded = []
    for run in runs:
        meta, frames, _, _ = read_run(run)
        loaded.append((meta, frames, sorted(frames)))
    label = ImageFont.truetype(FONT_BOLD, 16)

    board_w = max(meta["width"] for meta, _, _ in loaded) * cell
    board_h = max(meta["height"] for meta, _, _ in loaded) * cell
    width = PAD * 2 + board_w * len(loaded) + GAP * (len(loaded) - 1)
    height = PAD + LABEL_H + board_h + PAD
    longest = max(len(ticks) for _, _, ticks in loaded)

    # Speed comes from showing fewer ticks, not from shorter frames: browsers
    # clamp a GIF delay under about 50ms and play it slower than asked.
    steps = list(range(0, longest, every))
    if steps[-1] != longest - 1:
        steps.append(longest - 1)

    images = []
    for step in steps:
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        for column, (meta, frames, ticks) in enumerate(loaded):
            x = PAD + column * (board_w + GAP)
            over = step >= len(ticks)
            frame = frames[ticks[min(step, len(ticks) - 1)]]

            draw.text(
                (x, PAD),
                meta["player"],
                font=label,
                fill=DIM if over else INK,
            )
            # Lives, right-aligned over the board they belong to: a spent one
            # stays in place as an empty dot, so the row reads as 1 of 3 rather
            # than as one lonely marker.
            for life in range(meta["lives"]):
                left = x + board_w - (meta["lives"] - life) * 20
                draw.ellipse(
                    (left, PAD + 5, left + 12, PAD + 17),
                    fill=RED if life < frame["lives"] else GRID,
                )
            draw_board(draw, frame, meta, x, PAD + LABEL_H, cell)
            if over:
                box = (
                    x - 6,
                    PAD + LABEL_H - 6,
                    x + board_w + 6,
                    PAD + LABEL_H + board_h + 6,
                )
                faded = Image.blend(
                    image.crop(box),
                    Image.new("RGB", (box[2] - box[0], box[3] - box[1]), BG),
                    FADE,
                )
                image.paste(faded, box)
        images.append(image)

    # No `loop` argument, so no Netscape loop extension: the animation runs once.
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=[duration] * (len(images) - 1) + [3000],
        optimize=True,
    )
    print(f"{out}  {len(images)} frames  {Path(out).stat().st_size / 1024:.0f} kB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("out")
    parser.add_argument("--cell", type=int, default=CELL)
    parser.add_argument("--duration", type=int, default=80, help="ms per drawn frame")
    parser.add_argument(
        "--every", type=int, default=1, help="draw one frame per N world ticks"
    )
    args = parser.parse_args()
    render(
        args.runs, args.out, cell=args.cell, duration=args.duration, every=args.every
    )


if __name__ == "__main__":
    main()
