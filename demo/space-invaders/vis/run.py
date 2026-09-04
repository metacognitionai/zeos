# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Play a game and draw it, from the settings in `settings.json`.

    python demo/space-invaders/vis/run.py              # play, then draw
    python demo/space-invaders/vis/run.py --live        # watch it play, no GIF
    python demo/space-invaders/vis/run.py --draw-only   # draw what is already there
    python demo/space-invaders/vis/run.py --set play.view=grid
    python demo/space-invaders/vis/run.py other-settings.json

`board` and `play` become flags to the demo's `agent`; `draw` becomes flags to
`make_gif.py`. A setting of `null` is left off, so the demo's own default (or
`.env`) decides it. Playing costs API calls; drawing costs nothing, which is why
it can be asked for on its own.

`--live` draws the board in the terminal as it is played and stops there, which
is the way to try a setting out: change `settings.json`, watch, Ctrl-C when it
looks wrong. The run is still written, so a game worth keeping can be drawn
afterwards with `--draw-only`.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
ROOT = DEMO.parents[1]


def flags(settings):
    """`{"monster_rows": 2}` -> `["--monster-rows", "2"]`, dropping nulls."""
    out = []
    for key, value in settings.items():
        if value is None:
            continue
        out += [f"--{key.replace('_', '-')}", str(value)]
    return out


def override(settings, assignment):
    """Apply one `section.key=value` to the settings that were read.

    The value is read as JSON so numbers stay numbers and `null` stays null,
    falling back to the bare string for the common case of a word.
    """
    path, _, raw = assignment.partition("=")
    keys = path.split(".")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    target = settings
    for key in keys[:-1]:
        target = target[key]
    if keys[-1] not in target:
        sys.exit(f"{path}: no such setting")
    target[keys[-1]] = value


def run(command, cwd):
    print("$", " ".join(str(part) for part in command), flush=True)
    try:
        result = subprocess.run(command, cwd=cwd)
    except KeyboardInterrupt:
        sys.exit(130)
    if result.returncode:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", nargs="?", default=HERE / "settings.json")
    parser.add_argument(
        "--live",
        action="store_true",
        help="draw the board in the terminal as it plays, and make no GIF",
    )
    parser.add_argument(
        "--draw-only",
        action="store_true",
        help="redraw the run the settings name, without playing a new one",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="section.key=value",
        help="change one setting for this run only, leaving settings.json alone",
    )
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    for assignment in args.set:
        override(settings, assignment)

    run_dir = HERE / settings["run"]
    if not args.draw_only:
        run(
            [
                "uv",
                "run",
                "agent",
                *flags(settings["play"]),
                *flags(settings["board"]),
                "--render" if args.live else "--no-render",
                "--out",
                str(run_dir),
            ],
            cwd=DEMO,
        )
    if args.live:
        return
    run(
        [
            "uv",
            "run",
            "--with",
            "pillow",
            "python",
            str(HERE / "make_gif.py"),
            str(run_dir),
            str(HERE / settings["gif"]),
            *flags(settings["draw"]),
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
