# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Replay a run in a browser.

    viewer                                  # browse runs/ at http://127.0.0.1:8123
    viewer --root somewhere/else
    viewer --export runs/<id> run.html      # one self-contained file

The index lists every run and every comparison; a comparison opens as its table
and each of its episodes opens as a run. A run opens as a scrubber over the
world's ticks, with the board, the decision that landed on that tick, and -- for
a scheduled run -- what the kernel was doing at the same moment.

Read-only. Nothing here writes into a run directory.
"""

import argparse
import webbrowser
from pathlib import Path

from .web.server import DEFAULT_PORT, export, serve


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=Path("runs"), help="directory of runs"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--export",
        nargs=2,
        metavar=("RUN", "OUT"),
        help="write one run as a self-contained page instead of serving",
    )
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.export:
        run, out = args.export
        print(f"wrote {export(Path(run), Path(out))}")
        return

    server = serve(args.root, port=args.port)
    where = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"serving {args.root} at {where}\nctrl-c to stop")
    if not args.no_open:
        webbrowser.open(where)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
