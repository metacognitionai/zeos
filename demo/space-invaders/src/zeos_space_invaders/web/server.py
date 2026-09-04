# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Serve `runs/` as a page, or write one run out as a file.

All I/O stops here: `payload.py` is a pure function of a directory and the page
a pure function of the payload, so both are tested without a socket. Served,
`__DATA__` is `null` and the page fetches what it needs; exported, the data is
inlined and the file needs no server, no network and no external asset.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from ..runlog import RunReader
from . import payload

STATIC = Path(__file__).resolve().parent / "static"

#: Where the data goes. Deliberately not the name of the global it feeds
#: (`window.__RUN__`), so it appears in the shell exactly once.
PLACEHOLDER = "__DATA__"

#: The page's own two assets, always inlined.
ASSETS = (("/*CSS*/", "viewer.css"), ("/*JS*/", "viewer.js"))

#: The vendored table library, inlined only for the served page: an exported
#: run has no sortable table, and does not want half a megabyte to draw one.
VENDOR = (
    ("/*VENDOR_CSS*/", "vendor/tabulator/tabulator.min.css"),
    ("/*VENDOR_JS*/", "vendor/tabulator/tabulator.min.js"),
)


def page(data=None, vendor=True):
    """The single page, with the stylesheets, the scripts and the data inlined.

    The scripts are substituted in before the data, so a placeholder that also
    appeared in one of them would have the JSON spliced into an expression.
    `vendor=False` leaves the table library out, and nothing else changes.
    """
    built = (STATIC / "index.html").read_text("utf-8")
    for marker, name in ASSETS:
        built = _inline(built, marker, (STATIC / name).read_text("utf-8"))
    for marker, name in VENDOR:
        text = (STATIC / name).read_text("utf-8") if vendor else ""
        built = _inline(built, marker, text)
    found = built.count(PLACEHOLDER)
    if found != 1:
        # Counted before substituting: a second occurrence gets the JSON spliced
        # into it and fails to parse in the browser rather than here.
        raise ValueError(
            f"{PLACEHOLDER} appears {found} times in the assembled page, not once"
            " -- the script must not use the placeholder's own name"
        )
    return built.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))


def _inline(built, marker, text):
    """Substitute one asset, insisting its marker is there exactly once.

    Checked per asset because the markers are substituted in order: an asset
    containing a later marker would otherwise have that asset spliced into it
    silently.
    """
    found = built.count(marker)
    if found != 1:
        raise ValueError(
            f"{marker} appears {found} times in the page being assembled, not "
            "once -- no asset may contain another asset's marker"
        )
    return built.replace(marker, text)


def export(run, out):
    """One self-contained file for one run. Opens from a filesystem, offline."""
    reader = RunReader(run)
    data = {"route": "run", "path": reader.path.name, "run": payload.episode(reader)}
    Path(out).write_text(page(data, vendor=False), "utf-8")
    return out


def under(root, relative):
    """Resolve a request path inside `root`, or None if it points outside it.

    Checked rather than trusted: this serves whatever the URL names, so without
    it `../../..` reads any file the user can.
    """
    root = Path(root).resolve()
    try:
        target = (root / unquote(relative)).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        return None
    return target if (target / "meta.json").is_file() else None


class Handler(BaseHTTPRequestHandler):
    root = Path("runs")

    def do_GET(self):
        # The query string is not part of the route: a `?` reaching `partition`
        # would turn the page itself into a 404.
        path, _, _query = self.path.partition("?")
        route, _, rest = path.lstrip("/").partition("/")
        if route in ("", "index.html"):
            return self._send(page().encode(), "text/html; charset=utf-8")
        if route != "api":
            return self.send_error(404)

        kind, _, relative = rest.partition("/")
        if kind == "index":
            return self._json(payload.index(self.root))
        target = under(self.root, relative) if relative else None
        if target is None:
            return self.send_error(404, "no such run")
        try:
            reader = RunReader(target)
        except ValueError as exc:
            return self.send_error(409, str(exc))
        if kind == "compare":
            return self._json(payload.comparison(reader))
        if kind == "run":
            return self._json(payload.episode(reader))
        return self.send_error(404)

    def _json(self, obj):
        self._send(json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is reassembled from `static/` on every request and a run
        # directory grows while it is watched, so nothing here may be cached.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return  # the page is the output; access logs are noise


#: Not 8000: `zeos debug` defaults to it, and the two are opened on the same run.
DEFAULT_PORT = 8123


def serve(root, port=DEFAULT_PORT, host="127.0.0.1"):
    """A bound server; the caller runs it.

    The root travels on a subclass rather than on `Handler` itself, so two
    servers in one process do not overwrite each other's directory.
    """
    handler = type("RunHandler", (Handler,), {"root": Path(root)})
    return ThreadingHTTPServer((host, port), handler)
