# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assembling the page, and serving it. All I/O stops here.

``payload.py`` is a pure function of a loaded case and a journal, and the page is a
pure function of the payload -- which is what lets both be tested without a socket,
and what would let a live run be served later by changing where the records come
from and nothing else.

The page is one shell with three placeholders. Served, ``__DATA__`` is ``null`` and
the page asks for what it needs; exported, the data is inlined and the file needs no
server, no network and no external asset. A recorded run that needs a running
service to be read is not a thing anyone can attach to an issue.

Unlike the general-purpose viewer this pattern comes from, this server publishes
exactly one case and one journal, both fixed when it was constructed. There is no
user-controlled path, so there is no path to traverse out of.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

__all__ = ["ASSETS", "PLACEHOLDER", "export", "page", "serve"]

STATIC = Path(__file__).resolve().parent / "static"

#: Where the data goes. Deliberately not the name of the global it feeds
#: (``window.__ZEOS__``), so it appears in the assembled page exactly once -- the
#: scripts are substituted in before the data is, and a script mentioning the
#: placeholder's own name would have the JSON spliced into the middle of an
#: expression.
PLACEHOLDER = "__DATA__"

#: The page's own assets, always inlined. There are no others: no CDN, no vendored
#: library, nothing the exported file could fail to find offline.
ASSETS = (("/*CSS*/", "debugger.css"), ("/*JS*/", "debugger.js"))


def _inline(built: str, marker: str, text: str) -> str:
    """Substitute one asset, insisting its marker is there exactly once.

    Checked per asset rather than once at the end, because the markers are
    substituted in order: an asset that happened to contain a later marker would
    otherwise have that asset spliced into it silently.
    """
    found = built.count(marker)
    if found != 1:
        raise ValueError(
            f"{marker} appears {found} times in the page being assembled, not once "
            "-- no asset may contain another asset's marker"
        )
    return built.replace(marker, text)


def page(data: Any = None) -> str:
    """The single page, with the stylesheet, the script and the data inlined."""
    built = (STATIC / "index.html").read_text("utf-8")
    for marker, name in ASSETS:
        built = _inline(built, marker, (STATIC / name).read_text("utf-8"))
    # Counted before substituting rather than checked after: a second occurrence
    # does not survive, it gets the JSON spliced into it, and the page then fails to
    # parse in a browser rather than failing to build here.
    found = built.count(PLACEHOLDER)
    if found != 1:
        raise ValueError(
            f"{PLACEHOLDER} appears {found} times in the assembled page, not once "
            "-- no asset may use the placeholder's own name"
        )
    return built.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))


def export(payload: Any, out: Path) -> Path:
    """One self-contained file. Opens from a filesystem, offline, forever."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" for the same reason the journal writer pins it: a file whose bytes
    # differ by platform is a file two people cannot compare.
    out.write_text(page(payload), encoding="utf-8", newline="\n")
    return out


class _Base(BaseHTTPRequestHandler):
    """Response plumbing. What is served is decided by the subclass ``serve`` makes."""

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Nothing here is worth caching and the page is reassembled from static/ on
        # every request, so without this a reload after editing the CSS shows the
        # page from before the edit.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # The parameter keeps the base class's name, shadowing a builtin, because an
        # override that renames it is a different method as far as a type checker is
        # concerned.
        return  # the page is the output; access logs are noise


def serve(
    payload_fn: Callable[[], Any],
    *,
    port: int = 8000,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """A bound server; the caller runs it.

    The handler closes over ``payload_fn`` rather than carrying it on a shared class
    attribute, so two servers in one process cannot overwrite each other's case. The
    function is called per request rather than once, so re-running a case and
    reloading shows the new run.
    """

    class Handler(_Base):
        def do_GET(self) -> None:
            # The query string is not part of the route: anyone appending one to
            # force a reload should still get the page rather than a 404.
            path, _, _query = self.path.partition("?")
            route = path.strip("/")
            if route in ("", "index.html"):
                self._send(page().encode("utf-8"), "text/html; charset=utf-8")
            elif route == "api/payload":
                body = json.dumps(payload_fn(), separators=(",", ":")).encode("utf-8")
                self._send(body, "application/json")
            else:
                self.send_error(404)

    return ThreadingHTTPServer((host, port), Handler)
