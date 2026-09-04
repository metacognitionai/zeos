# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Open a run directory and append to it, one writer per episode.

`meta.json` is written the moment the directory is created, so a run that dies
halfway still says what it was, and `summary.json` appearing is the signal that
a run finished. Appends are locked because the realtime clock writes frames from
its own thread while the main loop writes decisions, and the interleaving *is*
the data.
"""

from __future__ import annotations

import functools
import json
import subprocess
import threading
import time
from pathlib import Path

from .schema import SCHEMA_VERSION, Frame

RUNS = Path("runs")


@functools.cache
def git_commit(cwd=None):
    """The checkout a run was made from, or None outside a git tree.

    `cwd` is for the tests, and the answer is cached because `compare` opens one
    writer per episode: twenty episodes would otherwise be forty-two `git`
    processes before the first tick.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if head.returncode:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        ).stdout.strip()
        return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def meta_for(name, config=None):
    """What a run says about itself; `compare` writes the same header for the
    directory holding its episodes."""
    return {
        "schema": SCHEMA_VERSION,
        "run": name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": git_commit(),
        **(config or {}),
    }


def stamp():
    return time.strftime("%Y%m%d-%H%M%S")


class RunWriter:
    """A run directory, opened once and appended to from any thread."""

    def __init__(self, path, config=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self.meta = meta_for(self.path.name, config)
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        self._events = (self.path / "events.jsonl").open("w")
        self._kernel = None

    @classmethod
    def create(cls, name, config=None, root=RUNS):
        """A fresh, timestamped directory under `root`. Never reuses one."""
        return cls(Path(root) / f"{stamp()}-{name}", config)

    # -- the event stream ----------------------------------------------------

    def frame(self, info, over=False):
        self._write(self._events, Frame.of(info, over=over).record())

    def decision(self, decision):
        if decision.tick is None:
            raise ValueError(
                f"decision by {decision.by!r} has no tick: the clock owns tick "
                "numbers and has to stamp one before this is written"
            )
        with self._lock:
            decision.seq = self._seq
            self._seq += 1
        self._write(self._events, decision.record())

    def kernel(self, tick, records):
        """One tick's worth of a zeos journal, opened lazily.

        `records` arrive in zeos's own wire format because this module imports
        nothing but the standard library; only the world tick is added, which
        zeos tolerates, so the file stays a journal its own tools can read.
        """
        if not records:
            return
        if self._kernel is None:
            self._kernel = (self.path / "kernel.jsonl").open("w")
        for record in records:
            self._write(self._kernel, {"tick": tick, **record})

    def _write(self, handle, record):
        with self._lock:
            if handle.closed:
                # The realtime clock is a daemon thread that may still be
                # ticking during a Ctrl-C teardown; dropping its last frame
                # beats a traceback on top of the interrupt.
                return
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    # -- finishing -----------------------------------------------------------

    def finish(self, summary):
        """Write the verdict; its presence is what marks a run as complete.

        The verdict only: `meta.json` beside it already says what was run, and
        `RunReader.row()` is where the two are put back together.
        """
        (self.path / "summary.json").write_text(
            json.dumps({"run": self.path.name, **summary}, indent=2) + "\n"
        )
        return summary

    def close(self):
        for handle in (self._events, self._kernel):
            if handle is not None and not handle.closed:
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
