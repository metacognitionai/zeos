# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A device adapter for the keyboard, reading keys without blocking so the run loop keeps going."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["Console", "available"]


def available() -> bool:
    """Whether there is a terminal to read, which is false under pytest, pipes and CI."""
    return sys.stdin.isatty()


@contextmanager
def _cbreak() -> Iterator[None]:
    """Deliver keystrokes as they are typed, and always put the terminal back on the way out."""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class Console:
    """Keystrokes in, complete inputs out, with two states: idle, and collecting a number."""

    def __init__(self) -> None:
        self._digits: str | None = None

    @property
    def collecting(self) -> bool:
        return self._digits is not None

    def begin_number(self) -> None:
        self._digits = ""

    def poll(self) -> tuple[str | None, str | None]:
        """Read whatever is waiting and return ``(event, number)``, never blocking."""
        event: str | None = None
        number: str | None = None
        while select.select([sys.stdin], [], [], 0)[0]:
            char = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="ignore")
            if not char:
                break
            if self._digits is None:
                if char == " ":
                    event = "interrupt"
                elif char in ("q", "\x03"):
                    event = "quit"
                continue
            if char in ("\r", "\n"):
                if not self._digits:
                    # Enter on an empty prompt is a slip, so keep collecting rather than give up.
                    continue
                number, self._digits = self._digits, None
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif char.isdigit():
                self._digits += char
                sys.stdout.write(char)
                sys.stdout.flush()
            elif char in ("\x7f", "\b") and self._digits:
                self._digits = self._digits[:-1]
                sys.stdout.write("\b \b")
                sys.stdout.flush()
        return event, number

    @staticmethod
    @contextmanager
    def attached() -> Iterator["Console | None"]:
        """Give back a console if there is a terminal, and nothing at all if there is not."""
        if not available():
            yield None
            return
        with _cbreak():
            yield Console()
