# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The move machine with the local Claude Code CLI in the model seat.

See ``move_machine`` for the protocol; this file is only the asking.
``claude_machine`` calls the Anthropic API directly, which needs its own
``ANTHROPIC_API_KEY``. This instead shells out to the ``claude`` binary already
installed and logged in on this machine, one ``claude --print`` process per move --
so it rides whatever session that login already carries (a Pro/Max subscription,
if that is how ``claude login`` was set up) rather than opening a second,
separately billed API account. If ``ANTHROPIC_API_KEY`` is set in the environment,
the CLI uses that instead of the login session, same as it would interactively.
"""

from __future__ import annotations

import subprocess
import tempfile

from move_machine import SYSTEM, MoveMachine

from zeos.core.ids import JobId

__all__ = ["ClaudeCodeMachine"]


class ClaudeCodeMachine(MoveMachine):
    """``MoveMachine`` asking a local ``claude --print`` process for each move."""

    def __init__(self, *, model: str = "sonnet", block_size: int = 16) -> None:
        super().__init__(block_size=block_size)
        self._model = model
        # `claude --print` reads the CLAUDE.md of the directory it starts in and puts it
        # in the model's context, whatever `--system-prompt` says. A job's context is its
        # goal and its own moves, so the process is run from an empty directory instead
        # of from wherever the demo happened to be launched.
        self._nowhere = tempfile.TemporaryDirectory(prefix="zeos-counter-seat-")

    def _ask(self, job: JobId, prompt: str) -> str:
        # Each move is a fresh, stateless process: the prompt already carries the
        # job's whole transcript (see MoveMachine.decode), so there is nothing for
        # a resumed CLI session to add and every call is independent -- one beat
        # of the kernel's tick, not a running conversation.
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model",
                self._model,
                "--system-prompt",
                SYSTEM,
                # No tool this move ever needs: the system prompt asks for exactly
                # one line of text, never a file read or a shell command.
                "--allowedTools",
                "",
                "--output-format",
                "text",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=self._nowhere.name,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude --print exited {result.returncode} for job {job}: {result.stderr.strip()}"
            )
        return result.stdout.strip()
