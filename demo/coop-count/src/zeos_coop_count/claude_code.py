# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A command source that answers kernel syscalls with a local ``claude --print`` process.

Same source as ``claude.py``'s ``ClaudeSource`` -- same ABI, same prompt, same reply
cleanup -- except the one call that differs: this asks a local, already-authenticated
``claude`` binary instead of the Anthropic API client, one ``claude --print`` process per
command. That rides whatever session the machine's own ``claude login`` already carries
(a Pro/Max subscription, if that is how it was set up) instead of a second, separately
billed API account. If ``ANTHROPIC_API_KEY`` is set in the environment, the CLI uses that
instead of the login session, same as it would interactively.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping, Sequence

from zeos_coop_count.claude import one_command, prompt_for, system_for
from zeos_coop_count.seat import Turn
from zeos_coop_count.syscall import ALIASES

__all__ = ["ClaudeCodeSource", "DEFAULT_MODEL"]

DEFAULT_MODEL = "sonnet"

#: Same level, and for the same reason, as the API seat asks for.
EFFORT = "high"


class ClaudeCodeSource:
    """A command source that asks a local ``claude --print`` process for each command."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        valued: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._model = model
        # `claude --print` reads the CLAUDE.md of the directory it starts in and puts it
        # in the model's context, whatever `--system-prompt` says. A job's context is its
        # descriptor and what arrived on its pipes, so the process is run from an empty
        # directory instead of from wherever the demo happened to be launched.
        self._nowhere = tempfile.TemporaryDirectory(prefix="zeos-coop-count-seat-")
        self._aliases: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (descriptors or {}).items()
        }
        self._valued: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in (valued or {}).items()}

    def next_command(self, turn: Turn) -> str:
        """One ``claude --print`` call, giving one command."""
        system = system_for(
            self._aliases.get(turn.descriptor, ALIASES), self._valued.get(turn.descriptor, ())
        )
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model",
                self._model,
                "--effort",
                EFFORT,
                "--system-prompt",
                system,
                # No tool this seat's command set ever needs: `say`, `write`, `read`
                # and `exit` are all just words, never a file read or a shell command.
                "--allowedTools",
                "",
                "--output-format",
                "text",
            ],
            input=prompt_for(turn),
            capture_output=True,
            text=True,
            timeout=120,
            cwd=self._nowhere.name,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude --print exited {result.returncode} for job {turn.job}: "
                f"{result.stderr.strip()}"
            )
        return one_command(result.stdout.strip())
