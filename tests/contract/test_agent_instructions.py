# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Policy-as-test: the agent instructions are one file, and it arrives on every platform.

``CLAUDE.md`` was a symlink to ``AGENTS.md``. That works where symlinks do and fails
silently where they do not: git-for-windows checks a symlink out as a plain file holding
the link target unless ``core.symlinks`` is on, which it is not by default, so a whole
repository's agent instructions became the nine bytes ``AGENTS.md``. Nothing complained,
because nothing was broken in any way a test could see -- the rules simply were not
there.

It also corrupted in one direction. The text hooks treat the degraded file as prose and
append a trailing newline, so a commit from Windows changes the link target to
``AGENTS.md\\n`` -- dangling for everyone on a platform that has symlinks, and invisible
to the author who did it.

So the file is an ordinary one that imports the other. Between them the two checks below
catch a reintroduced symlink on every platform: ``is_symlink`` where symlinks exist, and
the content where they do not.
"""

from __future__ import annotations

from pathlib import Path

import zeos

REPO = Path(zeos.__file__).resolve().parents[2]
AGENTS = REPO / "AGENTS.md"
CLAUDE = REPO / "CLAUDE.md"

#: The whole of CLAUDE.md. An import, so the instructions have one home.
IMPORT = "@AGENTS.md"


def test_the_instructions_live_in_agents_md() -> None:
    assert AGENTS.is_file()
    assert "## Hard rules" in AGENTS.read_text("utf-8")


def test_claude_md_is_not_a_symlink() -> None:
    assert CLAUDE.is_file(), "CLAUDE.md is missing"
    assert not CLAUDE.is_symlink(), (
        "CLAUDE.md is a symlink again. Windows checks that out as a plain file holding "
        "the link target, so the instructions never reach an agent working there."
    )


def test_claude_md_imports_rather_than_copies() -> None:
    """A copy would drift, and drifted instructions are worse than absent ones."""
    assert CLAUDE.read_text("utf-8").strip() == IMPORT
