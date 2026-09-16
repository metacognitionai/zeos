# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``run_all`` runs one case through every seat and keeps a journal of each.

The claude and qwen seats need a key and 2.7GB of weights, so only the scripted
seat is run here. The environment a seat is launched with is the other half of
the script and is asserted directly: the claude seat's key reaches the SDK only
through ``os.environ``, and a ``.env`` must never override a shell that exports
one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from run_all import environment, main


def test_env_file_reaches_the_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-the-file\n# a comment\n\n")
    assert environment(env_file)["ANTHROPIC_API_KEY"] == "from-the-file"


def test_an_exported_variable_wins(tmp_path: Path, monkeypatch) -> None:  # pyright: ignore[reportMissingParameterType]
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-the-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    assert environment(env_file)["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_a_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    assert environment(tmp_path / "absent") == dict(os.environ)


def test_the_scripted_seat_writes_its_journal(tmp_path: Path) -> None:
    assert main(["--seats", "scripted", "--runs", str(tmp_path)]) == 0
    journal = tmp_path / "simple_count-scripted.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [e["kind"] for e in events].count("job.completed") == 2
