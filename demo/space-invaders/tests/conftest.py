# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures every test area shares. The stubs themselves are in `stubs.py`."""

import pytest

from zeos_space_invaders.runlog import RunReader, RunWriter


@pytest.fixture
def run(tmp_path):
    """An open run directory, closed when the test ends."""
    with RunWriter(tmp_path / "run") as writer:
        yield writer


@pytest.fixture
def written(tmp_path):
    """Read back whatever the `run` fixture wrote."""
    return lambda: RunReader(tmp_path / "run")


@pytest.fixture(autouse=True)
def elsewhere(tmp_path, monkeypatch):
    """Run every test somewhere disposable: `RunWriter.create` writes to `runs/`
    relative to the working directory, and tests that drive `agent` without
    `--out` would otherwise litter the developer's own."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Keep the developer's own `.env` out of every test: `cli` and `compare` call
    `load_env()` on the way in."""
    monkeypatch.setattr("zeos_space_invaders.utils.config.find_env_file", lambda: None)
