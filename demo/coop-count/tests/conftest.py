# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def model_path() -> Path:
    from zeos_coop_count import model as model_mod

    return Path(os.environ.get("ZEOS_COUNT_MODEL", ROOT / "models" / model_mod.DEFAULT_FILE))


@pytest.fixture(scope="session")
def llama_model():  # pyright: ignore[reportMissingParameterType]
    path = model_path()
    if not path.is_file():
        pytest.skip(f"no model at {path}; run 'zeos-count fetch-model'")
    from zeos_coop_count.machine import LlamaModel

    model = LlamaModel(path)
    yield model
    model.free()


@pytest.fixture
def machines() -> Iterator[list]:  # pyright: ignore[reportMissingTypeArgument]
    """Tracks every machine a test builds so they are all closed at the end."""
    built: list = []
    yield built
    for m in built:
        close = getattr(m, "close", None)
        if close is not None:
            close()
