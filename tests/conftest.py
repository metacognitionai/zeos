# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Suite-wide pytest hooks.

``--trace-decode`` turns on ``KernelConfig.trace_decode`` for every kernel a test
builds, so what each job decodes at every tick streams to the terminal. Pair it
with ``-s``, or pytest captures the output.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from zeos.core.kernel import Kernel, KernelConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--trace-decode",
        action="store_true",
        default=False,
        help="print every Decoded event to the terminal as it happens (use with -s)",
    )


@pytest.fixture(autouse=True)
def trace_decode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if not request.config.getoption("--trace-decode"):
        return
    original = Kernel.__init__

    def traced(self: Kernel, *args: Any, **kwargs: Any) -> None:
        config: KernelConfig = kwargs.get("config") or KernelConfig()
        kwargs["config"] = replace(config, trace_decode=True)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Kernel, "__init__", traced)
