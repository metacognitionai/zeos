# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""In-process transport: everything on one node.

The degenerate case, and the only one that ships in phase 1. Both ends of every
pipe live in the same ``PipeTable``, so delivery is what the table already did and
there is nothing to poll.

Its value is not what it does but where it sits: every pipe is resolved through a
transport, so adding a federated one is an addition rather than a change to the
kernel, the descriptor format, or any job's semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from zeos.core.ids import PipeName
from zeos.core.pipes import PipeTable
from zeos.machine.base import Token
from zeos.transport.base import InboundFrame

__all__ = ["LocalTransport"]


class LocalTransport:
    """Carries every pipe, in-process."""

    def __init__(self, pipes: PipeTable) -> None:
        self._pipes = pipes

    @property
    def name(self) -> str:
        return "local"

    def carries(self, pipe: PipeName) -> bool:
        # Phase 1 has one node, so the local transport carries everything. A
        # federated setup would consult the topology here.
        _ = pipe
        return True

    def deliver(self, pipe: PipeName, tokens: Sequence[Token]) -> int:
        return self._pipes.ensure(pipe).write(tokens)

    def poll(self) -> Iterable[InboundFrame]:
        # Nothing arrives asynchronously when both ends are in this process; local
        # writes are already visible in the table by the time they return.
        return ()
