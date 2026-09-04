# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Pipe transports -- the distribution seam.

This interface is the whole of phase 1's distribution work, and it is deliberately
small. The claim of the distribution design is that pipes are already the right seam:
they are the only inter-job communication mechanism, and the kernel already chooses
transport per pipe (zero-copy over shared KV when producer and consumer share a
model, text copy otherwise -- core §4.2). "Same node vs. different node" is one more
transport choice behind an interface that already had to exist.

So there is nothing clever here. What matters is the two constraints it enforces:

1. **A job cannot observe which transport carries its pipe.** Pipe operations block
   identically whether the peer is in-process or across a link, so federation
   changes latency, not semantics.
2. **Pipes are addressed by name.** Never by object reference and never by node
   address, so placement can change without touching a descriptor.

Only ``LocalTransport`` ships in phase 1. When a real transport lands, two things
follow from the specs and should be encoded here rather than rediscovered:
zero-copy is impossible across a link (it needs a shared KV materialisation, so
cross-node pipes always fall back to text copy), and inbound traffic must carry its
ring and principal so that provenance survives the hop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from zeos.core.ids import PipeName
from zeos.machine.base import Token

__all__ = ["PipeTransport", "InboundFrame"]


class InboundFrame(Protocol):
    """Tokens arriving from a peer, with the provenance they arrived under."""

    @property
    def pipe(self) -> PipeName: ...

    @property
    def tokens(self) -> tuple[Token, ...]: ...


@runtime_checkable
class PipeTransport(Protocol):
    """How a named pipe reaches its far end."""

    @property
    def name(self) -> str:
        """Transport identifier, journalled on ``PipeCreated``."""
        ...

    def carries(self, pipe: PipeName) -> bool:
        """Whether this transport is responsible for the named pipe."""
        ...

    def deliver(self, pipe: PipeName, tokens: Sequence[Token]) -> int:
        """Push tokens toward the far end. Returns the count accepted.

        A short return is backpressure, not an error -- the same contract as a local
        pipe, which is what keeps rate matching working across a link.
        """
        ...

    def poll(self) -> Iterable[InboundFrame]:
        """Tokens that have arrived from the far end since the last poll.

        Called by the driver, never by the kernel: polling is I/O, and the kernel
        does no I/O. The kernel learns about arrivals only as pipe writes.
        """
        ...
