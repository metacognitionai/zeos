# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Faults, and the notices the model reads when one fires.

Faults are interrupts targeting the faulting job's ``on_fault`` policy -- the same
mechanism as any other interrupt, so error handling is also just descriptors
(core §6.3). Nothing here dispatches; this module decides *what should happen* and
what the model should be told, and the kernel does it.

Two rules from the specs shape this module:

* **Fail loudly.** Every enforcement action that changes behaviour emits a fault or
  a log event with provenance attached. There is no silent
  degradation and no silent aging -- a starved job raises a visible fault rather
  than being quietly boosted.
* **The notice is part of the behaviour.** What the kernel writes into a job's
  context when it faults is read by the model, so the text is under test, not a
  presentation detail.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from zeos.core.ids import DescriptorName, FaultKind, JobId, OnFault, PipeName, SegmentId
from zeos.descriptor.schema import FaultPolicy

__all__ = ["Fault", "FaultAction", "FaultResolution", "resolve", "render_notice"]


@dataclass(frozen=True, slots=True)
class Fault:
    """A raised fault, with enough context to explain itself.

    ``detail`` is written for a human reading the journal *and* for the model
    reading the notice; a fault that says only "privilege fault" forces both to go
    digging, which is the failure mode this design exists to avoid.
    """

    kind: FaultKind
    job: JobId
    detail: str
    segment: SegmentId | None = None
    pipe: PipeName | None = None

    @property
    def is_hard(self) -> bool:
        """Hard faults hold regardless of model behaviour; soft ones are trainable
        and measured."""
        return self.kind is not FaultKind.X_VIOLATION


class FaultAction(enum.StrEnum):
    CONTINUE = "continue"  # notice injected; job keeps running
    RETRY = "retry"  # notice injected; the failed operation is retried
    ABORT = "abort"  # job terminates FAULTED
    DISPATCH_HANDLER = "dispatch-handler"  # hand off to a descriptor
    ESCALATE = "escalate"  # abort and inform the parent


@dataclass(frozen=True, slots=True)
class FaultResolution:
    action: FaultAction
    handler: DescriptorName | None = None
    notice: str = ""


def resolve(fault: Fault, policy: FaultPolicy) -> FaultResolution:
    """Map a fault plus the descriptor's declared policy onto what the kernel does."""
    notice = render_notice(fault)
    match policy.kind:
        case OnFault.ABORT:
            return FaultResolution(FaultAction.ABORT, notice=notice)
        case OnFault.RETRY:
            return FaultResolution(FaultAction.RETRY, notice=notice)
        case OnFault.HANDLER:
            return FaultResolution(
                FaultAction.DISPATCH_HANDLER, handler=policy.handler, notice=notice
            )
        case OnFault.ESCALATE:
            return FaultResolution(FaultAction.ESCALATE, notice=notice)


def render_notice(fault: Fault) -> str:
    """The ring-0 text injected into the faulting job's context.

     Framed with reserved control tokens so it cannot be forged by content
    . The framing is what carries authority; the body is prose.
    """
    parts = [f"<FAULT kind={fault.kind.value}>"]
    parts.append(fault.detail)
    if fault.segment is not None:
        parts.append(f"offending segment: {fault.segment}")
    if fault.pipe is not None:
        parts.append(f"offending pipe: {fault.pipe}")
    parts.append("</FAULT>")
    return " ".join(parts)
