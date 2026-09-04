# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assembling a ZEOS kernel around the llama backend."""

from __future__ import annotations

from collections.abc import Sequence

from zeos.core.events import Event
from zeos.core.ids import ObjectName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.loader import CaseBundle
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineBackend
from zeos.transport.base import PipeTransport
from zeos.transport.local import LocalTransport
from zeos.world.store import WorldStore

from zeos_coop_count.machine import DEFAULT_BLOCK_SIZE, LlamaMachine, LlamaModel

__all__ = ["bound_aliases", "build_kernel", "seat_maps", "valued_aliases"]


def bound_aliases(descriptor: Descriptor) -> tuple[str, ...]:
    """Which pipe aliases this descriptor binds, which is all the grammar lets a job name."""
    bindings = descriptor.pipes
    named = [a for a in ("stdin", "stdout", "tools") if bindings.resolve(a) is not None]
    return tuple(named + sorted(bindings.extra))


def valued_aliases(descriptor: Descriptor, pipes: Sequence[PipeSpec]) -> tuple[str, ...]:
    """Aliases bound to an actuator, a pipe whose writes carry a value and become world state."""
    backed = {spec.name for spec in pipes if spec.world_object}
    return tuple(a for a in bound_aliases(descriptor) if descriptor.pipes.resolve(a) in backed)


def seat_maps(bundle: CaseBundle) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """For each descriptor, the aliases it binds and which of those carry a value."""
    return (
        {str(name): bound_aliases(d) for name, d in bundle.descriptors.items()},
        {str(name): valued_aliases(d, bundle.pipes) for name, d in bundle.descriptors.items()},
    )


def build_kernel(
    bundle: CaseBundle,
    model: LlamaModel | None = None,
    *,
    machine: MachineBackend | None = None,
    journal_sink: list[Event] | None = None,
    config: KernelConfig | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_ctx: int = 8192,
    n_threads: int = 8,
    **machine_kwargs: object,
) -> tuple[Kernel, PipeTransport, MachineBackend]:
    """Build a kernel, given either a ``model`` to wrap in a seat or a seat built elsewhere."""
    if (model is None) == (machine is None):
        raise ValueError("pass exactly one of model= (build the llama seat) or machine=")
    descriptors, valued = seat_maps(bundle)
    if machine is None:
        assert model is not None
        machine = LlamaMachine(
            model,
            descriptors=descriptors,
            valued=valued,
            block_size=block_size,
            n_ctx=n_ctx,
            n_seq_max=max(8, len(bundle.descriptors) + 2),
            n_threads=n_threads,
            # This case has no compartments, so every mask narrowing here comes from paging.
            enforce_mask=False,
            **machine_kwargs,  # pyright: ignore[reportArgumentType]
        )
    pipes = PipeTable(bundle.pipes)
    world = WorldStore()
    kernel = Kernel(
        descriptors=bundle.descriptors,
        machine=machine,
        pipes=pipes,
        vectors=VectorTable(bundle.vectors),
        world=world,
        resources=ResourceTable(bundle.resources),
        platforms=bundle.platforms,
        principals=bundle.principals,
        gates=bundle.gates,
        journal_sink=journal_sink,
        config=config or KernelConfig(case=bundle.name),
    )
    for obj, value in sorted(bundle.world.items()):
        world.set(ObjectName(obj), value, at=kernel.clock)
    return kernel, LocalTransport(pipes), machine
