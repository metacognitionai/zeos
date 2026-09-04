# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Namespaced world state, and the read/write sets defined over it.

This is what makes selective resume invalidation possible. Descriptors declare the
world state they depend on (``reads``) and may change (``writes``); the kernel also
accumulates what a job actually touched, and the union of declared and observed is
used at resume time (core §3).

Pure and stdlib-only: the kernel core consults this during ``step()``, so it must
not perform I/O or read a clock. Persistence and device adapters live on the driver
side and reach the store only by applying writes.

**Values are strings.** Deliberately, not lazily: threshold evaluation belongs to
device adapters (a gas sensor crossing 25 ppm becomes a pipe write, not a numeric
comparison in the kernel), and what the kernel needs the value *for* -- diffing,
RESUME text, status-region rendering -- is exactly the string a model reads.

**Patterns.** A read/write-set entry is either an exact object name
(``plant.unit_a``) or a namespace wildcard (``plant.*``), which matches any object
under that prefix. Full globbing is deliberately not supported: read/write sets are
a load-time analysis surface, and richer patterns would make
"do these two descriptors conflict?" undecidable at load time for no real gain.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.clock import Clock, format_duration
from zeos.core.events import StateDelta
from zeos.core.ids import JobId, ObjectName

__all__ = ["ObjectSet", "Write", "WorldStore"]

_WILDCARD_SUFFIX = ".*"


@dataclass(frozen=True, slots=True)
class ObjectSet:
    """A set of world-state objects, named exactly or by namespace wildcard."""

    patterns: frozenset[str] = frozenset()

    @staticmethod
    def of(patterns: Iterable[str]) -> ObjectSet:
        cleaned: set[str] = set()
        for raw in patterns:
            pattern = raw.strip()
            if not pattern:
                continue
            if pattern.count("*") > 1 or (
                "*" in pattern and not pattern.endswith(_WILDCARD_SUFFIX)
            ):
                raise ValueError(
                    f"unsupported pattern {raw!r}: only exact names and "
                    "'namespace.*' wildcards are allowed"
                )
            cleaned.add(pattern)
        return ObjectSet(frozenset(cleaned))

    def matches(self, obj: ObjectName) -> bool:
        for pattern in self.patterns:
            if pattern.endswith(_WILDCARD_SUFFIX):
                if obj.startswith(pattern[: -len("*")]):
                    return True
            elif obj == pattern:
                return True
        return False

    def filter(self, objects: Iterable[ObjectName]) -> tuple[ObjectName, ...]:
        return tuple(o for o in objects if self.matches(o))

    def intersects(self, other: ObjectSet) -> bool:
        """Whether the two sets could ever name the same object.

        Conservative for wildcards -- ``plant.*`` and ``plant.unit_a`` intersect --
        which is the right bias for a load-time conflict check: a false positive is
        a review conversation, a false negative is an unflagged race.
        """
        for a in self.patterns:
            for b in other.patterns:
                a_wild, b_wild = a.endswith(_WILDCARD_SUFFIX), b.endswith(_WILDCARD_SUFFIX)
                a_stem, b_stem = a[: -len("*")], b[: -len("*")]
                if a_wild and b_wild:
                    if a_stem.startswith(b_stem) or b_stem.startswith(a_stem):
                        return True
                elif a_wild:
                    if b.startswith(a_stem):
                        return True
                elif b_wild:
                    if a.startswith(b_stem):
                        return True
                elif a == b:
                    return True
        return False

    def __or__(self, other: ObjectSet) -> ObjectSet:
        return ObjectSet(self.patterns | other.patterns)

    def __bool__(self) -> bool:
        return bool(self.patterns)

    def render(self) -> str:
        return ", ".join(sorted(self.patterns)) or "(none)"


@dataclass(frozen=True, slots=True)
class Write:
    """One recorded change. The history of these is what a resume diff is built from."""

    obj: ObjectName
    before: str
    after: str
    at: Clock
    by: JobId | None
    #: Why this write matters when the value alone does not say so. A write with a
    #: note is never idempotent, however equal the values.
    note: str = ""


@dataclass
class WorldStore:
    """Current values plus an append-only write history.

    History is kept rather than only current values because a resumed job needs to
    know *what changed while it was gone*, and the before-value at suspension time
    is not recoverable from the current state alone.
    """

    # Parameterised factories rather than bare ``dict``/``list``: the bare forms
    # infer as ``dict[Unknown, Unknown]`` under strict checking.
    values: dict[ObjectName, str] = field(default_factory=dict[ObjectName, str])
    history: list[Write] = field(default_factory=list["Write"])
    #: When each object last arrived from its authoritative node, for objects this
    #: node holds only a *replica* of. Absent for objects this
    #: node owns, which is the distinction that matters: a replica has an age, and
    #: authoritative state does not.
    synced_at: dict[ObjectName, Clock] = field(default_factory=dict[ObjectName, Clock])

    def get(self, obj: ObjectName, default: str = "") -> str:
        return self.values.get(obj, default)

    def has(self, obj: ObjectName) -> bool:
        return obj in self.values

    def objects(self) -> tuple[ObjectName, ...]:
        return tuple(sorted(self.values))

    def set(
        self,
        obj: ObjectName,
        value: str,
        *,
        at: Clock,
        by: JobId | None = None,
        note: str = "",
    ) -> Write | None:
        """Apply a write. Returns the record, or ``None`` if nothing material changed.

        Idempotent writes are dropped rather than recorded, so that a sensor
        adapter republishing a steady value does not inflate every resumed job's
        dirty set with changes that did not happen.

        **Unless the write carries a note.** Some changes are not expressible in the
        value: a gripper recalibrated to different offsets is still ``gripper-std``,
        and a job resuming in a new body needs to know. A noted
        write is material by definition, and the note travels into the resume diff.
        """
        before = self.values.get(obj, "")
        if before == value and not note:
            return None
        self.values[obj] = value
        record = Write(obj=obj, before=before, after=value, at=at, by=by, note=note)
        self.history.append(record)
        return record

    def mark_synced(self, obj: ObjectName, *, at: Clock) -> None:
        """Record that a replica of ``obj`` is fresh as of ``at``."""
        self.synced_at[obj] = at

    def age_ns(self, obj: ObjectName, *, now: Clock) -> int | None:
        """How stale this node's copy of ``obj`` is, in virtual nanoseconds.

        ``None`` for anything this node has never received as a replica -- which
        includes everything it is authoritative for. That is the honest answer: an
        object you own has no age, it has a value.
        """
        synced = self.synced_at.get(obj)
        if synced is None:
            return None
        return max(0, now.virtual_ns - synced.virtual_ns)

    def render_with_age(self, objects: Sequence[ObjectName], *, now: Clock) -> str:
        """Render for injection, labelling replicas with their age.

        > A job reading ``robot.position`` from a replica should see its staleness,
        > because a plan that would be safe against fresh state may not be safe
        > against state that is 400 ms old.

        The label is the whole requirement. Whether models actually plan more
        conservatively against state they are told is stale is **OQ-D4**, and
        unanswerable until a real model reads one of these.
        """
        lines: list[str] = []
        for obj in objects:
            age = self.age_ns(obj, now=now)
            suffix = "" if age is None else f"  (replica, {format_duration(age)} old)"
            lines.append(f"{obj}: {self.get(obj)}{suffix}")
        return "\n".join(lines)

    def writes_since(self, since: Clock) -> tuple[Write, ...]:
        """Writes recorded at or after ``since`` (by token clock)."""
        return tuple(w for w in self.history if w.at.token_clock >= since.token_clock)

    def written_objects_since(self, since: Clock) -> ObjectSet:
        return ObjectSet(frozenset(str(w.obj) for w in self.writes_since(since)))

    def dirty_for(
        self, read_set: ObjectSet, since: Clock, *, exclude_job: JobId | None = None
    ) -> tuple[StateDelta, ...]:
        """The RESUME diff: what this job depends on that changed while it was
        suspended (core §6.2).

        Collapses multiple writes to the same object into one delta spanning the
        whole suspension -- the model needs to know where the value *ended up* versus
        where it was, not the intermediate steps, and intermediate steps would bury
        the salient change.

        ``exclude_job`` drops the job's own writes: a job is not surprised by its
        own effects, and including them would train it to distrust its own actions.
        """
        first_before: dict[ObjectName, str] = {}
        last_after: dict[ObjectName, str] = {}
        notes: dict[ObjectName, list[str]] = {}
        for write in self.writes_since(since):
            if exclude_job is not None and write.by == exclude_job:
                continue
            if not read_set.matches(write.obj):
                continue
            first_before.setdefault(write.obj, write.before)
            last_after[write.obj] = write.after
            if write.note:
                notes.setdefault(write.obj, []).append(write.note)

        return tuple(
            StateDelta(
                obj=obj,
                before=first_before[obj],
                after=last_after[obj],
                note="; ".join(notes.get(obj, ())),
            )
            for obj in sorted(last_after)
            # A value that ended where it started is a round trip, and not worth
            # revalidating -- *unless* a note says the sameness is misleading.
            if first_before[obj] != last_after[obj] or notes.get(obj)
        )

    def snapshot(self) -> Mapping[ObjectName, str]:
        return dict(self.values)

    def render(self, objects: Sequence[ObjectName]) -> str:
        """Render objects for injection into a status region."""
        return "\n".join(f"{obj}: {self.get(obj, '(unset)')}" for obj in objects)
