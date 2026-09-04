# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""World state, read/write sets, and the resume diff built from them."""

from __future__ import annotations

import pytest

from zeos.core.clock import Clock
from zeos.core.ids import JobId, ObjectName
from zeos.world.store import ObjectSet, WorldStore

T0 = Clock(token_clock=0, virtual_ns=0)
T5 = Clock(token_clock=5, virtual_ns=5_000)
T9 = Clock(token_clock=9, virtual_ns=9_000)

JOB_A = JobId(1)
JOB_B = JobId(2)


def obj(name: str) -> ObjectName:
    return ObjectName(name)


# -- ObjectSet ---------------------------------------------------------------


def test_exact_and_wildcard_matching() -> None:
    s = ObjectSet.of(["plant.unit_a", "robot.*"])
    assert s.matches(obj("plant.unit_a"))
    assert not s.matches(obj("plant.fan_4"))
    assert s.matches(obj("robot.position"))
    assert s.matches(obj("robot.arm.pose"))
    assert not s.matches(obj("robotics.position"))


def test_unsupported_patterns_are_rejected_at_construction() -> None:
    """Read/write sets are a load-time analysis surface; richer patterns would make
    conflict detection undecidable for no real gain."""
    with pytest.raises(ValueError, match="unsupported pattern"):
        ObjectSet.of(["plant.*.fan"])
    with pytest.raises(ValueError, match="unsupported pattern"):
        ObjectSet.of(["*"])


def test_blank_patterns_are_ignored() -> None:
    assert ObjectSet.of(["", "  ", "a.b"]).patterns == frozenset({"a.b"})


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (["robot.position"], ["robot.position"], True),
        (["robot.position"], ["robot.velocity"], False),
        (["robot.*"], ["robot.position"], True),
        (["robot.position"], ["robot.*"], True),
        (["robot.*"], ["robot.arm.*"], True),
        (["robot.*"], ["plant.*"], False),
    ],
)
def test_intersects_is_conservative(left: list[str], right: list[str], expected: bool) -> None:
    assert ObjectSet.of(left).intersects(ObjectSet.of(right)) is expected


def test_union() -> None:
    combined = ObjectSet.of(["a.b"]) | ObjectSet.of(["c.*"])
    assert combined.matches(obj("a.b")) and combined.matches(obj("c.d"))


# -- WorldStore --------------------------------------------------------------


def test_set_records_a_write() -> None:
    store = WorldStore()
    write = store.set(obj("plant.unit_a"), "running", at=T0, by=JOB_A)
    assert write is not None
    assert (write.before, write.after) == ("", "running")
    assert store.get(obj("plant.unit_a")) == "running"


def test_idempotent_write_is_dropped() -> None:
    """A sensor republishing a steady value must not inflate every resumed job's
    dirty set with changes that did not happen."""
    store = WorldStore()
    store.set(obj("plant.unit_a"), "running", at=T0)
    assert store.set(obj("plant.unit_a"), "running", at=T5) is None
    assert len(store.history) == 1


def test_writes_since_filters_by_token_clock() -> None:
    store = WorldStore()
    store.set(obj("a.x"), "1", at=T0)
    store.set(obj("a.y"), "2", at=T9)
    assert [w.obj for w in store.writes_since(T5)] == [obj("a.y")]


def test_dirty_for_collapses_intermediate_writes() -> None:
    """The model needs where the value ended up versus where it was; intermediate
    steps would bury the salient change."""
    store = WorldStore()
    store.set(obj("robot.position"), "bench-3", at=T0)
    store.set(obj("robot.position"), "midway", at=T5, by=JOB_B)
    store.set(obj("robot.position"), "doorway", at=T9, by=JOB_B)

    dirty = store.dirty_for(ObjectSet.of(["robot.*"]), since=T5)
    assert len(dirty) == 1
    assert (dirty[0].obj, dirty[0].before, dirty[0].after) == (
        obj("robot.position"),
        "bench-3",
        "doorway",
    )


def test_dirty_for_respects_the_read_set() -> None:
    store = WorldStore()
    store.set(obj("robot.position"), "doorway", at=T9, by=JOB_B)
    store.set(obj("weather.temp"), "31", at=T9, by=JOB_B)
    dirty = store.dirty_for(ObjectSet.of(["robot.*"]), since=T5)
    assert [d.obj for d in dirty] == [obj("robot.position")]


def test_dirty_for_excludes_the_jobs_own_writes() -> None:
    """A job is not surprised by its own effects, and including them would train it
    to distrust its own actions."""
    store = WorldStore()
    store.set(obj("robot.position"), "doorway", at=T9, by=JOB_A)
    store.set(obj("robot.gripper"), "open", at=T9, by=JOB_B)

    dirty = store.dirty_for(ObjectSet.of(["robot.*"]), since=T5, exclude_job=JOB_A)
    assert [d.obj for d in dirty] == [obj("robot.gripper")]


def test_dirty_for_drops_round_trips() -> None:
    """Changed and changed back is not a change the resumed job needs to revalidate."""
    store = WorldStore()
    store.set(obj("plant.valve"), "shut", at=T0)
    store.set(obj("plant.valve"), "open", at=T5, by=JOB_B)
    store.set(obj("plant.valve"), "shut", at=T9, by=JOB_B)
    assert store.dirty_for(ObjectSet.of(["plant.*"]), since=T5) == ()


def test_dirty_for_is_empty_when_nothing_relevant_changed() -> None:
    store = WorldStore()
    store.set(obj("other.thing"), "1", at=T9, by=JOB_B)
    assert store.dirty_for(ObjectSet.of(["robot.*"]), since=T0) == ()


def test_dirty_output_is_sorted_for_determinism() -> None:
    store = WorldStore()
    for name in ("robot.z", "robot.a", "robot.m"):
        store.set(obj(name), "v", at=T9, by=JOB_B)
    dirty = store.dirty_for(ObjectSet.of(["robot.*"]), since=T0)
    assert [d.obj for d in dirty] == [obj("robot.a"), obj("robot.m"), obj("robot.z")]


def test_render_for_status_region() -> None:
    store = WorldStore()
    store.set(obj("site.alarm_state"), "nominal", at=T0)
    rendered = store.render([obj("site.alarm_state"), obj("site.missing")])
    assert rendered == "site.alarm_state: nominal\nsite.missing: (unset)"
