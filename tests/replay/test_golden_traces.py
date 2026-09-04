# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Replay tier: the journal is a replayable trace, so a run becomes a test case.

The Programming Model: "the journal is a replayable trace. A field incident replays as
a test case: same events, same seeds, assert the fixed behaviour."

What is checked here is the **golden shape**: a committed digest of the run's event-kind
sequence, so that an unintended behavioural change anywhere in the kernel shows up as a
diff in a specific place rather than as a vague suspicion. Byte-stability and the file
round trip are checked in ``tests/integration/test_stage_b_case.py``, against the same
fixture and closer to the driver that produces them.

The digest is deliberately a *hash of the event kind sequence* rather than of the whole
journal. Full-journal hashes break on every cosmetic change -- a reworded fault detail,
an added field -- and a fixture that cries wolf gets regenerated reflexively, which
destroys its value. The kind sequence changes only when the *shape* of the run changes,
which is what we actually want to be told about.

The subject is ``tests/fixtures/smoke``, which describes no application. This tier used
to run a case describing a real industry; the shape of a run is a property of the kernel
rather than of whatever domain a case happens to describe, so the kernel repository does
not need one in order to check it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from zeos.core.ids import PipeName
from zeos.core.kernel import KernelConfig
from zeos.descriptor.loader import load_case
from zeos.driver import Driver, ScheduledEvent, build_kernel
from zeos.journal.writer import Journal, read_journal

CASE = Path(__file__).parent.parent / "fixtures" / "smoke"
GOLDEN = Path(__file__).parent / "golden"

#: Mid-flight for the goal job, so the run contains a preemption and a resume rather
#: than two jobs that never met. Matches the value the Stage B tests use.
ALARM_AT_NS = 3_000_000

ALARM = (ScheduledEvent(at_ns=ALARM_AT_NS, pipe=PipeName("sensors.threshold"), text="level 42"),)


def shape_digest(kinds: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for kind in kinds:
        digest.update(kind.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run(journal_path: Path, schedule: tuple[ScheduledEvent, ...] = ALARM) -> list[str]:
    bundle = load_case(CASE)
    kernel, transport = build_kernel(
        bundle, journal_sink=[], config=KernelConfig(case=bundle.name), block_size=8
    )
    journal = Journal(journal_path)
    driver = Driver(kernel, transport=transport, journal=journal)
    driver.boot(bundle.boot)
    driver.run(schedule)
    journal.close()
    return [type(record.event).__name__ for record in read_journal(journal_path)]


def test_run_shape_matches_the_golden_trace(tmp_path: Path) -> None:
    """Regenerate with ``uv run python -m tests.replay.regenerate`` after an
    intentional behavioural change, and read the diff before committing it."""
    kinds = run(tmp_path / "run.jsonl")
    expected_path = GOLDEN / "smoke.shape"
    if not expected_path.is_file():
        pytest.skip(f"no golden trace; write {expected_path}")
    expected = expected_path.read_text("utf-8").strip()
    actual = shape_digest(kinds)
    assert actual == expected, (
        f"run shape changed (was {expected}, now {actual}); {len(kinds)} events. "
        "If this was intended, regenerate the golden trace."
    )


def test_the_digest_is_measuring_something(tmp_path: Path) -> None:
    """A sanity check on the fixture itself.

    If a run with the interrupt and a run without it hashed the same, the digest would
    not be sensitive to the one thing this fixture exists to exercise, and a golden
    trace over it would be decoration.
    """
    with_alarm = run(tmp_path / "with.jsonl")
    without = run(tmp_path / "without.jsonl", schedule=())
    assert shape_digest(with_alarm) != shape_digest(without)


def test_the_trace_contains_the_expected_landmarks(tmp_path: Path) -> None:
    """The digest tells you *that* something changed; these say *what* should be
    there, so a failure is diagnosable without decoding a hash.

    ``IntegrityDemoted`` is not among them because this fixture reads no ring-3 pipe.
    Demotion is pinned in the contract tier instead, where a behaviour reads an
    external feed and its final integrity is asserted.

    ``AttentionDenied`` is on the list because of how it nearly left it. Rewording the
    fixture's prose changed its token counts, the masking bookkeeping moved with them,
    and the count went to zero -- caught only by the digest, because no test named the
    mechanism. A digest tells you something moved; it does not tell you that a
    protection stopped being exercised. So the mechanism is named here now.
    """
    kinds = run(tmp_path / "run.jsonl")
    for landmark in (
        "VectorFired",
        "JobPreempted",
        "JobResumed",
        "CapabilityChecked",
        "WorldWritten",
        "MaskUpdated",
        "AttentionDenied",
    ):
        assert landmark in kinds, f"expected {landmark} in the trace"
