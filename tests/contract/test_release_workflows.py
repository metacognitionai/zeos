# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Policy-as-test: the strings PyPI checks are the ones this repository uses.

Trusted publishing has no secret. What PyPI verifies is a tuple -- repository,
owner, workflow filename, environment name -- registered once on the index and
never seen again by anyone here. Rename a workflow file or an environment and
nothing fails until an upload is refused, which is the worst moment to find out.

So the four strings are pinned here, against the table in ``docs/release.md``
that a human reads when registering them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import zeos

REPO = Path(zeos.__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

#: Workflow filename -> the environment its publishing job must run in.
PUBLISHERS = {"release.yml": "release", "testpypi.yml": "testpypi"}


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("filename", "environment"), sorted(PUBLISHERS.items()))
def test_the_publishing_job_runs_in_the_environment_pypi_expects(
    filename: str, environment: str
) -> None:
    publish = _workflow(filename)["jobs"]["publish"]
    declared = publish["environment"]
    assert declared["name"] == environment, (
        f"{filename} publishes from {declared['name']!r}, not {environment!r}"
    )
    # The OIDC token is the whole credential; without this the exchange cannot
    # happen at all.
    assert publish["permissions"]["id-token"] == "write"


@pytest.mark.parametrize("filename", sorted(PUBLISHERS))
def test_nothing_is_granted_that_is_not_asked_for(filename: str) -> None:
    """A workflow holding an upload token starts from nothing, per job."""
    workflow = _workflow(filename)
    assert workflow["permissions"] == {}, "the workflow's default permissions must be empty"
    for name, job in workflow["jobs"].items():
        assert "permissions" in job, f"{filename}: job {name} declares no permissions"


@pytest.mark.parametrize("filename", sorted(PUBLISHERS))
def test_every_action_is_pinned_to_a_commit(filename: str) -> None:
    """A tag can be moved by whoever owns the action; these jobs are next door to
    an upload token, so they take a commit or nothing."""
    for name, job in _workflow(filename)["jobs"].items():
        for step in job["steps"]:
            uses = step.get("uses")
            if uses is None:
                continue
            _, _, ref = uses.partition("@")
            assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref), (
                f"{filename}: job {name} uses {uses}, which is not a commit sha"
            )


def test_the_documented_table_says_what_the_workflows_do() -> None:
    """``docs/release.md`` is where a human reads the four strings off when
    registering a publisher, so it is part of the contract, not a description
    of it."""
    prose = (REPO / "docs" / "release.md").read_text(encoding="utf-8")
    for filename, environment in PUBLISHERS.items():
        assert f"`{filename}`" in prose
        assert f"`{environment}`" in prose
