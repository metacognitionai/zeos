# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Policy-as-test: the built distributions contain what they claim, and nothing else.

The wheel is the install artifact: it must contain the ``zeos`` package and its
dist-info, and nothing further -- a class of leak that has already happened once
in this repo's history. The sdist is the source artifact: looser by design (it
may carry tests, docs, and demos), but it must never carry secrets or run
artifacts, and it must be able to produce the wheel.

The release workflow runs the same checks before an upload; this copy runs on
every CI pass so a packaging leak is caught at the PR that introduces it.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

import zeos

REPO = Path(zeos.__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dist_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--package", "zeos", "--out-dir", str(out)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return out


def test_the_wheel_contains_zeos_and_nothing_else(dist_dir: Path) -> None:
    (wheel,) = dist_dir.glob("zeos-*.whl")
    names = zipfile.ZipFile(wheel).namelist()

    strays = [n for n in names if not n.startswith(("zeos/", "zeos-")) or n.startswith("zeos-demo")]
    assert not strays, f"the wheel must ship only the zeos package: {strays}"
    assert any(n == "zeos/kernel.py" or n.endswith("/kernel.py") for n in names)
    assert any(n.startswith("zeos/core/") for n in names), "the kernel core is missing"
    assert not any(n.startswith(("tests/", "docs/")) for n in names)


def test_the_wheel_carries_the_type_marker(dist_dir: Path) -> None:
    (wheel,) = dist_dir.glob("zeos-*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert "zeos/py.typed" in names, "downstream type checking depends on the marker"


def test_the_sdist_builds_the_wheel_and_leaks_no_artifacts(dist_dir: Path) -> None:
    (sdist,) = dist_dir.glob("zeos-*.tar.gz")
    names = tarfile.open(sdist).getnames()

    root = names[0].split("/")[0]
    assert {f"{root}/pyproject.toml"} <= set(names)
    assert any(n.startswith(f"{root}/src/zeos/") for n in names), "no source in the sdist"
    assert not any(n.endswith(".env") for n in names), "a secrets file leaked into the sdist"
    assert not any("/runs/" in n for n in names), "run artifacts leaked into the sdist"
    # Model weights are downloaded, not authored, and one of them is larger than
    # everything else here put together. They are gitignored by each demo's own
    # .gitignore, which hatchling does not read, so the exclusion has to be declared
    # in the sdist target -- and checked, because the failure is a multi-gigabyte
    # upload rather than a wrong file list.
    assert not any(n.endswith(".gguf") for n in names), "model weights leaked into the sdist"
    assert sdist.stat().st_size < 32 * 1024 * 1024, (
        f"the sdist is {sdist.stat().st_size // (1024 * 1024)}MB; something large leaked"
    )
    assert not any(n.endswith((".pyc", ".jsonl")) or "__pycache__" in n for n in names) or any(
        "tests/fixtures" in n for n in names if n.endswith(".jsonl")
    )


def test_the_sdist_version_matches_the_wheel(dist_dir: Path) -> None:
    (wheel,) = dist_dir.glob("zeos-*.whl")
    (sdist,) = dist_dir.glob("zeos-*.tar.gz")
    wheel_version = wheel.name.split("-")[1]
    sdist_version = sdist.name.removesuffix(".tar.gz").split("-")[1]
    assert wheel_version == sdist_version
    assert sys.version_info >= (3, 12)  # the floor the wheel declares


def test_the_wheel_declares_the_licence_and_ships_it(dist_dir: Path) -> None:
    """PyPI renders what the metadata says, not what the repository contains.

    The expression and the file are separate failures: a wheel can name a licence
    it does not carry, or carry one it does not name.
    """
    (wheel,) = dist_dir.glob("zeos-*.whl")
    archive = zipfile.ZipFile(wheel)
    (metadata,) = (n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
    declared = archive.read(metadata).decode("utf-8")

    with (REPO / "pyproject.toml").open("rb") as handle:
        expected = tomllib.load(handle)["project"]["license"]
    assert f"License-Expression: {expected}" in declared
    # A `License ::` classifier alongside the expression is rejected by PyPI at
    # upload, which is the last place anyone wants to discover it.
    assert "Classifier: License ::" not in declared
    assert "Author: " in declared, "an unattributed package on PyPI"

    (licence,) = (n for n in archive.namelist() if n.endswith("licenses/LICENSE"))
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in archive.read(licence).decode("utf-8")


def test_the_sdist_leaves_the_repository_machinery_behind(dist_dir: Path) -> None:
    """A source release is what is needed to build and read ZEOS. CI wiring and
    agent instructions are how this repository is worked on, which is a different
    question and not one an installer asks."""
    (sdist,) = dist_dir.glob("zeos-*.tar.gz")
    names = tarfile.open(sdist).getnames()
    root = names[0].split("/")[0]

    for unwanted in (f"{root}/.github", f"{root}/AGENTS.md", f"{root}/CLAUDE.md"):
        assert not any(n == unwanted or n.startswith(f"{unwanted}/") for n in names), (
            f"{unwanted} should not be in the sdist"
        )
    # The corresponding source still has to be there in full.
    assert any(n.startswith(f"{root}/docs/") for n in names)
    assert any(n.startswith(f"{root}/demo/") for n in names)
    assert f"{root}/LICENSE" in names


def test_the_package_reports_the_version_the_metadata_declares() -> None:
    """``zeos.__version__`` reads the installed distribution, so this catches a
    stale install as well as a version written in two places."""
    with (REPO / "pyproject.toml").open("rb") as handle:
        assert zeos.__version__ == tomllib.load(handle)["project"]["version"]
