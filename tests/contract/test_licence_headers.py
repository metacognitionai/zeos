# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Policy-as-test: every source file carries the licence it is distributed under.

The AGPL's own "How to Apply These Terms" asks for a per-file notice, and the
SPDX short form is the modern spelling of it. The point is a file that is copied
out of this tree -- into an issue, a gist, another repository -- still says what
it is licensed as.

The expected identifier is read from ``pyproject.toml`` rather than written here
twice, so the package metadata is the single source of truth and a licence change
is one edit, not a hunt.

Third-party material is deliberately excluded: anything under a ``vendor``
directory keeps the licence it arrived with.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import zeos

REPO = Path(zeos.__file__).resolve().parents[2]
ROOTS = ("src", "tests", "demo")
SUFFIXES = (".py", ".js", ".css")


def _declared_licence() -> str:
    with (REPO / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["license"]


def _sources() -> list[Path]:
    found: list[Path] = []
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            # Vendored assets keep their own licence, and a minified bundle is
            # not a file anyone edits.
            if "vendor" in path.relative_to(REPO).parts or ".min." in path.name:
                continue
            found.append(path)
    return found


def test_there_are_sources_to_check() -> None:
    """A glob that silently matches nothing would pass every test below."""
    assert len(_sources()) > 100


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_every_source_file_declares_the_licence(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    notice = f"SPDX-License-Identifier: {_declared_licence()}"
    assert notice in text, f"{path.relative_to(REPO)} has no `{notice}` header"
    # The identifier names the licence; this points at the copy of it, for a reader
    # who has the file but not the tree.
    assert "LICENSE file in the root directory of this source tree." in text, (
        f"{path.relative_to(REPO)} does not point at the LICENSE file"
    )


def test_the_licence_file_is_the_one_the_metadata_names() -> None:
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert _declared_licence() == "AGPL-3.0-only"
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 19 November 2007" in text
    # Section 13 is why the AGPL was chosen over the GPL; a truncated copy that
    # lost it would still look like a licence.
    assert "13. Remote Network Interaction" in text


def test_the_debugger_offers_its_source_to_whoever_it_is_served_to() -> None:
    """AGPL section 13: a networked debugger owes its users the Corresponding Source.

    The offer is asserted against the URL the package metadata publishes, so the two
    cannot drift into pointing at different repositories.
    """
    with (REPO / "pyproject.toml").open("rb") as handle:
        homepage = tomllib.load(handle)["project"]["urls"]["Homepage"]
    page = (REPO / "src/zeos/debugger/static/index.html").read_text(encoding="utf-8")
    assert f'id="source" href="{homepage}"' in page
