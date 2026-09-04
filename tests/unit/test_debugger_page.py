# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Assembling the page: the substitutions, and the promise that it is self-contained.

The marker checks are not defensive programming -- they are the one failure this
assembly has, and it is silent. A marker appearing twice does not raise on its own:
the JSON gets spliced into the middle of whichever copy came second, and the page
fails to parse in a browser instead of failing to build here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zeos.debugger import server
from zeos.debugger.payload import build_payload
from zeos.debugger.server import ASSETS, PLACEHOLDER, export, page
from zeos.descriptor.loader import load_case

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


def test_the_assembled_page_inlines_everything() -> None:
    built = page({"structure": {"case": "smoke"}})

    assert PLACEHOLDER not in built
    for marker, _name in ASSETS:
        assert marker not in built
    assert "<script src=" not in built, "an exported page must fetch nothing"
    assert "<link " not in built, "an exported page must fetch nothing"


def test_the_data_lands_in_the_page() -> None:
    assert '"case":"smoke"' in page({"case": "smoke"})
    assert "window.__ZEOS__ = null" in page()


def test_a_missing_marker_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shell = tmp_path / "index.html"
    shell.write_text("<html><style>/*CSS*/</style></html>", encoding="utf-8")
    (tmp_path / "debugger.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "debugger.js").write_text("void 0;", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC", tmp_path)

    with pytest.raises(ValueError, match=r"/\*JS\*/ appears 0 times"):
        page()


def test_a_repeated_placeholder_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shell = tmp_path / "index.html"
    shell.write_text(
        "<html><style>/*CSS*/</style><script>/*JS*/</script>"
        f"<b>{PLACEHOLDER}</b><i>{PLACEHOLDER}</i></html>",
        encoding="utf-8",
    )
    (tmp_path / "debugger.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "debugger.js").write_text("void 0;", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC", tmp_path)

    with pytest.raises(ValueError, match=f"{PLACEHOLDER} appears 2 times"):
        page()


def test_an_asset_may_not_carry_another_assets_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shell = tmp_path / "index.html"
    shell.write_text(
        f"<html><style>/*CSS*/</style><script>/*JS*/</script>{PLACEHOLDER}</html>",
        encoding="utf-8",
    )
    (tmp_path / "debugger.css").write_text("body{} /*JS*/", encoding="utf-8")
    (tmp_path / "debugger.js").write_text("void 0;", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC", tmp_path)

    with pytest.raises(ValueError, match=r"/\*JS\*/ appears 2 times"):
        page()


def test_export_writes_one_openable_file(tmp_path: Path) -> None:
    payload = build_payload(load_case(SMOKE))
    out = export(payload, tmp_path / "nested" / "smoke.html")

    written = out.read_text(encoding="utf-8")
    assert out.exists()
    assert written.startswith("<!doctype html>")
    assert "smoke" in written
    assert "<script src=" not in written
