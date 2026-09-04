# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Policy-as-test: what the page is allowed to know, and what it may reach for.

Two drift classes this catches, both of which are silent in a browser.

**Event kinds.** The mark vocabulary ships as data (``payload`` carries
``MARKED_KINDS``) precisely so the renderer holds no list of its own. If one of
those kinds is ever renamed, this fails instead of the jump-to dropdown quietly
going empty.

**Externals.** An exported page has to open from a filesystem with no network. A
stylesheet link or a script src added to the shell would work perfectly on the
machine that wrote it and be blank everywhere else, which is the worst shape a
regression can have.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from zeos.core.events import EVENT_REGISTRY
from zeos.debugger.payload import MOVEMENTS
from zeos.debugger.server import ASSETS, PLACEHOLDER, STATIC
from zeos.monitor.state import MARKED_KINDS

SHELL = STATIC / "index.html"
SCRIPT = STATIC / "debugger.js"


def test_every_marked_kind_is_a_real_event() -> None:
    unknown = sorted(set(MARKED_KINDS) - set(EVENT_REGISTRY))
    assert not unknown, f"the scrubber offers marks for events that do not exist: {unknown}"


def test_the_page_draws_every_movement_the_log_can_carry() -> None:
    """``MOVEMENTS`` is the vocabulary and the page holds the glyphs, because a glyph
    is presentation and a vocabulary is data. That split only stays honest if the two
    are checked against each other: a movement the page has no entry for draws with
    its raw name and no colour, which is a regression nobody would notice.
    """
    body = SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"var MOVES = \{(.*?)\n  \};", body, re.S)
    assert block, "debugger.js no longer declares a MOVES table"
    drawn = set(re.findall(r"(\w+): \{", block.group(1)))
    assert drawn == set(MOVEMENTS), (
        f"the page draws {sorted(drawn)} but the token log carries {sorted(MOVEMENTS)}"
    )


def test_a_hidden_element_can_actually_hide() -> None:
    """Every element the shell marks ``hidden`` must still be hideable once the page's
    own CSS has had its say.

    The trap is silent and cost a round trip to catch: ``hidden`` works through the user
    agent's ``[hidden] { display: none }``, which *any* author rule setting ``display``
    outranks. Give a hidden element a class that says ``display: flex`` and it is on
    screen from first paint, and toggling ``.hidden`` in JavaScript does nothing at all
    -- no error, no warning, just a dialog that will not close.

    So for each selector the shell hides, if the stylesheet gives it a ``display``, it
    must also carry a ``[hidden]`` rule of its own.
    """
    shell = SHELL.read_text(encoding="utf-8")
    css = (STATIC / "debugger.css").read_text(encoding="utf-8")

    hidden_tags = re.findall(r"<[^>]*\bhidden\b[^>]*>", shell)
    assert hidden_tags, "the shell hides nothing; this test has lost its subject"

    selectors: set[str] = set()
    for tag in hidden_tags:
        for attr, prefix in (("id", "#"), ("class", ".")):
            found = re.search(rf'{attr}="([^"]+)"', tag)
            if found:
                selectors.update(prefix + token for token in found.group(1).split())

    for selector in sorted(selectors):
        # Does any rule for this selector alone set a display?
        pattern = re.escape(selector) + r"(?![\w-])[^{}]*\{[^{}]*\bdisplay\s*:"
        if not re.search(pattern, css):
            continue
        escaped = re.escape(selector)
        assert re.search(rf"{escaped}\[hidden\]|\[hidden\][^{{}}]*{escaped}", css), (
            f"{selector} is hidden in the shell and given a display by the stylesheet, "
            f"so it needs a {selector}[hidden] rule -- without one the element is "
            "always on screen and toggling .hidden silently does nothing"
        )


def test_every_custom_property_the_stylesheet_uses_is_one_it_defines() -> None:
    """A ``var()`` naming a property that does not exist resolves to nothing.

    Mostly that is a wrong colour, which is visible. In ``grid-template-columns`` it is
    not: the declaration becomes invalid and the whole layout falls back, so a single
    typo in a variable name moves every pane on the page with no error anywhere. The
    resizable side panel puts two custom properties on that exact declaration, so the
    cheap check is worth having.
    """
    css = (STATIC / "debugger.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"^\s*(--[\w-]+)\s*:", css, re.M))
    used = set(re.findall(r"var\(\s*(--[\w-]+)", css))
    # A var() may name a fallback for something set from script; none do today.
    missing = sorted(used - defined)
    assert not missing, f"the stylesheet uses custom properties it never defines: {missing}"


#: Two absolute URLs the page is allowed to carry, neither of which it fetches: the
#: SVG XML namespace is an identifier rather than an address, and the source link is
#: the offer AGPL section 13 requires the debugger to make to whoever it is served to
#: -- followed on a click, never loaded. Every other absolute URL would be something
#: the exported file cannot reach from a filesystem. ``tests/contract/
#: test_licence_headers.py`` pins the second against the metadata's own Homepage.
ALLOWED_URLS = frozenset({"http://www.w3.org/2000/svg", "https://github.com/metacognitionai/zeos"})


def test_the_page_fetches_nothing() -> None:
    for path in (SHELL, *(STATIC / name for _marker, name in ASSETS)):
        text = path.read_text(encoding="utf-8")
        assert "<script src=" not in text, f"{path.name} loads an external script"
        assert "<link " not in text, f"{path.name} loads an external stylesheet"
        found = set(re.findall(r"https?://[^\s\"'();]+", text)) - ALLOWED_URLS
        assert not found, f"{path.name} reaches for {sorted(found)}"


@pytest.mark.parametrize("name", [name for _marker, name in ASSETS])
def test_no_asset_carries_another_assets_marker(name: str) -> None:
    """The substitutions happen in order, so an asset containing a later marker would
    have that asset spliced into it silently."""
    text = (STATIC / name).read_text(encoding="utf-8")
    for marker, other in ASSETS:
        if other != name:
            assert marker not in text, f"{name} contains {other}'s marker"


def test_no_asset_uses_the_placeholders_own_name() -> None:
    """The scripts are substituted in before the data is, so a script mentioning
    ``__DATA__`` would have the JSON spliced into the middle of an expression."""
    for _marker, name in ASSETS:
        assert PLACEHOLDER not in (STATIC / name).read_text(encoding="utf-8")


def test_the_static_assets_ship_with_the_package() -> None:
    """They are package data, not repository files: ``zeos debug`` has to work from
    an installed wheel, on a case directory outside this repository."""
    for _marker, name in ASSETS:
        assert (STATIC / name).is_file()
    assert SHELL.is_file()
    assert STATIC.parent.name == "debugger"
    assert Path(STATIC).is_relative_to(Path(__file__).resolve().parents[2] / "src" / "zeos")
