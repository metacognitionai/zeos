# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The server and the page it hands over.

The page draws what `payload` returns and that is tested next door; here it is
the boundary: which URL reaches which run, which are refused, and that the
exported file stands on its own.
"""

import http.client
import json
import threading

import pytest

from zeos_space_invaders import viewer
from zeos_space_invaders.runlog import RunWriter
from zeos_space_invaders.web import server


def make_run(path, config=None):
    info = {
        "ticks": 0,
        "score": 0,
        "lives": 3,
        "player": 4,
        "monsters": {},
        "missile": None,
        "dangers": [],
        "can_shoot": True,
        "won": False,
    }
    with RunWriter(path, config or {"player": "random", "clock": "step"}) as run:
        run.frame(info)
        run.finish({"outcome": "lost", "score": 0, "decisions": 0})
    return path


@pytest.fixture
def site(tmp_path):
    """A server on a port the OS picks, serving a runs directory of two."""
    make_run(tmp_path / "20260101-000000-random-seed1")
    root = tmp_path / "20260102-000000-compare"
    make_run(root / "episodes" / "random-seed0", {"player": "random", "seed": 0})
    RunWriter(root, {"kind": "compare", "players": ["random"], "seeds": 1}).close()

    httpd = server.serve(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def fetch(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read().decode()
    headers = dict(response.getheaders())
    conn.close()
    return response.status, body, headers


def get(port, path):
    status, body, _ = fetch(port, path)
    return status, body


def test_a_query_string_is_not_part_of_the_route(site):
    assert get(site, "/?v=2")[0] == 200
    assert get(site, "/api/index?v=2")[0] == 200


def test_the_page_is_never_cached(site):
    """It is reassembled from `static/` on every request and a run directory
    grows while it is being watched."""
    _, _, headers = fetch(site, "/")
    assert headers["Cache-Control"] == "no-store"
    _, _, headers = fetch(site, "/api/index")
    assert headers["Cache-Control"] == "no-store"


def test_the_page_comes_back_whole(site):
    status, body = get(site, "/")
    assert status == 200
    for marker in ("__DATA__", "/*JS*/", "/*CSS*/", "/*VENDOR_JS*/", "/*VENDOR_CSS*/"):
        assert marker not in body, marker
    assert "window.__RUN__ = null" in body, "a served page fetches its own data"
    # A string from the minified library and nowhere else; our own comments say
    # "Tabulator" too often for that word to prove anything.
    assert "Data pipeline handlers must have a prior" in body, (
        "the index's table library is inlined, not linked"
    )


def test_the_served_page_asks_the_network_for_nothing_either(site):
    """Half a megabyte of somebody else's JavaScript is exactly where an asset
    request would sneak back in."""
    _, body = get(site, "/")
    for line in body.splitlines():
        if "http" not in line:
            continue
        assert "www.w3.org/2000/svg" in line, line[:200]


def test_the_index_lists_what_is_on_disk(site):
    status, body = get(site, "/api/index")
    rows = json.loads(body)
    assert status == 200 and len(rows) == 2
    assert {row["is_compare"] for row in rows} == {True, False}


def test_an_episode_of_a_comparison_is_addressed_through_its_comparison(site):
    status, body = get(site, "/api/run/20260102-000000-compare/episodes/random-seed0")
    assert status == 200
    assert json.loads(body)["meta"]["seed"] == 0


def test_a_comparison_answers_on_its_own_route(site):
    status, body = get(site, "/api/compare/20260102-000000-compare")
    assert status == 200 and json.loads(body)["meta"]["kind"] == "compare"


@pytest.mark.parametrize(
    "path",
    ["/api/run/../../../etc", "/api/run/%2e%2e/%2e%2e", "/api/run/nothing-here"],
)
def test_a_path_that_leaves_the_runs_directory_is_refused(site, path):
    """It serves whatever the URL names, so this is the only thing stopping it."""
    assert fetch(site, path)[0] == 404


def test_an_unknown_route_is_a_404_rather_than_a_traceback(site):
    assert fetch(site, "/api/nonsense/x")[0] == 404
    assert fetch(site, "/elsewhere")[0] == 404


def test_a_run_from_another_layout_says_so_rather_than_500(tmp_path):
    old = tmp_path / "20260101-000000-old"
    old.mkdir()
    (old / "meta.json").write_text('{"schema": 1}')
    httpd = server.serve(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = get(httpd.server_address[1], "/api/run/20260101-000000-old")
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert status == 409


# --- the exported file -------------------------------------------------------


def test_an_exported_run_carries_its_own_data(tmp_path):
    run = make_run(tmp_path / "20260101-000000-random-seed1")
    out = tmp_path / "run.html"
    server.export(run, out)
    page = out.read_text()
    assert '"frames"' in page and "window.__RUN__ = {" in page


def test_an_exported_run_leaves_the_table_library_behind(tmp_path):
    """A run screen has no sortable table -- the index and the comparison do --
    so an exported run must not carry half a megabyte to draw one."""
    run = make_run(tmp_path / "20260101-000000-random-seed1")
    out = tmp_path / "run.html"
    server.export(run, out)
    page = out.read_text()
    # A string from the minified library and nowhere else.
    assert "Data pipeline handlers must have a prior" not in page
    assert "/*VENDOR_JS*/" not in page, "the marker is emptied, not left in place"
    served = server.page()
    assert len(served) > len(page) + 300_000, "and the served page does carry it"


def test_an_exported_run_asks_the_network_for_nothing(tmp_path):
    """It has to open from a filesystem, on a machine with no network."""
    run = make_run(tmp_path / "20260101-000000-random-seed1")
    out = tmp_path / "run.html"
    server.export(run, out)
    page = out.read_text()
    for line in page.splitlines():
        if "http" not in line:
            continue
        # The SVG namespace is a name, not an address; nothing else may appear.
        assert "www.w3.org/2000/svg" in line, line


def test_a_second_use_of_the_placeholder_is_an_error_not_a_broken_page(
    tmp_path, monkeypatch
):
    """The placeholder is consumed by the substitution, so the check has to be
    that the assembled page names it exactly once."""
    static = _shell(tmp_path, js="const DATA = window.__DATA__;")
    monkeypatch.setattr(server, "STATIC", static)
    with pytest.raises(ValueError, match="appears 2 times"):
        server.page()


def test_an_asset_holding_another_assets_marker_is_an_error(tmp_path, monkeypatch):
    """The assets go in one at a time, so the count is per asset."""
    static = _shell(tmp_path, css="/*JS*/")
    monkeypatch.setattr(server, "STATIC", static)
    with pytest.raises(ValueError, match=r"/\*JS\*/ appears 2 times"):
        server.page()


def _shell(tmp_path, css="", js=""):
    """A minimal `static/` with all four markers and no real assets."""
    static = tmp_path / "static"
    (static / "vendor" / "tabulator").mkdir(parents=True)
    (static / "viewer.css").write_text(css)
    (static / "viewer.js").write_text(js)
    (static / "vendor" / "tabulator" / "tabulator.min.css").write_text("")
    (static / "vendor" / "tabulator" / "tabulator.min.js").write_text("")
    (static / "index.html").write_text(
        "<style>/*VENDOR_CSS*/</style><style>/*CSS*/</style>"
        "<script>window.__RUN__ = __DATA__;</script>"
        "<script>/*VENDOR_JS*/</script><script>/*JS*/</script>"
    )
    return static


# --- the entry point ---------------------------------------------------------


class FakeServer:
    """Stands in for the real one: bound, then interrupted, then closed."""

    def __init__(self):
        self.server_address = ("127.0.0.1", 8000)
        self.closed = False

    def serve_forever(self):
        raise KeyboardInterrupt  # what ctrl-c does to the real loop

    def server_close(self):
        self.closed = True


def test_serving_says_where_and_closes_the_socket_on_ctrl_c(
    tmp_path, capsys, monkeypatch
):
    fake, opened = FakeServer(), []
    monkeypatch.setattr(viewer, "serve", lambda root, port: fake)
    monkeypatch.setattr(viewer.webbrowser, "open", opened.append)

    viewer.main(["--root", str(tmp_path)])

    assert "http://127.0.0.1:8000" in capsys.readouterr().out
    assert opened == ["http://127.0.0.1:8000"]
    assert fake.closed, "the port would stay bound until the process died"


def test_no_open_leaves_the_browser_alone(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(viewer, "serve", lambda root, port: FakeServer())
    monkeypatch.setattr(viewer.webbrowser, "open", opened.append)
    viewer.main(["--root", str(tmp_path), "--no-open"])
    assert opened == []


def test_export_from_the_command_line(tmp_path, capsys):
    run = make_run(tmp_path / "20260101-000000-random-seed1")
    out = tmp_path / "one.html"
    viewer.main(["--export", str(run), str(out)])
    assert out.is_file() and str(out) in capsys.readouterr().out


def test_the_board_and_its_pieces_take_their_size_from_one_place():
    """`W`/`H` are the fallback for runs recorded before `meta` carried the size,
    so their only legitimate uses are the declaration and the two `||` defaults."""
    import re

    js = (server.STATIC / "viewer.js").read_text()
    uses = [
        line.strip()
        for line in js.splitlines()
        if re.search(r"(?<![\w$])[WH](?![\w$])", line)
    ]
    assert uses == [
        "const W = 9, H = 8, CELL = 40;",
        "let played = {w: W, h: H};",
        "played = {w: meta.width || W, h: meta.height || H};",
    ], f"a hardcoded board dimension reached the drawing: {uses}"
