# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What both backends share: the prompt, the parser, and the record.

`PromptPlayer` owns all three, so a comparison between backends or between views
measures the thing under test. Driven through the Claude stub here because one
of the two had to be picked -- the assertions are about the rendered turn and the
`Decision` that comes back, neither of which is vendor-specific.
"""

import pytest
from stubs import claude_player, state


def test_history_accumulates_up_to_the_window():
    p = claude_player(history=2)
    for step in range(4):
        p.choose(f"BOARD{step}", state(step=step))
    last = p.client.calls[-1]["messages"][0]["content"]
    assert "BOARD1" in last and "BOARD2" in last  # inside the two-turn window
    assert "BOARD0" not in last  # older than the window
    assert last.count("you played:") == 2


def test_first_turn_has_no_history_section():
    p = claude_player()
    p.choose("BOARD", state())
    assert "Recent turns" not in p.client.calls[0]["messages"][0]["content"]


def test_status_line_reports_the_gun():
    p = claude_player()
    p.choose("BOARD", state(step=3, score=20, lives=2, can_shoot=False))
    content = p.client.calls[0]["messages"][0]["content"]
    assert "step 3 | score 20 | lives 2 | missile in flight: yes" in content


def test_unparseable_reply_falls_back_and_is_flagged():
    p = claude_player("I refuse to play")
    action, record = p.choose("BOARD", state())
    assert action == "left" and record.parsed is False


def test_an_action_is_read_out_of_a_sentence():
    """Models rarely answer with the bare word, and a parser is not a model."""
    p = claude_player("I'll go with `shoot` this turn.")
    action, record = p.choose("BOARD", state())
    assert action == "shoot" and record.parsed is True


def test_the_decision_carries_the_prompt_that_produced_it():
    """Not derivable from the frames: it is what the model actually saw."""
    p = claude_player("left")
    _, record = p.choose("THE BOARD", state(step=2))
    assert "THE BOARD" in record.prompt
    assert record.by == "model" and record.tick == 2


def test_the_default_prompt_is_built_from_the_view_and_the_shared_rules():
    """Only the format and decision sections vary; the rules are one copy."""
    from zeos_space_invaders.game.rules import DEFAULTS
    from zeos_space_invaders.utils import VIEWS
    from zeos_space_invaders.utils.views import describe

    shared = None
    for name, view in VIEWS.items():
        prompt = claude_player(view=view(), prompt=None).prompt
        section = describe(view(), DEFAULTS)
        assert section.strip().splitlines()[0] in prompt, name
        rules = prompt.split("---", 1)[1]
        head = rules[: rules.index("##", 1)] if "##" in rules[1:] else rules
        shared = shared if shared is not None else head
        assert head == shared, f"the rules body drifted for {name}"


def test_respond_is_the_backend_seam_and_has_no_default():
    """`PromptPlayer` renders and parses; issuing the request is the backend's."""
    import pytest

    from zeos_space_invaders.players import PromptPlayer

    with pytest.raises(NotImplementedError):
        PromptPlayer(prompt="RULES").respond("BOARD", "none")


def test_a_broken_environment_is_not_reported_as_a_missing_extra():
    """An SDK that is installed but cannot import keeps its traceback."""
    import importlib

    import pytest

    from zeos_space_invaders.players.base import SDKMissing, sdk

    def import_module(name):
        raise ModuleNotFoundError("No module named 'httpx'", name="httpx")

    original = importlib.import_module
    importlib.import_module = import_module
    try:
        with pytest.raises(ModuleNotFoundError) as raised:
            sdk("anthropic", "claude")
    finally:
        importlib.import_module = original
    assert not isinstance(raised.value, SDKMissing)
    assert "httpx" in str(raised.value)


def _unstreamed(backend):
    """The same player either side; the two stub factories take different shapes,
    spelt out here rather than papered over."""
    from stubs import claude_player, message, openai_player

    if backend == "claude":
        return claude_player("left", stream=False)
    return openai_player({"output": [message("left")]}, stream=False)


@pytest.mark.parametrize("backend", ["openai", "claude"])
def test_no_stream_asks_for_the_whole_reply_and_still_yields_pieces(backend):
    """`--no-stream` is a wire-level switch, not a different code path: a caller
    reads pieces either way, there are just two of them."""
    player = _unstreamed(backend)
    assert list(player.respond_stream("BOARD", "none")) == [("answer", "left")]
    assert "stream" not in player.client.calls[0], "it streamed anyway"


def test_the_missing_sdk_hint_names_a_distribution_that_exists():
    """The hint is a command someone pastes, so both halves must resolve.

    Read from the installed metadata rather than written down twice: a distribution
    renamed in `pyproject.toml` should fail here rather than in a user's shell.
    """
    from importlib.metadata import metadata

    from zeos_space_invaders.players.base import SDKMissing, sdk

    dist = metadata("zeos-space-invaders")
    extras = dist.get_all("Provides-Extra") or []
    with pytest.raises(SDKMissing) as raised:
        sdk("no_such_sdk", "claude")
    assert f"{dist['Name']}[claude]" in str(raised.value)
    assert "claude" in extras
