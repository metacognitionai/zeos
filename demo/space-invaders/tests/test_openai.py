# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""OpenAIPlayer against a stub Responses client — no server needed.

Only what is specific to this backend: the Responses API request shape, the
effort switch, where reasoning hides in the output list, and the retry when
reasoning eats the whole budget. What both backends share is in
`test_prompt_player.py`.
"""

import pytest
from stubs import StubResponses, message, reasoning, state, usage

from zeos_space_invaders.players import OpenAIPlayer, ThinkingNotDisabled

# --- the request ------------------------------------------------------------


def test_the_prompt_is_sent_as_instructions_and_the_board_as_input():
    stub = StubResponses()
    OpenAIPlayer(prompt="THE RULES", client=stub).choose("BOARD", state())
    sent = stub.calls[0]
    assert sent["instructions"] == "THE RULES"
    assert "BOARD" in sent["input"]


def test_effort_is_passed_straight_through():
    for level in ("none", "low", "high"):
        stub = StubResponses()
        OpenAIPlayer(prompt="RULES", effort=level, client=stub).choose("BOARD", state())
        assert stub.calls[0]["reasoning"] == {"effort": level}


def test_omitting_effort_leaves_the_server_default_alone():
    stub = StubResponses()
    OpenAIPlayer(prompt="RULES", client=stub).choose("BOARD", state())
    assert "reasoning" not in stub.calls[0]


def test_reasoning_and_bare_modes_get_different_output_budgets():
    thinking = StubResponses()
    OpenAIPlayer(prompt="RULES", effort="high", client=thinking).choose(
        "BOARD", state()
    )
    assert thinking.calls[0]["max_output_tokens"] == 5000

    bare = StubResponses()
    OpenAIPlayer(prompt="RULES", effort="none", client=bare).choose("BOARD", state())
    assert bare.calls[0]["max_output_tokens"] == 32


def test_top_k_is_only_sent_when_configured():
    """It is not in the OpenAI schema: self-hosted servers take it, hosted ones
    reject the request outright, so it cannot be a default."""
    off = StubResponses()
    OpenAIPlayer(prompt="RULES", client=off).choose("BOARD", state())
    assert "extra_body" not in off.calls[0]

    on = StubResponses()
    OpenAIPlayer(prompt="RULES", top_k=20, client=on).choose("BOARD", state())
    assert on.calls[0]["extra_body"] == {"top_k": 20}


def test_unknown_effort_levels_are_rejected():
    with pytest.raises(ValueError, match="unknown effort"):
        OpenAIPlayer(prompt="RULES", effort="enormous", client=object())


def test_it_shares_the_prompt_format_with_the_claude_player():
    """Same board presentation, or the comparison between models is not fair."""
    from zeos_space_invaders.players import ClaudePlayer

    turn = state(step=2, score=10, lives=1, can_shoot=False)
    local = OpenAIPlayer(prompt="RULES", client=StubResponses())
    claude = ClaudePlayer(prompt="RULES", client=object())  # never called
    assert local.render_turn("BOARD", turn) == claude.render_turn("BOARD", turn)


# --- reading the reply ------------------------------------------------------


def test_reasoning_cannot_reach_the_action_parser():
    """Typed items make it impossible for the parser to read a move out of the
    reasoning."""
    stub = StubResponses(
        {
            "output": [
                reasoning(verbatim="I could go right, but shoot is better"),
                message("left"),
            ]
        }
    )
    action, record = OpenAIPlayer(prompt="RULES", client=stub).choose("BOARD", state())
    assert action == "left"  # not "right", not "shoot"
    assert "shoot is better" in record.reasoning


def test_a_summary_only_reasoning_item_is_still_captured():
    stub = StubResponses(
        {"output": [reasoning(summary="weighed left against shoot"), message("shoot")]}
    )
    action, record = OpenAIPlayer(prompt="RULES", client=stub).choose("BOARD", state())
    assert action == "shoot"
    assert record.reasoning == "weighed left against shoot"


def test_a_reply_with_no_message_item_is_flagged_unparseable():
    stub = StubResponses(
        {
            "output": [reasoning(verbatim="still deciding")],
            "reason": "max_output_tokens",
            "usage": usage(reasoning_tokens=5000),
        }
    )
    action, record = OpenAIPlayer(prompt="RULES", effort="high", client=stub).choose(
        "BOARD", state()
    )
    assert record.parsed is False
    assert action == "left"  # the documented fallback
    assert record.reasoning  # but the reasoning is still logged


def test_budget_exhaustion_retries_at_effort_none_instead_of_falling_back():
    """A reasoner that loops to the token cap should still get to pick a move."""
    stub = StubResponses(
        {
            "output": [reasoning(verbatim="looping forever")],
            "reason": "max_output_tokens",
            "usage": usage(reasoning_tokens=5000, output_tokens=5000),
        },
        {"output": [message("right")], "usage": usage(output_tokens=2)},
    )
    action, record = OpenAIPlayer(prompt="RULES", effort="high", client=stub).choose(
        "BOARD", state()
    )
    assert action == "right"  # a real choice, not the blind fallback
    assert record.parsed is True
    assert record.retried is True
    assert stub.calls[0]["reasoning"] == {"effort": "high"}
    assert stub.calls[1]["reasoning"] == {"effort": "none"}
    assert record.usage["output_tokens"] == 5002  # both calls counted


def test_a_normal_reasoning_reply_is_not_retried():
    stub = StubResponses(
        {"output": [reasoning(verbatim="thought about it"), message("left")]}
    )
    _, record = OpenAIPlayer(prompt="RULES", effort="high", client=stub).choose(
        "BOARD", state()
    )
    assert record.retried is False
    assert len(stub.calls) == 1


# --- the assertion that `none` was honoured ---------------------------------


def still_reasoning():
    """A server that reasons whatever it was asked."""
    return StubResponses(
        {
            "output": [reasoning(verbatim="thinking out loud")],
            "reason": "max_output_tokens",
            "usage": usage(reasoning_tokens=15),
        },
        {"output": [message("right")]},
    )


def test_a_server_that_ignores_effort_none_stops_the_run():
    """The failure this check exists for: scoring a model that never stopped
    reasoning is scoring nothing at all."""
    player = OpenAIPlayer(prompt="RULES", effort="none", client=still_reasoning())
    with pytest.raises(ThinkingNotDisabled, match="reasoned anyway"):
        player.choose("BOARD", state())


def test_strict_false_scores_it_anyway():
    player = OpenAIPlayer(
        prompt="RULES", effort="none", strict=False, client=still_reasoning()
    )
    action, _ = player.choose("BOARD", state())
    assert action in ("left", "right", "shoot")


def test_a_real_effort_level_is_not_asserted_against():
    player = OpenAIPlayer(prompt="RULES", effort="high", client=still_reasoning())
    player.choose("BOARD", state())  # reasoning is expected here


def test_a_missing_key_falls_back_to_the_local_server_convention():
    """Self-hosted servers ignore the key but the SDK insists on one."""
    import os

    from zeos_space_invaders.players.openai_compat import default_api_key

    os.environ.pop("OPENAI_API_KEY", None)
    assert default_api_key() == "EMPTY"


def test_a_reply_with_no_usage_reports_none_rather_than_failing():
    """Not every server fills it in, and a run must not die over accounting."""
    from types import SimpleNamespace

    assert OpenAIPlayer._usage(SimpleNamespace()) == {}
    assert OpenAIPlayer._usage(SimpleNamespace(usage=None)) == {}


# --- streaming --------------------------------------------------------------


def test_a_streamed_reply_arrives_in_pieces_and_ends_where_the_whole_one_does():
    """Both paths have to agree, or `--no-stream` stops being a control."""
    turn = {"output": [reasoning(verbatim="hmm ok"), message("go left")]}
    streaming = OpenAIPlayer(prompt="RULES", effort="high", client=StubResponses(turn))
    lump = OpenAIPlayer(prompt="RULES", effort="high", client=StubResponses(turn))

    pieces = []
    stream = streaming.respond_stream("BOARD", "high")
    while True:
        try:
            pieces.append(next(stream))
        except StopIteration as done:
            got = done.value
            break

    assert len(pieces) > 2, "the whole reply arrived as one piece"
    assert got == lump.respond("BOARD", "high")
    assert "".join(t for c, t in pieces if c == "answer").strip() == "go left"
    assert "".join(t for c, t in pieces if c == "thinking").strip() == "hmm ok"


def test_a_stream_that_never_completes_says_so_rather_than_claiming_success():
    """A truncated connection is not a clean run."""
    player = OpenAIPlayer(
        prompt="RULES",
        effort="none",
        client=StubResponses({"output": [message("left")]}),
    )
    player.client.responses = player.client
    player.client.create = lambda **params: iter(())

    stream = player.respond_stream("BOARD", "none")
    with pytest.raises(StopIteration) as done:
        while True:
            next(stream)
    assert done.value.value == ("", "", "incomplete", {})


def test_the_board_is_sent_as_the_input_field():
    """One builder for every path, so they cannot drift on sampling or effort."""
    player = OpenAIPlayer(
        prompt="RULES",
        effort="none",
        client=StubResponses({"output": [message("left")]}),
    )
    list(player.respond_stream("THE BOARD", "none"))
    assert player.client.calls[0]["input"] == "THE BOARD"


# --- one endpoint table, so the two arms cannot drift -----------------------


@pytest.mark.parametrize(
    ("vendor", "player_side", "machine_side"),
    [
        ("openai", "openai_compat", "api_openai"),
        ("claude", "claude", "api_claude"),
    ],
)
def test_the_player_and_the_machine_resolve_to_the_same_endpoint(
    vendor, player_side, machine_side, monkeypatch
):
    """Asserted with a bare environment, because a `.env` setting both variables
    masks any divergence between the two tables."""
    import importlib

    for suffix in ("MODEL", "BASE_URL", "API_KEY"):
        monkeypatch.delenv(
            f"{'OPENAI' if vendor == 'openai' else 'ANTHROPIC'}_{suffix}", raising=False
        )
    player = importlib.import_module(f"zeos_space_invaders.players.{player_side}")
    machine = importlib.import_module(
        f"zeos_space_invaders.players.zeos.{machine_side}"
    )
    built = (
        machine.OpenAIAPIMachine() if vendor == "openai" else machine.ClaudeAPIMachine()
    )

    assert built.model == player.default_model()
    assert built.base_url == player.default_base_url()


def test_a_flag_beats_the_environment_which_beats_the_fallback(monkeypatch):
    """The order `utils/config.py` states for the whole demo, asserted on the one
    table that now implements it for every caller."""
    from zeos_space_invaders.players.endpoints import OPENAI

    monkeypatch.setenv("OPENAI_MODEL", "from-the-environment")
    assert OPENAI.resolve().model == "from-the-environment"
    assert OPENAI.resolve(model="from-a-flag").model == "from-a-flag"

    monkeypatch.setenv("OPENAI_BASE_URL", "")  # `.env.example` ships it valueless
    assert OPENAI.resolve().base_url == OPENAI.base_url, "empty must mean unset"


def test_both_arms_ask_the_endpoint_for_the_same_thing(monkeypatch):
    """A comparison labelled "same model, same effort" has to send the same
    sampling and the same reasoning switch from both arms."""
    from stubs import StubChatClient

    from zeos_space_invaders.players.zeos import build_machine

    monkeypatch.delenv("OPENAI_TOP_K", raising=False)
    loop = OpenAIPlayer(prompt="RULES", effort="none", client=object())
    machine = build_machine("openai", effort="none", client=StubChatClient())

    assert (machine.temperature, machine.top_p, machine.top_k) == (
        loop.temperature,
        loop.top_p,
        loop.top_k,
    ), "the two arms sample differently"

    sent = machine._params([{"role": "user", "content": "BOARD"}], continuing=False)
    asked = loop._params("BOARD", "none")
    assert sent["reasoning_effort"] == "none", "the machine asked for reasoning"
    assert asked["reasoning"] == {"effort": "none"}
    assert (sent["temperature"], sent["top_p"]) == (
        asked["temperature"],
        asked["top_p"],
    )


def test_none_is_spelt_the_way_each_backend_spells_it():
    """`none` is not Anthropic's vocabulary -- `output_config.effort` takes
    low..max -- so it is sent as thinking off."""
    from zeos_space_invaders.players.zeos import build_machine

    claude = build_machine("claude", effort="none", client=object())
    assert claude.thinking is False and claude.effort is None
