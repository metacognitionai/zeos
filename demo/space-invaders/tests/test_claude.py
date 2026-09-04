# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""ClaudePlayer against a stub Anthropic client — no key needed.

Only what is specific to this backend: the thinking parameter, the token
budgets, and the prompt cache. What both backends share is in
`test_prompt_player.py`.
"""

import json

import pytest
from stubs import StubClient, claude_player, state


def test_thinking_mode_sends_the_thinking_parameter():
    p = claude_player(effort="high")
    p.choose("BOARD", state())
    sent = p.client.calls[0]
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert sent["max_tokens"] == 4096


def test_no_thinking_mode_says_disabled_rather_than_saying_nothing():
    """Sonnet 5 and Opus 5 think by default, so a request that leaves `thinking`
    out reasons through the whole budget and returns no text."""
    p = claude_player(effort="none")
    p.choose("BOARD", state())
    sent = p.client.calls[0]
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["max_tokens"] == 64


def test_thinking_tokens_are_what_the_effort_none_guard_reads():
    """The guard is only as good as what it counts, and this backend reports
    reasoning in a field of its own even when the thinking block is empty."""
    from zeos_space_invaders.players import ClaudePlayer, ThinkingNotDisabled

    usage = {"output_tokens": 64, "output_tokens_details": {"thinking_tokens": 64}}
    assert ClaudePlayer.reasoning_tokens(usage) == 64
    assert ClaudePlayer.reasoning_tokens({"output_tokens": 3}) == 0

    with pytest.raises(ThinkingNotDisabled, match="64 reasoning tokens"):
        claude_player(effort="none").check_no_reasoning("", usage)


def test_only_the_system_prompt_is_cached():
    """The board changes every turn; caching the last block would never hit."""
    p = claude_player()
    p.choose("BOARD", state())
    sent = p.client.calls[0]
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent
    assert "cache_control" not in sent["messages"][0]


def test_the_system_prompt_is_byte_identical_across_turns():
    """Any drift in the prefix silently destroys the cache hit rate."""
    p = claude_player()
    for step in range(4):
        p.choose(f"BOARD {step}", state(step=step))
    assert len({json.dumps(c["system"]) for c in p.client.calls}) == 1


def test_reasoning_is_captured_but_never_parsed():
    p = claude_player("shoot", effort="high", think="I should move left of it")
    action, record = p.choose("BOARD", state())
    assert action == "shoot"  # not misread from the reasoning
    assert record.reasoning == "I should move left of it"


def test_usage_is_reported_as_the_sdk_gives_it():
    p = claude_player("left")
    _, record = p.choose("BOARD", state())
    assert record.usage == {"input_tokens": 100, "output_tokens": 5}


def test_a_client_is_only_built_when_none_was_handed_in():
    """Every test in the suite injects one; a real run has the SDK build it."""
    p = claude_player("left")
    assert isinstance(p.client, StubClient)


def test_the_model_and_key_come_from_the_sdk_s_own_variables(monkeypatch):
    """Sharing the SDK's names is why a Claude Code session's endpoint is
    inherited -- the banner prints where a run is pointing for that reason."""
    from zeos_space_invaders.players import claude

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-from-the-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://inherited:4000")
    assert claude.default_model() == "claude-from-the-env"
    assert claude.default_api_key() == "sk-test"
    assert claude.default_base_url() == "http://inherited:4000"

    monkeypatch.delenv("ANTHROPIC_MODEL")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_BASE_URL")
    assert claude.default_model() == claude.DEFAULT_MODEL
    assert claude.default_api_key() is None
    assert claude.default_base_url() is None


# --- streaming --------------------------------------------------------------


def test_a_streamed_reply_arrives_in_pieces_with_the_channels_kept_apart():
    """The action parser must never see the thinking. See `base.respond_stream`."""
    player = claude_player("go left", think="hmm ok")

    pieces = []
    stream = player.respond_stream("BOARD", "high")
    while True:
        try:
            pieces.append(next(stream))
        except StopIteration as done:
            text, reasoning, stop_reason, usage = done.value
            break

    assert len(pieces) > 2, "the whole reply arrived as one piece"
    assert text.strip() == "go left" and reasoning.strip() == "hmm ok"
    assert stop_reason == "end_turn"
    assert usage == {"input_tokens": 100, "output_tokens": 5}
    assert player.client.calls[0]["stream"] is True


def test_a_usage_the_sdk_did_not_fill_in_is_no_usage_rather_than_a_crash():
    from zeos_space_invaders.players.claude import _usage

    assert _usage(None) == {}
    assert _usage({"input_tokens": 1}) == {"input_tokens": 1}
