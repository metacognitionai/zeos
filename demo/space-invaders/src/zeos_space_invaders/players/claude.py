# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""One action per step, over the Claude API.

`effort` is one of EFFORTS, or None for the API default. `none` means answer
from a single forward pass with no thinking block at all; everything else is
adaptive thinking at that depth. Nothing else differs between the two, so a pair
of runs isolates what the reasoning is worth.
"""

from .base import EFFORTS, PromptPlayer, sdk
from .endpoints import ANTHROPIC

#: `endpoints.ANTHROPIC`'s, kept as a name because the CLI reads it.
DEFAULT_MODEL = ANTHROPIC.model


# `minimal` is an OpenAI level the Claude API does not take.
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Thinking tokens count against this.
MAX_TOKENS_THINKING = 4096
MAX_TOKENS_BARE = 64


def default_model():
    return ANTHROPIC.resolve().model


def default_api_key():
    return ANTHROPIC.resolve().api_key


def default_base_url():
    """Empty means the official endpoint. See `endpoints.ANTHROPIC`."""
    return ANTHROPIC.resolve().base_url


class ClaudePlayer(PromptPlayer):
    def __init__(
        self,
        prompt=None,
        model=None,
        effort=None,
        history=5,
        api_key=None,
        base_url=None,
        client=None,
        view=None,
        strict=True,
        stream=True,
        rules=None,
    ):
        super().__init__(
            prompt=prompt, history=history, view=view, stream=stream, rules=rules
        )
        if effort is not None and effort not in EFFORTS:
            raise ValueError(
                f"unknown effort {effort!r}; expected one of {', '.join(EFFORTS)}"
            )
        if effort not in (None, "none") and effort not in CLAUDE_EFFORTS:
            raise ValueError(
                f"the Claude API does not take effort {effort!r}; "
                f"it accepts {', '.join(('none', *CLAUDE_EFFORTS))}"
            )
        self.model = model or default_model()
        self.effort = effort
        self.strict = strict
        if client is not None:
            self.client = client
        else:
            self.client = sdk("anthropic", "claude").Anthropic(
                api_key=api_key or default_api_key(),
                base_url=base_url or default_base_url(),
            )

    def choose(self, obs, info):
        """Return (action, record)."""
        text, reasoning, stop_reason, usage = self.respond(
            self.render_turn(obs, info), self.effort
        )
        if self.strict and self.effort == "none":
            self.check_no_reasoning(reasoning, usage)
        return self.record(obs, info, text, reasoning, stop_reason, usage)

    @staticmethod
    def reasoning_tokens(usage):
        """What this backend spent thinking, whatever it says it did.

        `check_no_reasoning` is only as good as this: without it the guard sees
        an empty thinking block, reads zero, and lets a run through that reasoned
        for its whole budget -- exactly the failure it exists to catch.
        """
        return (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0

    def _params(self, rendered, effort):
        """The request, built once for both the streamed and unstreamed paths.

        One builder rather than two, so `--stream` stays the only difference
        between the runs it is supposed to be the only difference between.
        """
        thinking = effort != "none"
        params = {
            "model": self.model,
            "max_tokens": MAX_TOKENS_THINKING if thinking else MAX_TOKENS_BARE,
            # Cached explicitly: the top-level auto form would target the last
            # block, which is the board and changes every turn.
            "system": [
                {
                    "type": "text",
                    "text": self.prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": rendered}],
        }
        if thinking:
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
        else:
            # Said, not left unsaid: thinking is on by default, so omitting the
            # parameter spends the whole bare budget reasoning and returns no
            # text, which scores as the fallback move.
            params["thinking"] = {"type": "disabled"}
        if effort not in (None, "none"):
            params["output_config"] = {"effort": effort}
        return params

    def respond(self, rendered, effort):
        """One request against an already-rendered board. See `base.respond`."""
        response = self.client.messages.create(**self._params(rendered, effort))

        text = "".join(b.text for b in response.content if b.type == "text")
        reasoning = "\n".join(
            b.thinking for b in response.content if b.type == "thinking"
        )
        return text, reasoning, response.stop_reason, _usage(response.usage)

    def respond_stream(self, rendered, effort):
        """The Messages API in streaming mode. See `base.respond_stream`.

        Thinking and text arrive as separately typed deltas inside
        `content_block_delta`, so the answer parser never has to guess which of
        the two it is looking at.
        """
        if not self.stream:
            return (yield from super().respond_stream(rendered, effort))

        params = self._params(rendered, effort)
        params["stream"] = True
        answer, thought = [], []
        stop_reason, totals = None, {}
        for event in self.client.messages.create(**params):
            kind = getattr(event, "type", "")
            if kind == "content_block_delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", "") == "text_delta":
                    answer.append(delta.text)
                    yield "answer", delta.text
                elif getattr(delta, "type", "") == "thinking_delta":
                    thought.append(delta.thinking)
                    yield "thinking", delta.thinking
            elif kind == "message_start":
                totals.update(_usage(getattr(event.message, "usage", None)))
            elif kind == "message_delta":
                # Output totals land here, on the event that also carries the
                # stop reason; input totals arrived with `message_start`.
                stop_reason = getattr(event.delta, "stop_reason", None) or stop_reason
                totals.update(_usage(getattr(event, "usage", None)))
        # Both joined without a separator: these are fragments of one block,
        # not the several blocks `respond` puts a newline between.
        return "".join(answer), "".join(thought), stop_reason, totals


def _usage(reported):
    """A usage object as a plain dict, whichever shape the SDK handed back."""
    if reported is None:
        return {}
    return reported.model_dump() if hasattr(reported, "model_dump") else dict(reported)
