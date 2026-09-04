# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Any endpoint speaking the OpenAI Responses API, local or hosted.

Responses rather than chat completions because reasoning arrives as typed output
items instead of a field whose name varies by server version, and because effort
is a first-class parameter rather than a chat-template argument smuggled through
`extra_body`. `effort` is one of EFFORTS, or None to leave the server's default
alone; `none` means do not reason, and is checked rather than trusted, because a
server that ignores it scores whatever a truncated reply parses as.
"""

from .base import EFFORTS, PromptPlayer, ThinkingNotDisabled, sdk
from .endpoints import OPENAI
from .sampling import sampling_for, top_k_from_env
from .sampling import thinking as is_thinking

#: Kept as names because the CLI banner and `compare` read them; the values are
#: `endpoints.OPENAI`'s.
DEFAULT_BASE_URL = OPENAI.base_url
DEFAULT_MODEL = OPENAI.model

__all__ = ["EFFORTS", "OpenAIPlayer", "ThinkingNotDisabled"]


def default_base_url():
    return OPENAI.resolve().base_url


def default_model():
    return OPENAI.resolve().model


def default_api_key():
    return OPENAI.resolve().api_key


#: Kept as a name because the tests and the CLI read it; the value is
#: `sampling.top_k_from_env`'s, so both arms of a comparison ask for the same.
def default_top_k():
    return top_k_from_env()


# Reasoning counts against max_output_tokens, so a budget the reasoning outruns
# reads as a stupid model rather than a truncated one; this plus the prompt has
# to fit inside the server's context length.
MAX_TOKENS_THINKING = 5000
MAX_TOKENS_BARE = 32


class OpenAIPlayer(PromptPlayer):
    def __init__(
        self,
        prompt=None,
        base_url=None,
        model=None,
        effort=None,
        history=5,
        temperature=None,
        top_p=None,
        top_k=None,
        api_key=None,
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
        self.model = model or default_model()
        self.effort = effort
        self.thinking = is_thinking(effort)
        self.strict = strict
        # One table, shared with the machine the kernel decodes through, so a run
        # labelled `--effort none` is the same request on both arms.
        sampling = sampling_for(effort)
        self.temperature = sampling.temperature if temperature is None else temperature
        self.top_p = sampling.top_p if top_p is None else top_p
        self.top_k = sampling.top_k if top_k is None else top_k
        if client is not None:
            self.client = client
        else:
            self.client = sdk("openai", "openai").OpenAI(
                base_url=base_url or default_base_url(),
                api_key=api_key or default_api_key(),
            )

    def choose(self, obs, info):
        text, reasoning, status, usage = self._ask(obs, info, self.effort)
        if self.strict and self.effort == "none":
            self.check_no_reasoning(reasoning, usage)
        retried = False
        if not text and status == "max_output_tokens":
            # Reasoning ate the whole budget; a second, effort-free ask beats the
            # blind fallback.
            retried = True
            text, _, status, retry_usage = self._ask(obs, info, "none")
            for key, value in retry_usage.items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        action, decision = self.record(obs, info, text, reasoning, status, usage)
        decision.retried = retried
        return action, decision

    @staticmethod
    def reasoning_tokens(usage):
        return (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0

    def _ask(self, obs, info, effort):
        return self.respond(self.render_turn(obs, info), effort)

    def _params(self, rendered, effort):
        """The request, built once for every path through this backend.

        One builder rather than several, so `--stream` stays the only difference
        between the runs it is supposed to be the only difference between.
        """
        # Keyed on this call's effort, which the effort-none retry makes differ
        # from the one the player was built with.
        thinking = is_thinking(effort)
        bare = sampling_for(effort)
        params = {
            "model": self.model,
            "instructions": self.prompt,
            "input": rendered,
            "max_output_tokens": MAX_TOKENS_THINKING if thinking else MAX_TOKENS_BARE,
            "temperature": self.temperature if thinking else bare.temperature,
            "top_p": self.top_p if thinking else bare.top_p,
        }
        if effort is not None:
            params["reasoning"] = {"effort": effort}
        if self.top_k is not None:
            params["extra_body"] = {"top_k": self.top_k}
        return params

    def respond(self, rendered, effort):
        response = self.client.responses.create(**self._params(rendered, effort))

        text, thought = self._split(response)
        return text, thought, self._status(response), self._usage(response)

    def respond_stream(self, rendered, effort):
        """The Responses API in streaming mode. See `base.respond_stream`.

        Reasoning and answer arrive as separately typed deltas, the same seam
        `_split` reads off a completed response, so the two paths agree on what
        counts as the answer.
        """
        if not self.stream:
            return (yield from super().respond_stream(rendered, effort))

        params = self._params(rendered, effort)
        params["stream"] = True
        answer, thought, final = [], [], None
        for event in self.client.responses.create(**params):
            kind = getattr(event, "type", "")
            delta = getattr(event, "delta", None)
            if kind == "response.output_text.delta" and delta:
                answer.append(delta)
                yield "answer", delta
            elif (
                kind.startswith("response.reasoning")
                and kind.endswith(".delta")
                and delta
            ):
                # Both `reasoning_text` and `reasoning_summary_text`: a hosted
                # API releases only a summary.
                thought.append(delta)
                yield "thinking", delta
            elif kind in ("response.completed", "response.incomplete"):
                final = getattr(event, "response", None)
        text = "".join(answer).strip()
        thinking = "".join(thought).strip()
        if final is None:
            # No terminal event means no known reason, and saying so beats
            # reporting a clean completion never seen.
            return text, thinking, "incomplete", {}
        return text, thinking, self._status(final), self._usage(final)

    @staticmethod
    def _split(response):
        """Answer and reasoning, read off the typed output items.

        Servers that return the chain of thought put it in the reasoning item's
        `content` as `reasoning_text`; hosted APIs that release only a summary
        put that in `summary`. Neither can leak into the answer the way an
        inline `<think>` block would.
        """
        answer, thought = [], []
        for item in getattr(response, "output", None) or []:
            kind = getattr(item, "type", None)
            if kind == "message":
                answer += [
                    c.text
                    for c in (item.content or [])
                    if getattr(c, "type", None) == "output_text"
                ]
            elif kind == "reasoning":
                thought += [
                    c.text
                    for c in (getattr(item, "content", None) or [])
                    if getattr(c, "type", None) == "reasoning_text"
                ]
                thought += [s.text for s in (getattr(item, "summary", None) or [])]
        return "\n".join(answer).strip(), "\n".join(thought).strip()

    @staticmethod
    def _status(response):
        """Why generation ended: 'completed', or the reason it did not."""
        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            return getattr(details, "reason", None) or "incomplete"
        return getattr(response, "status", None) or "completed"

    @staticmethod
    def _usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
