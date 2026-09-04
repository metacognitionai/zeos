# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The OpenAI-compatible backend for `api_machine.APIMachineBase`.

One vendor, one wire format, one continuation policy; everything else is in the
base class. It speaks chat completions rather than Responses because that is what
served models (vLLM, SGLang, Ollama, LM Studio) speak. The syscall channel is
enforced with `response_format` and the strict schema in `json` mode and by the
prompt alone in `text` mode. There is no tool-calling format: a server that
parses tool calls after the fact enforces nothing a content schema does not.

A half-finished reply goes out as a trailing assistant message. OpenAI's endpoint
restarts the turn; vLLM and SGLang accept `continue_final_message` as a true
prefill. Undetectable, so it is a flag, off because OpenAI 400s on unknown fields.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..base import SDKMissing
from ..endpoints import OPENAI
from ..sampling import sampling_for
from .api_machine import APIMachineBase, Piece

__all__ = ["OpenAIAPIMachine", "SDKMissing"]

#: A cancelled producer only leaves the graveyard by returning, so a server that
#: holds the connection open must not be allowed to park one there for ever.
DEFAULT_TIMEOUT = 120.0


class OpenAIAPIMachine(APIMachineBase):
    """A ZEOS machine backed by any endpoint speaking the OpenAI chat API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        continue_final_message: bool = False,
        client: Any = None,
        **machine_kwargs: Any,
    ) -> None:
        super().__init__(**machine_kwargs)
        # Resolved through the same table the prompt-loop player uses, so the two
        # arms of a comparison cannot pick different models out of one shell.
        endpoint = OPENAI.resolve(model=model, base_url=base_url, api_key=api_key)
        self.model = endpoint.model
        self.base_url = endpoint.base_url
        #: Effort and its sampling come from the table `players/openai_compat.py`
        #: reads, so the two arms of one comparison ask the endpoint for the same.
        self.effort = effort
        sampling = sampling_for(effort)
        self.temperature = sampling.temperature if temperature is None else temperature
        self.top_p = sampling.top_p if top_p is None else top_p
        self.top_k = sampling.top_k
        self.timeout = timeout
        self.continue_final_message = continue_final_message
        self._api_key = endpoint.api_key
        #: Injected by a caller with its own client, which needs no SDK installed.
        self._client = client
        #: What the server said it charged; a cancelled generation reports none
        #: while having cost real prefill.
        self.usage: dict[str, int] = {}

    # -- the client ----------------------------------------------------------

    @property
    def client(self) -> Any:
        """The SDK client, built on first use and not at import.

        `openai` is an optional extra, so naming it at module scope would put a
        vendor dependency on runs that never reach a model at all.
        """
        if self._client is None:
            try:
                import openai
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.split(".")[0] != "openai":
                    # The SDK is installed and something it imports is not: a
                    # broken environment, not a missing extra.
                    raise
                raise SDKMissing(
                    "this machine needs the openai SDK, which is not installed: "
                    "run `uv sync --extra openai`"
                ) from exc
            self._client = openai.OpenAI(
                base_url=self.base_url, api_key=self._api_key, timeout=self.timeout
            )
        return self._client

    # -- the request ---------------------------------------------------------

    def _params(
        self, messages: list[dict[str, str]], *, continuing: bool
    ) -> dict[str, Any]:
        """The request, built in one place so sampling parameters cannot drift."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self._max_tokens,
            "stream": True,
            # Without it a streamed request reports no usage at all.
            "stream_options": {"include_usage": True},
        }
        if self.effort is not None:
            # The chat-completions spelling of the Responses API's `reasoning.effort`.
            params["reasoning_effort"] = self.effort
        if self.top_k is not None:
            params.setdefault("extra_body", {})["top_k"] = self.top_k
        if self.syscall_format == "json":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": self.syscall_schema(),
            }
        if continuing and self.continue_final_message:
            # vLLM and SGLang only; `add_generation_prompt` has to go false with
            # it, or the template closes the turn the model is asked to continue.
            params.setdefault("extra_body", {}).update(
                {"continue_final_message": True, "add_generation_prompt": False}
            )
        return params

    # -- the vendor seam -----------------------------------------------------

    def _request(self, ctx: Any, *, continuing: bool) -> dict[str, Any]:
        """The request, built on the kernel's thread; see `APIMachineBase._request`."""
        return self._params(self._messages(ctx), continuing=continuing)

    def _stream(self, request: object, descriptor: str, gen: Any) -> Iterator[Piece]:
        """Issue the request and yield typed pieces, closing the stream on exit.

        Runs on the producer thread, the only one permitted to close the stream.
        Reasoning is marked so it reaches `T` without touching the syscall
        extractor, or a model musing that a write would be wrong could be read as
        having issued one. `descriptor` is unused: one model serves every job.
        """
        stream = self.client.chat.completions.create(**request)  # type: ignore[arg-type]
        # Published for `_abort`, which is the only other thread that touches it.
        gen.wire = stream
        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    self._add_usage(usage)
                if not chunk.choices:
                    # The usage chunk, and any keep-alive: no content, not an end.
                    continue
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning_content", None)
                if thinking:
                    yield Piece("thinking", str(thinking))
                text = getattr(delta, "content", None)
                if text:
                    yield Piece("text", str(text))
        finally:
            gen.wire = None
            stream.close()

    def _add_usage(self, usage: Any) -> None:
        """Sum what the server reported, on the producer thread only.

        `cached_tokens` is the provider's own account of how much prefix it did not
        have to prefill, the closest measure of what a context change costs.
        """
        cached = getattr(usage, "prompt_tokens_details", None)
        for key, value in (
            ("prompt_tokens", getattr(usage, "prompt_tokens", 0)),
            ("completion_tokens", getattr(usage, "completion_tokens", 0)),
            ("cached_tokens", getattr(cached, "cached_tokens", 0) if cached else 0),
        ):
            if value:
                self.usage[key] = self.usage.get(key, 0) + int(value)
