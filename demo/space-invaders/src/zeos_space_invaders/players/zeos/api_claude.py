# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The Anthropic backend for `api_machine.APIMachineBase`.

Same machine, second wire format; only where the Messages API differs from chat
completions is here. The descriptor body is a top-level `system` block carrying
the one `cache_control` marker, because it is the only part of a job's context
that never changes. The model refuses assistant prefill, so a turn boundary is
machine framing: `CONTINUATION` closes the turn and opens a new one, and never
enters `T`. Sampling parameters do not exist; effort is `output_config.effort`.

Thinking is disabled explicitly, because omitting the switch is not turning it
off. With it on, reasoning arrives as `thinking_delta` and enters `T` as `THINK`,
charged to the budget and never transmitted: a thinking block needs the signature
that closes it, which a turn the kernel cut short never produced.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..base import SDKMissing
from ..endpoints import ANTHROPIC
from .api_machine import IN, OUT, APIMachineBase, Piece, _Context

__all__ = ["ClaudeAPIMachine", "SDKMissing"]

#: `endpoints.ANTHROPIC`'s, kept as a name because the tests read it.
DEFAULT_MODEL = ANTHROPIC.model
DEFAULT_TIMEOUT = 120.0

#: The turn boundary: inert because no job wrote it, constant so the prefix
#: stays cacheable, and never part of `T`.
CONTINUATION = "Carry on."


class ClaudeAPIMachine(APIMachineBase):
    """A ZEOS machine backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        thinking_display: str = "summarized",
        effort: str | None = None,
        cache: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        client: Any = None,
        **machine_kwargs: Any,
    ) -> None:
        super().__init__(**machine_kwargs)
        # Resolved through the same table `players/claude.py` uses, so the two
        # arms of a comparison cannot pick different models out of one shell.
        endpoint = ANTHROPIC.resolve(model=model, base_url=base_url, api_key=api_key)
        self.model = endpoint.model
        self.base_url = endpoint.base_url
        # `"none"` is the CLI's vocabulary, not Anthropic's: `output_config.effort`
        # takes low..max, so "do not reason" is spelt the way this backend spells it.
        if effort == "none":
            thinking, effort = False, None
        #: Said explicitly, because Sonnet 5 reasons by default.
        self.thinking = thinking
        #: Asked for explicitly: `adaptive` with no `display` returns a thinking
        #: block whose text is empty, so `T` would learn nothing of the reasoning.
        self.thinking_display = thinking_display
        #: The only lever on Sonnet 5: `budget_tokens` is refused, and `adaptive`
        #: alone lets the model decide not to think at all.
        self.effort = effort
        self.cache = cache
        self.timeout = timeout
        #: `None` when nothing is configured, so the SDK raises naming the variable
        #: instead of a placeholder key becoming a 401 further from the cause.
        self._api_key = endpoint.api_key
        self._client = client
        #: What the server said it charged; `cache_read_input_tokens` is the
        #: provider's own account of how much prefix it did not have to prefill.
        self.usage: dict[str, int] = {}

    # -- the client ----------------------------------------------------------

    @property
    def client(self) -> Any:
        """The SDK client, built on first use and not at import.

        `anthropic` is an optional extra, so naming it at module scope would put a
        vendor dependency on runs that never reach a model at all.
        """
        if self._client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.split(".")[0] != "anthropic":
                    # The SDK is installed and something it imports is not: a
                    # broken environment, not a missing extra.
                    raise
                raise SDKMissing(
                    "this machine needs the anthropic SDK, which is not "
                    "installed: run `uv sync --extra claude`"
                ) from exc
            self._client = anthropic.Anthropic(
                base_url=self.base_url,
                api_key=self._api_key or None,
                timeout=self.timeout,
            )
        return self._client

    # -- the wire format -----------------------------------------------------

    def _wire(self, ctx: _Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """`(system, messages)` for the context as it stands.

        The first injected run is the descriptor body and becomes the `system`
        block with the cache breakpoint. `messages` may not be empty and may not
        end with an assistant turn, so whenever the transcript does not end with
        something said *to* the job the machine closes the turn itself.
        """
        runs = self._runs(ctx)
        system: list[dict[str, Any]] = []
        body: list[tuple[str, str]] = list(runs)
        if body and body[0][0] == IN:
            text = body.pop(0)[1]
            block: dict[str, Any] = {"type": "text", "text": text}
            if self.cache:
                block["cache_control"] = {"type": "ephemeral"}
            system = [block]
        messages: list[dict[str, Any]] = [
            {"role": "assistant" if origin == OUT else "user", "content": text}
            for origin, text in body
        ]
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": CONTINUATION})
        return system, messages

    def _thinking(self) -> dict[str, Any]:
        """The thinking block, said explicitly: omitting it is not turning it off."""
        if not self.thinking:
            return {"type": "disabled"}
        return {"type": "adaptive", "display": self.thinking_display}

    def _params(self, ctx: _Context) -> dict[str, Any]:
        """The request, built in one place for every path through this class."""
        system, messages = self._wire(ctx)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            # Explicit, because omitting it is not the same as disabling it.
            "thinking": self._thinking(),
        }
        if system:
            params["system"] = system
        output: dict[str, Any] = {}
        if self.syscall_format == "json":
            # Anthropic's structured outputs take the bare schema; the name and the
            # strict flag OpenAI wants have no counterpart here.
            output["format"] = {
                "type": "json_schema",
                "schema": self.syscall_schema()["schema"],
            }
        if self.effort:
            output["effort"] = self.effort
        if output:
            params["output_config"] = output
        return params

    # -- the vendor seam -----------------------------------------------------

    def _request(self, ctx: _Context, *, continuing: bool) -> dict[str, Any]:
        """The request, built on the kernel's thread; `_wire` closes the turn."""
        return self._params(ctx)

    def _stream(self, request: object, descriptor: str, gen: Any) -> Iterator[Piece]:
        """Yield typed pieces, and close the HTTP stream on the way out.

        Runs on the producer thread, the only one permitted to close the stream.
        Reasoning is marked so it reaches `T` without touching the syscall
        extractor; `signature_delta` seals a block this backend never sends back.
        """
        stream = self.client.messages.stream(**request)  # type: ignore[arg-type]
        with stream as events:
            # Published for `_abort`; inside the `with`, so what is published is
            # the stream rather than its manager.
            gen.wire = events
            try:
                yield from self._pieces(events)
            finally:
                gen.wire = None

    def _pieces(self, events: Any) -> Iterator[Piece]:
        """The delta loop, split out so the publication above can wrap it."""
        for event in events:
            if event.type == "message_delta":
                self._add_usage(getattr(event, "usage", None))
                continue
            if event.type != "content_block_delta":
                continue
            delta = event.delta
            if delta.type == "text_delta":
                yield Piece("text", str(delta.text))
            elif delta.type == "thinking_delta":
                yield Piece("thinking", str(delta.thinking))

    def _add_usage(self, usage: Any) -> None:
        """Sum what the server reported, on the producer thread only."""
        if usage is None:
            return
        details = getattr(usage, "output_tokens_details", None)
        for key, value in (
            ("input_tokens", getattr(usage, "input_tokens", 0)),
            ("output_tokens", getattr(usage, "output_tokens", 0)),
            ("cache_read", getattr(usage, "cache_read_input_tokens", 0)),
            ("cache_write", getattr(usage, "cache_creation_input_tokens", 0)),
            (
                "thinking_tokens",
                getattr(details, "thinking_tokens", 0) if details else 0,
            ),
        ):
            if value:
                self.usage[key] = self.usage.get(key, 0) + int(value)
