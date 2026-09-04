# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Fake SDK clients and scripted machines, so the suite runs with neither vendor.

Each client is shaped like the one the real player or machine holds: a `create`
that records every request it was sent and replays canned turns. Nothing here
reaches a network, and `uv sync` without an extra still runs every test.
"""

import re
from types import SimpleNamespace


def pieces(text):
    """One delta per word, separators kept, so the pieces rejoin exactly and the
    pilot has work to do between the first piece and the last."""
    return [part for part in re.split(r"(\s+)", text or "") if part]


def state(step=0, score=0, lives=3, can_shoot=True):
    """A player's `info` argument; `steps` is what the views print and `ticks` is
    what a decision is stamped with."""
    return {
        "steps": step,
        "ticks": step,
        "score": score,
        "lives": lives,
        "can_shoot": can_shoot,
    }


# --- the Anthropic client ----------------------------------------------------


class Block(SimpleNamespace):
    pass


class StubClient:
    """Replays canned replies and records every request it was sent."""

    def __init__(self, replies, reasoning=None):
        self.replies = list(replies)
        self.reasoning = reasoning
        self.calls = []
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        content = [Block(type="text", text=self.replies[index])]
        if self.reasoning:
            content.insert(0, Block(type="thinking", thinking=self.reasoning))
        message = SimpleNamespace(
            content=content,
            stop_reason="end_turn",
            usage=SimpleNamespace(
                model_dump=lambda: {"input_tokens": 100, "output_tokens": 5}
            ),
        )
        return self._stream(message) if params.get("stream") else message

    @staticmethod
    def _stream(message):
        """The same turn, as the events a streaming Messages API would send."""
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100})
            ),
        )
        for block in message.content:
            thinking = block.type == "thinking"
            text = block.thinking if thinking else block.text
            for piece in pieces(text):
                delta = SimpleNamespace(
                    type="thinking_delta" if thinking else "text_delta"
                )
                setattr(delta, "thinking" if thinking else "text", piece)
                yield SimpleNamespace(type="content_block_delta", delta=delta)
        yield SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason=message.stop_reason),
            usage=SimpleNamespace(model_dump=lambda: {"output_tokens": 5}),
        )


# --- the OpenAI Responses client ---------------------------------------------


def message(text):
    return SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def reasoning(verbatim=None, summary=None):
    """Servers return the chain of thought verbatim; hosted APIs, a summary."""
    return SimpleNamespace(
        type="reasoning",
        content=[SimpleNamespace(type="reasoning_text", text=verbatim)]
        if verbatim
        else [],
        summary=[SimpleNamespace(type="summary_text", text=summary)] if summary else [],
    )


def usage(reasoning_tokens=0, output_tokens=40):
    return SimpleNamespace(
        model_dump=lambda: {
            "input_tokens": 500,
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
    )


class StubResponses:
    """Replays canned turns and records every request it was sent."""

    def __init__(self, *turns):
        self.turns = list(turns) or [{"output": [message("left")]}]
        self.calls = []
        self.responses = self

    def create(self, **params):
        self.calls.append(params)
        turn = self.turns[min(len(self.calls) - 1, len(self.turns) - 1)]
        incomplete = turn.get("reason")
        response = SimpleNamespace(
            output=turn["output"],
            status="incomplete" if incomplete else "completed",
            incomplete_details=SimpleNamespace(reason=incomplete)
            if incomplete
            else None,
            usage=turn.get("usage") or usage(),
        )
        return self._stream(response) if params.get("stream") else response

    @staticmethod
    def _stream(response):
        """The same turn as streaming deltas, built from the canned response so a
        test cannot describe one thing to the streamed path and another to the
        unstreamed one."""
        for item in response.output:
            if item.type == "reasoning":
                parts = [*(item.content or []), *(item.summary or [])]
                kind = "response.reasoning_text.delta"
            elif item.type == "message":
                parts, kind = item.content, "response.output_text.delta"
            else:  # pragma: no cover
                continue
            for part in parts:
                for piece in pieces(part.text):
                    yield SimpleNamespace(type=kind, delta=piece)
        yield SimpleNamespace(
            type="response.incomplete"
            if response.status == "incomplete"
            else "response.completed",
            response=response,
        )


# --- the two players those clients belong to ---------------------------------


def claude_player(*replies, effort=None, think=None, **kw):
    from zeos_space_invaders.players import ClaudePlayer

    return ClaudePlayer(
        prompt=kw.pop("prompt", "RULES"),
        effort=effort,
        client=StubClient(list(replies) or ["left"] * 400, reasoning=think),
        **kw,
    )


def openai_player(*turns, effort="none", view=None, **kw):
    from zeos_space_invaders.players import OpenAIPlayer
    from zeos_space_invaders.utils import VIEWS

    return OpenAIPlayer(
        client=StubResponses(*turns),
        effort=effort,
        view=view or VIEWS["lead"](),
        **kw,
    )


# --- a machine whose completions are scripted -------------------------------

#: One turn of the schema: a move and nothing else, as a whole document because
#: the machine's extractor closes the request when the object closes.
TURN = '{{"move": "{move}"}}'

#: `pieces` in one turn; a `delay` is charged per piece, so a turn lasts
#: ``PIECES_PER_TURN * delay``.
PIECES_PER_TURN = len(pieces(TURN.format(move="left")))

#: A turn slow enough to be caught in the middle of. The driver runs a batch of
#: at most ``ZeosDriver.max_ticks`` boundaries at ``DEFAULT_STALL_S`` each (64 x
#: 1ms), and a turn that does not outlast it is wholly queued before the kernel's
#: first decode, leaving nothing in flight to preempt, invalidate or leave
#: unfinished.
SLOW = 0.05  # a 100ms turn against a 64ms batch

#: A turn fast enough that a paced world can wait for it: ``tick_seconds`` must
#: outlast a turn, and a tick of zero is a world over before the model has
#: answered once.
BRISK = 0.005  # a 10ms turn
TICK = 0.02  # ...against a tick that outlasts it five times over


class StubStream:
    """A streamed chat completion, one delta per word; without `delay` the whole
    reply is queued before the kernel's first decode and there is nothing for the
    reflex to displace."""

    def __init__(self, text, delay=0.0):
        self.text, self.delay, self.closed = text, delay, False

    def __iter__(self):
        import time

        for piece in pieces(self.text):
            if self.delay:
                time.sleep(self.delay)
            delta = SimpleNamespace(content=piece, reasoning_content=None)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=len(self.text.split()),
                completion_tokens=len(pieces(self.text)),
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )

    def close(self):
        self.closed = True


class _Replay:
    """Replays a canned list of moves, one per request, and records what it was
    sent; shared so the two vendors provably differ only in the stream shape."""

    def __init__(self, moves=("shoot",), delay=0.0):
        self.moves, self.delay = list(moves or ("shoot",)), delay
        self.requests = []

    def _next(self, **params):
        """Record the request, and return the stream for this turn."""
        self.requests.append(params)
        move = self.moves[min(len(self.requests) - 1, len(self.moves) - 1)]
        return self._stream_for(TURN.format(move=move))

    def _stream_for(self, text):
        raise NotImplementedError


class StubChatClient(_Replay):
    """A chat-completions client that replays canned turns and records requests."""

    def __init__(self, moves=("shoot",), delay=0.0):
        super().__init__(moves, delay)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._next))

    def _stream_for(self, text):
        return StubStream(text, delay=self.delay)


class ScriptedClient(_Replay):
    """A chat-completions client that always replies with one fixed document, for
    the replies `TURN` cannot express."""

    def __init__(self, reply, delay=0.0):
        super().__init__((reply,), delay)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._next))

    def _next(self, **params):
        self.requests.append(params)
        return self._stream_for(self.moves[0])

    def _stream_for(self, text):
        return StubStream(text, delay=self.delay)


def _scripted(backend, client, **kw):
    """A machine over a scripted client, built through `build_machine` so a test
    drives the same assembly the CLI does, reflex included."""
    from zeos_space_invaders.players.zeos import build_machine

    machine = build_machine(backend, client=client, stall_s=0.001, **kw)
    machine.client_stub = client
    return machine


def stub_machine(*moves, delay=0.0, **kw):
    """An `OpenAIAPIMachine` over a scripted chat-completions client."""
    return _scripted("openai", StubChatClient(moves, delay=delay), **kw)


# --- the Anthropic side of the same seam ------------------------------------


class StubMessageStream:
    """A `messages.stream` context manager, one delta per word, shaped like the
    raw events `ClaudeAPIMachine._stream` reads."""

    def __init__(self, text, thinking=None, delay=0.0):
        self.text, self.thinking, self.delay = text, thinking, delay
        self.entered = self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_exc):
        self.exited = True
        return False

    def _delta(self, kind, **fields):
        delta = SimpleNamespace(type=kind, **fields)
        return SimpleNamespace(type="content_block_delta", delta=delta)

    def __iter__(self):
        import time

        if self.thinking:
            for piece in pieces(self.thinking):
                yield self._delta("thinking_delta", thinking=piece)
            yield self._delta("signature_delta", signature="Ercs" * 8)
        for piece in pieces(self.text):
            if self.delay:
                time.sleep(self.delay)
            yield self._delta("text_delta", text=piece)
        yield SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(
                input_tokens=len(self.text.split()),
                output_tokens=len(pieces(self.text)),
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                output_tokens_details=SimpleNamespace(
                    thinking_tokens=len(pieces(self.thinking or ""))
                ),
            ),
        )


class StubAnthropicClient(_Replay):
    """An Anthropic client that replays canned turns and records requests."""

    def __init__(self, moves=("shoot",), thinking=None, delay=0.0):
        super().__init__(moves, delay)
        self.thinking = thinking
        self.messages = SimpleNamespace(stream=self._next)

    def _stream_for(self, text):
        return StubMessageStream(text, thinking=self.thinking, delay=self.delay)


def stub_claude_machine(*moves, thinking=None, delay=0.0, **kw):
    """A `ClaudeAPIMachine` over a scripted Messages client."""
    client = StubAnthropicClient(moves, thinking=thinking, delay=delay)
    return _scripted("claude", client, **kw)
