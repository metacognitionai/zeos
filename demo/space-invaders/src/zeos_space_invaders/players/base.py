# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Prompt formatting and reply parsing, shared by every model backend.

Whatever drives the model, the board is presented and the reply is read the same
way, so a comparison between backends or between views measures the thing under
test; `respond` is the same request against a board the caller rendered, which
is how the zeos player reaches either backend without knowing which one it has.
"""

import importlib
from collections import deque
from pathlib import Path

from ..game import ACTIONS
from ..game.rules import DEFAULTS
from ..runlog import Decision
from ..utils.views import GridView, describe
from .rules_text import rules_section

PROMPTS = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT = PROMPTS / "zeos_space_invaderss_player.md"
FALLBACK_ACTION = "left"  # only when a reply cannot be parsed at all

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ThinkingNotDisabled(RuntimeError):
    """Effort was set to `none` and the server reasoned anyway."""


class SDKMissing(ModuleNotFoundError):
    """An optional vendor SDK is not installed.

    Its own class rather than a bare ModuleNotFoundError so the CLI can print
    the one-line fix for *this* and still let an unrelated import failure raise
    with its traceback intact.
    """


def sdk(module, extra):
    """A vendor SDK, imported the moment a player needs it and not before.

    Both are optional installs, and naming one at module import would put every
    backend's dependency on `uv run play`, which needs no model at all.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] != module:
            # The SDK is installed and something it imports is not: a broken
            # environment rather than a missing extra, so the traceback stays.
            raise
        raise SDKMissing(
            f"this player needs the {module} SDK, which is not installed: run "
            f"`uv sync --extra {extra}` "
            f"(or `pip install 'zeos-space-invaders[{extra}]'`)"
        ) from exc


def rules_prompt(view, rules=DEFAULTS):
    """Everything a player is told about the game, short of how to phrase a reply.

    One function because two players read it: the prompt loop adds `reply.md`
    and sends it as the system prompt, and the scheduled pilot has it appended
    to its descriptor body, so the two arms are told the same game.
    """
    return "\n".join(
        [
            describe(view, rules),
            "---\n",
            rules_section((PROMPTS / "rules_common.md").read_text(), rules),
            rules_section((PROMPTS / view.choose_file).read_text(), rules),
        ]
    )


class PromptPlayer:
    """Everything a prompted player does except issue the request."""

    def __init__(self, prompt=None, history=5, view=None, stream=True, rules=None):
        self.view = view or GridView()
        #: Taken here rather than read off the module: a prompt built from the
        #: defaults while the game runs 5x5 describes a board nobody is playing.
        self.rules = DEFAULTS if rules is None else rules
        self.prompt = prompt if prompt is not None else self.default_prompt()
        self.turns = deque(maxlen=history)  # (step, action, board) already played
        #: On by default: a reply that arrives in one lump leaves the scheduled
        #: job blocked, and a blocked job cannot be preempted.
        self.stream = stream

    def default_prompt(self):
        """Format section, shared rules, decision section, reply format.

        Only the first and third vary between views. The rest is one copy, so
        the rules cannot drift apart underneath a comparison.
        """
        return "\n".join(
            [rules_prompt(self.view, self.rules), (PROMPTS / "reply.md").read_text()]
        )

    def render_turn(self, obs, info):
        """Recent boards with the action taken, then the board now."""
        return self.render_history(self.view.state(obs, info))

    def render_history(self, now):
        """The same turn, from an already-rendered `now`.

        Split out for the scheduled player, whose board reaches it through a
        pipe already rendered, so both arms send the same shape of prompt.
        """
        parts = []
        if self.turns:
            parts.append("## Recent turns (oldest first)\n")
            for step, action, board in self.turns:
                parts.append(f"step {step} -> you played: {action}\n{board}\n")
        parts.append("## Now\n")
        parts.append(now)
        return "\n".join(parts)

    @staticmethod
    def parse(text):
        """The action in a reply, or None."""
        for word in text.lower().replace("`", " ").replace("*", " ").split():
            word = word.strip(".,!:;'\"()[]")
            if word in ACTIONS:
                return word
        return None

    def record(self, obs, info, text, reasoning, stop_reason, usage):
        """Build the log record and remember the turn. Returns (action, decision).

        The board is not copied in: the clock writes a frame for every tick and
        the decision names its tick, so a reader joins the two.
        """
        action = self.parse(text)
        decision = Decision(
            tick=info["ticks"],
            by="model",
            action=action or FALLBACK_ACTION,
            parsed=action is not None,
            prompt=self.render_turn(obs, info),
            reply=text,
            reasoning=reasoning,
            stop_reason=stop_reason,
            usage=usage,
        )
        self.turns.append(
            (info["steps"], decision.action, self.view.history(obs, info))
        )
        return decision.action, decision

    # -- one request, for callers that render the board themselves ----------

    def respond(self, rendered, effort):
        """One request against an already-rendered board.

        Returns `(text, reasoning, stop_reason, usage)`. Implemented per backend
        and used by `choose` too, so a caller that renders the state itself
        issues exactly the request the prompt loop would.
        """
        raise NotImplementedError

    def respond_stream(self, rendered, effort):
        """`respond`, delivered a piece at a time.

        Yields `(channel, text)` with `channel` `"thinking"` or `"answer"`, and
        returns the same four-tuple `respond` does. The channels stay apart so
        the action parser never reads a move out of the reasoning; this default
        issues one plain request, which is what `--no-stream` selects.
        """
        text, reasoning, stop_reason, usage = self.respond(rendered, effort)
        if reasoning:
            yield "thinking", reasoning
        if text:
            yield "answer", text
        return text, reasoning, stop_reason, usage

    @staticmethod
    def reasoning_tokens(usage):
        """Reasoning tokens this backend reports, if it separates them at all."""
        return 0

    def check_no_reasoning(self, reasoning, usage):
        """Refuse a server that reasoned when told not to.

        Checked rather than trusted: a server that ignores the switch scores
        whatever a truncated reply parses as, indistinguishable from a bad model.
        """
        spent = self.reasoning_tokens(usage)
        if not reasoning and not spent:
            return
        raise ThinkingNotDisabled(
            f"--effort none was requested but {self.model} reasoned anyway "
            f"({spent} reasoning tokens, {len(reasoning)} chars). This endpoint "
            f"does not honour it. Use a real effort level, or pass strict=False "
            f"to score it anyway."
        )
