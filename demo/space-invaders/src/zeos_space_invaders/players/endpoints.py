# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Where a vendor's requests go, resolved once for everything that sends them.

One table per vendor, read by the prompt-loop players and the ZEOS machines
alike, so the two arms of a comparison cannot pick different models out of one
shell. The defaults reach a served model on localhost, because a default that
reaches a paid endpoint by accident is the worse failure; the variable names are
the two SDKs' own, so a shell set up for either works unchanged.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A resolved destination: which model, where, with what key."""

    model: str
    base_url: str | None
    api_key: str | None


@dataclass(frozen=True, slots=True)
class Vendor:
    """One vendor's variable names and fallbacks.

    `prefix` names the variables rather than listing them, because all three
    follow the SDK's `<VENDOR>_MODEL` / `_BASE_URL` / `_API_KEY` convention and
    spelling them out invites the two halves of a pair to drift apart.
    """

    prefix: str
    model: str
    #: `None` means the SDK's own endpoint, which is what an empty variable means
    #: too: `.env.example` ships the key with no value.
    base_url: str | None
    #: Sent rather than omitted because both SDKs refuse a client without one and
    #: a local server ignores it; `None` where a missing key should be an error.
    api_key: str | None

    def resolve(self, *, model=None, base_url=None, api_key=None) -> Endpoint:
        """Flags first, then the environment, then the fallback.

        The order `utils/config.py` documents for the whole demo. An empty string
        counts as unset, so `OPENAI_BASE_URL=` in a `.env` is not a base URL.
        """
        return Endpoint(
            model=model or os.environ.get(f"{self.prefix}_MODEL") or self.model,
            base_url=(
                base_url or os.environ.get(f"{self.prefix}_BASE_URL") or self.base_url
            ),
            api_key=(
                api_key or os.environ.get(f"{self.prefix}_API_KEY") or self.api_key
            ),
        )


OPENAI = Vendor(
    prefix="OPENAI",
    model="Qwen/Qwen3-4B",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

ANTHROPIC = Vendor(
    prefix="ANTHROPIC",
    model="claude-sonnet-5",
    #: A Claude Code session exports `ANTHROPIC_BASE_URL`, so a run started
    #: inside one follows that session's endpoint.
    base_url=None,
    #: No placeholder: a missing key here reaches a paid endpoint, and the SDK's
    #: own error names the variable.
    api_key=None,
)
