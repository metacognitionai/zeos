# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""How hard the model is asked to think, and the sampling that goes with it.

One table, read by the prompt-loop player and by the machine behind the kernel,
so a run labelled `--effort none` is the same request on both arms.
"""

import os
from dataclasses import dataclass

#: Qwen's published recipe: reasoning wants a lower temperature and a wider
#: nucleus, a one-word answer the opposite.
TEMP_THINKING, TEMP_BARE = 0.6, 0.7
TOP_P_THINKING, TOP_P_BARE = 0.95, 0.8


@dataclass(frozen=True, slots=True)
class Sampling:
    """What to send for one effort level."""

    temperature: float
    top_p: float
    #: `None` means "do not send the field", which is what a hosted API needs.
    top_k: int | None


def top_k_from_env() -> int | None:
    """`top_k` is not in the OpenAI schema, so it is only sent when asked for.

    Qwen's published recipe includes it, but a hosted API rejects the field
    outright, so it cannot be a default.
    """
    value = os.environ.get("OPENAI_TOP_K")
    return int(value) if value else None


def thinking(effort: str | None) -> bool:
    """Whether this effort asks the model to reason at all.

    `None` is "say nothing and take the server's own default", which for a
    reasoning model means it reasons. Only the literal `"none"` turns it off.
    """
    return effort != "none"


def sampling_for(effort: str | None) -> Sampling:
    """The sampling parameters that go with an effort level."""
    thinks = thinking(effort)
    return Sampling(
        temperature=TEMP_THINKING if thinks else TEMP_BARE,
        top_p=TOP_P_THINKING if thinks else TOP_P_BARE,
        top_k=top_k_from_env(),
    )
