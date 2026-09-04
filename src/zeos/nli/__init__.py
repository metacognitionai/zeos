# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""ZEOS-NLI: natural-language instruction as compilation.

> **Instructions are compiled, not obeyed.**
>
> **The robot you speak to is a microphone. The thing you program is the fleet.**

An utterance arrives as data on a pipe carrying a principal, is compiled to an
inspectable artifact, and that artifact faces every gate any program faces. The
safety property is structural: a dangerous instruction fails the way ``rm -rf /``
fails for a non-root user -- permission denied, not moral disapproval.

The kernel-side halves live in ``core``: ``principals.py`` (who is asking, and what
they may cause) and ``gates.py`` (the semantic check on the way out). This package is
the language-facing half -- envelopes, compilation, and dispatch -- and is deliberately
the untrusted part.
"""

from zeos.nli.compiler import Artifact, ArtifactKind, Phrasing, RefusalReason, compile_utterance
from zeos.nli.dispatcher import Decision, OwnershipOp, OwnershipRequest, decide, echo_back
from zeos.nli.envelope import Deixis, InvocationSpec, Reference, Utterance, safety_word

__all__ = [
    "Artifact",
    "ArtifactKind",
    "Deixis",
    "Decision",
    "InvocationSpec",
    "OwnershipOp",
    "OwnershipRequest",
    "Phrasing",
    "Reference",
    "RefusalReason",
    "Utterance",
    "compile_utterance",
    "decide",
    "echo_back",
    "safety_word",
]
