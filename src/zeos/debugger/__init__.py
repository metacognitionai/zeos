# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The debugger: a case and a journal, drawn as one page.

Two things a descriptor tree does not otherwise show. The wiring -- which job reads
which pipe, which pipe fires which handler, and at what priorities -- is spread
across a directory of files that deliberately never mention each other, so the
composition the kernel performs is invisible until it runs. And the run itself is a
journal of some tens of event kinds, read one line at a time.

``payload.py`` projects both into one JSON object; ``server.py`` assembles the page
around it, either as a self-contained file or from a local server; ``static/`` draws
it. The division is the same one the rest of the repository keeps: the arithmetic is
in Python where pytest can reach it, and the page draws what it is given.
"""

from zeos.debugger.payload import build_payload, frames, structure
from zeos.debugger.server import export, page, serve

__all__ = ["build_payload", "export", "frames", "page", "serve", "structure"]
