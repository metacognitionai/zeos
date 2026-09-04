# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Serving a run as a page: the projection, the server, and the page itself.

Nothing here is imported by anything that plays a game.
"""

from .payload import comparison, episode, index

__all__ = ["comparison", "episode", "index"]
