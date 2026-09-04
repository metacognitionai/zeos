# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Observability: the journal, folded into a picture of the system.

Adds no kernel instrumentation. ``state.py`` is a pure fold from journal events to
snapshots, which is what lets the same viewer serve a live run and a recorded one --
and, because determinism is already a gate, lets a recorded incident be scrubbed
exactly rather than approximately.
"""

from zeos.monitor.state import Monitor, SystemView, Timeline, fold

__all__ = ["Monitor", "SystemView", "Timeline", "fold"]
