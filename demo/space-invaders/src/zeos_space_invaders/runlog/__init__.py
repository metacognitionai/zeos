# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What a run leaves behind: one record shape, one directory layout, one reader.

Every entry point writes through here -- `play` (a human), `agent` (one episode)
and `compare` (many) -- so a run is the same thing on disk whoever produced it.
See `schema.py` for the record shapes and `writer.py` for the layout.
"""

from .reader import RunReader, aimed
from .schema import AUTHORS, SCHEMA_VERSION, Decision, Frame
from .writer import RunWriter, git_commit, meta_for, stamp

__all__ = [
    "AUTHORS",
    "SCHEMA_VERSION",
    "Decision",
    "Frame",
    "RunReader",
    "RunWriter",
    "aimed",
    "git_commit",
    "meta_for",
    "stamp",
]
