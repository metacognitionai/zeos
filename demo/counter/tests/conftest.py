# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The demo is plain scripts rather than a package, so its directory has to be
importable the way it is when one of them is run directly."""

from __future__ import annotations

import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent

if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
