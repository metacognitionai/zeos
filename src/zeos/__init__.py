# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""ZEOS -- a transformer operating system."""

from importlib.metadata import version

__all__ = ["__version__"]

#: Read from the installed distribution rather than written here as well, so
#: ``pyproject.toml`` stays the one place the version is stated.
__version__ = version("zeos")
