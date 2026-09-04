# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Settings resolve as: .env file < environment variables < command-line flags.

A `--settings` file resolves the same way: it supplies flags, and a flag typed
on the command line wins over it.
"""

import argparse
import json
import os
from pathlib import Path

ENV_FILENAME = ".env"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_env_file():
    for candidate in (Path.cwd() / ENV_FILENAME, PROJECT_ROOT / ENV_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def load_env(path=None):
    """Load a .env into os.environ. Returns the path read, or None."""
    path = path or find_env_file()
    if path is None:
        return None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip().strip("'\"")
        # Empty means "not configured": .env.example ships valueless keys and
        # they must not blank out a real environment variable.
        if value and key not in os.environ:
            os.environ[key] = value
    return path


def settings_flags(argv, sections):
    """The flags a `--settings` file supplies, from the sections a command honours.

    `{"board": {"monster_rows": 2}}` becomes `["--monster-rows", "2"]`. These go
    before the command line at parse time, so a flag typed there wins over the
    file. A setting of `null` is left off, so the flag's own default decides it.

    `agent` asks for both sections; `play` asks for `board` alone, because a
    keyboard has no view, effort or history. Naming a section the file does not
    have is a KeyError rather than a quietly empty run.
    """
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--settings", type=Path)
    path = peek.parse_known_args(argv)[0].settings
    if path is None:
        return []
    settings = json.loads(Path(path).read_text())
    flags = []
    for section in sections:
        for key, value in settings[section].items():
            if value is not None:
                flags += [f"--{key.replace('_', '-')}", str(value)]
    return flags
