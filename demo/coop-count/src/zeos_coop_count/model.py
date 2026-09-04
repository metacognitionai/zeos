# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Where the model weights come from, kept apart from the machine because downloading is I/O."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_REPO", "DEFAULT_FILE", "default_path", "fetch"]

DEFAULT_REPO = "unsloth/Qwen3.5-4B-GGUF"
#: The default weights file, a 2.7GB Q4_K_M build of a hybrid model whose KV cache cannot truncate.
DEFAULT_FILE = "Qwen3.5-4B-Q4_K_M.gguf"


#: The thread count for every run, single-sourced because changing it changes the sampled tokens.
DEFAULT_THREADS = 32


def default_path(root: Path | None = None) -> Path:
    env = os.environ.get("ZEOS_COUNT_MODEL")
    if env:
        return Path(env)
    return (root or Path.cwd()) / "models" / DEFAULT_FILE


def fetch(
    repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE, *, into: Path | None = None
) -> Path:
    from huggingface_hub import hf_hub_download

    target = into or (Path.cwd() / "models")
    target.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo, filename, local_dir=str(target)))
