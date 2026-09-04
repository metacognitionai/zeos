# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Fetch the local model seat's weights into this demo.

Separate from ``qwen_machine`` because downloading is I/O and I/O is the driver's
business, not the machine's -- the same reason ``zeos.core`` reads no clock.

    uv run python demo/counter/fetch_model.py

Q4_K_M, 2.7GB. Qwen3.5 is a hybrid architecture -- some layers keep an ordinary
per-position KV, others an SSM recurrent state with no position to rewind to -- so
a backend that pages contexts pays for it; this seat never splices, so it does not
care. ``demo/coop-count`` uses the same weights and explains that cost in full.
"""

from __future__ import annotations

from pathlib import Path

REPO = "unsloth/Qwen3.5-4B-GGUF"
FILE = "Qwen3.5-4B-Q4_K_M.gguf"
MODELS = Path(__file__).resolve().parent / "models"


def main() -> int:
    from huggingface_hub import hf_hub_download

    target = MODELS / FILE
    if target.exists():
        print(f"already here: {target}")
        return 0
    MODELS.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO, FILE, local_dir=str(MODELS))
    print(f"fetched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
