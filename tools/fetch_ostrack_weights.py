"""Fetch the OSTrack-384 weights into submission/weights/checkpoints/.

The 354 MB safetensors cannot be committed to GitHub (>100 MB file limit).
Docker builders run this once before `docker build` so the image stays
self-contained (runtime is offline).

    python tools/fetch_ostrack_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.ostrack import download_ostrack_384_weights  # noqa: E402


def main() -> None:
    dest_dir = PROJECT_ROOT / "submission" / "weights" / "checkpoints"
    dest = download_ostrack_384_weights(dest_dir)
    print(f"weights ready: {dest}")


if __name__ == "__main__":
    main()
