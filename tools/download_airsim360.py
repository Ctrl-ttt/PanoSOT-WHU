"""Download selected AirSim360 Omni360-Scene assets from Hugging Face.

Examples
--------
  python tools/download_airsim360.py --scene NYC --components raw instance
  python tools/download_airsim360.py --scene DTW --components raw instance semantic

Files are placed below ``data/AirSim360`` by default.  Hugging Face's cache
and resumable downloader are used, so repeating an interrupted command only
downloads missing file ranges.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Insta360-Research/AirSim360"
REPO_TYPE = "dataset"

# Archive names are the ones published by the official AirSim360 dataset.
_COMPONENT_PATTERNS = {
    "raw": "Omni360-Scene/{scene}/{prefix}_Raw{part}.zip",
    "instance": "Omni360-Scene/{scene}/{prefix}_instance_panorama.zip",
    "semantic": "Omni360-Scene/{scene}/{prefix}_seg_panorama.zip",
    "depth": "Omni360-Scene/{scene}/{prefix}_depth_panorama.zip",
}
_SCENE_PREFIXES = {
    "CITYPARK": "citypark",
    "DTW": "dtw",
    "NYC": "nyc",
}
_SCENE_DIRECTORIES = {
    "CITYPARK": "CityPark",
    "DTW": "DTW",
    "NYC": "NYC",
}
_RAW_PARTS = {
    "CITYPARK": ["_Part1", "_Part2", "_Part3"],
    "DTW": [""],
    "NYC": [""],
}


def build_allow_patterns(scenes: list[str], components: list[str]) -> list[str]:
    """Return exact repository paths requested from the official dataset."""
    patterns: list[str] = []
    for scene in scenes:
        canonical = scene.upper()
        prefix = _SCENE_PREFIXES[canonical]
        for component in components:
            parts = _RAW_PARTS[canonical] if component == "raw" else [""]
            for part in parts:
                patterns.append(
                    _COMPONENT_PATTERNS[component].format(
                        scene=_SCENE_DIRECTORIES[canonical],
                        prefix=prefix,
                        part=part,
                    )
                )
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official AirSim360 Omni360-Scene data with resume support."
    )
    parser.add_argument(
        "--scene",
        nargs="+",
        default=["NYC"],
        choices=sorted(_SCENE_PREFIXES),
        type=str.upper,
        help="Scene(s) to download; NYC is the recommended first training split.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["raw", "instance"],
        choices=sorted(_COMPONENT_PATTERNS),
        help="Modalities to download. raw + instance are needed for tracker pretraining.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "AirSim360",
        help="Dataset destination (default: data/AirSim360).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the requested official repository files without downloading.",
    )
    args = parser.parse_args()

    patterns = build_allow_patterns(args.scene, args.components)
    print("Official dataset:", f"https://huggingface.co/datasets/{REPO_ID}")
    print("Requested files:")
    for pattern in patterns:
        print(" -", pattern)
    if args.dry_run:
        return

    try:
        # Some Windows/network configurations stall in the optional Xet client
        # before any bytes arrive. The regular HTTPS backend remains resumable.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with "
            "`pip install huggingface_hub`."
        ) from exc

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=args.output,
        # Keep incomplete transfers in the Hugging Face cache and resume them
        # on the next invocation instead of restarting multi-GB archives.
        max_workers=4,
    )
    print("Download complete:", args.output.resolve())


if __name__ == "__main__":
    main()
