"""Run the tangent-plane + OSTrack backend on an ERP image sequence.

Example:
    python tools/run_ostrack_demo.py \
        --sequence data/360tracking_demo \
        --init-box data/360tracking_demo/init_box.txt \
        --gt data/360tracking_demo/groundtruth.txt \
        --output results/ostrack_demo_pred.txt

Weights resolution:
    --weights auto   (default) use .cache/ostrack/OSTrack_vitb_384_*.safetensors,
                     downloading it from the HF mirror when missing;
    --weights PATH   load an explicit .pth / .safetensors checkpoint;
    --weights none   random initialization (smoke only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from panosot.factory import build_tracker  # noqa: E402
from panosot.io import load_boxes, load_image  # noqa: E402
from panosot.metrics import otb_metrics  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_frames(sequence_dir: Path, max_frames: int | None) -> list[Path]:
    image_dir = sequence_dir / "image"
    root = image_dir if image_dir.is_dir() else sequence_dir
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    return paths


def find_init_box(sequence_dir: Path) -> Path:
    for name in ("init.txt", "init_box.txt"):
        candidate = sequence_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no init.txt / init_box.txt under {sequence_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--init-box", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/ostrack_demo_pred.txt"))
    parser.add_argument("--variant", default="384", choices=["384", "256"])
    parser.add_argument("--weights", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-relocalize", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sequence_dir = args.sequence
    frames_paths = collect_frames(sequence_dir, args.max_frames)
    if not frames_paths:
        parser.error(f"no frames found under {sequence_dir}")
    init_path = args.init_box or find_init_box(sequence_dir)
    init_box = load_boxes(init_path)[0]

    weights = None if args.weights == "none" else args.weights
    tracker_kwargs = {}
    if args.no_relocalize:
        tracker_kwargs["relocalize_enabled"] = False
    tracker = build_tracker(
        backend="ostrack",
        variant=args.variant,
        weights_path=weights,
        device=args.device,
        tracker_kwargs=tracker_kwargs or None,
    )

    boxes = tracker.track_sequence(
        (load_image(path) for path in frames_paths), init_box
    )
    preds = np.asarray(boxes, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, preds, fmt="%.3f", delimiter=",")
    print(f"wrote {len(preds)} boxes to {args.output}")

    if args.gt is not None:
        gt = load_boxes(args.gt)
        gt = gt[: len(preds)]
        frame0 = load_image(frames_paths[0])
        metrics = otb_metrics(preds, gt, float(frame0.shape[1]))
        print("metrics:", metrics)


if __name__ == "__main__":
    main()
