"""Evaluate the tangent-plane + OSTrack backend on raw BFoV training videos.

Mirrors tools/evaluate_train_bfov.py (same BFoV->ERP geometry, same OTB metric,
same incremental/resume output) but with backend="ostrack".  Handcrafted/hybrid
results are directly comparable under the identical protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("WEIGHTS_DIR", str(PROJECT_ROOT / "submission" / "weights"))

from panosot.geometry import bfov_to_erp_bbox
from panosot.io import load_boxes
from panosot.metrics import otb_metrics
from submission.track import frame_from_cv, load_init_bfov
from tools.evaluate_train_bfov import (
    discover_sequences,
    load_done_rows,
    summarize,
    video_size,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default=str(PROJECT_ROOT / "train"))
    parser.add_argument("--splits", nargs="+", default=["real", "sim"], choices=["real", "sim"])
    parser.add_argument("--subset", default="", help="Comma-separated seq_XXXX or train_sim/seq_XXXX names.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of discovered sequences to evaluate.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "train_bfov_ostrack.json"))
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "results" / "train_bfov_ostrack.csv"))
    parser.add_argument("--variant", default="384", choices=["384", "256"])
    parser.add_argument("--weights", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-relocalize",
        action="store_true",
        help="Disable the expensive grid relocalization (recommended for offline CPU eval).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sequences already present in the output JSON (crash-safe re-launch).",
    )
    return parser.parse_args()


def iter_frames(seq_dir: Path, max_frames: int) -> Iterable[np.ndarray]:
    capture = cv2.VideoCapture(str(seq_dir / "video.mp4"))
    try:
        frame_index = 0
        while True:
            if max_frames > 0 and frame_index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame_from_cv(frame)
            frame_index += 1
    finally:
        capture.release()


def evaluate_sequence(tracker, name: str, seq_dir: Path, max_frames: int) -> dict[str, object]:
    image_width, image_height = video_size(seq_dir)
    init_bfov = load_init_bfov(seq_dir)
    init_box = bfov_to_erp_bbox(*init_bfov, image_width, image_height)

    start = time.perf_counter()
    predictions = tracker.track_sequence(iter_frames(seq_dir, max_frames), init_box)
    elapsed = time.perf_counter() - start

    labels_bfov = load_boxes(seq_dir / "groundtruth.txt")[: len(predictions)]
    targets = np.asarray(
        [bfov_to_erp_bbox(*row, image_width, image_height) for row in labels_bfov],
        dtype=np.float32,
    )
    pred_arr = np.asarray(predictions, dtype=np.float32)
    metrics = otb_metrics(pred_arr, targets, float(image_width))

    return {
        "name": name,
        "frames": len(predictions),
        "mode": "ostrack",
        "elapsed_sec": elapsed,
        "fps": len(predictions) / elapsed if elapsed > 0 else 0.0,
        "success_rate": metrics["success_rate"],
        "auc": metrics["auc"],
        "mean_iou": metrics["mean_iou"],
    }


def main() -> None:
    args = parse_args()
    selected = {item.strip() for item in args.subset.split(",") if item.strip()}
    sequences = discover_sequences(Path(args.train_root), args.splits, selected)
    if args.limit > 0:
        sequences = sequences[: args.limit]
    if not sequences:
        raise SystemExit("No training sequences found.")

    output_path = Path(args.output)
    csv_path = Path(args.csv)
    rows: list[dict[str, object]] = []
    done_names: set[str] = set()
    if args.resume:
        rows, done_names = load_done_rows(output_path)
        if done_names:
            print(f"Resuming: {len(done_names)} sequence(s) already done, skipping.", flush=True)

    from panosot.factory import build_tracker

    tracker_kwargs = {"relocalize_enabled": not args.no_relocalize}
    tracker = build_tracker(
        backend="ostrack",
        variant=args.variant,
        weights_path=args.weights,
        device=args.device,
        allow_download=False,
        tracker_kwargs=tracker_kwargs,
    )

    for index, (name, seq_dir) in enumerate(sequences, 1):
        if name in done_names:
            print(f"[{index}/{len(sequences)}] {name} ... (cached)", flush=True)
            continue
        print(f"[{index}/{len(sequences)}] {name} ...", flush=True)
        row = evaluate_sequence(tracker, name, seq_dir, args.max_frames)
        rows.append(row)
        print(
            f"  mode={row['mode']} frames={row['frames']} "
            f"SR={float(row['success_rate']):.4f} "
            f"AUC={float(row['auc']):.4f} "
            f"mIoU={float(row['mean_iou']):.4f} "
            f"FPS={float(row['fps']):.2f}",
            flush=True,
        )
        write_report(output_path, csv_path, rows)

    write_report(output_path, csv_path, rows)
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"Saved {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
