"""Evaluate the submission tracker on raw BFoV training videos.

This avoids the image-conversion step and scores predictions with the same
BFoV-to-ERP geometry used by the submission entrypoint.
"""
from __future__ import annotations

import argparse
import csv
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
from submission.track import frame_from_cv, load_init_bfov, make_tracker, should_use_deep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default=str(PROJECT_ROOT / "train"))
    parser.add_argument("--splits", nargs="+", default=["real", "sim"], choices=["real", "sim"])
    parser.add_argument("--mode", choices=["auto", "hand", "deep"], default="auto")
    parser.add_argument("--subset", default="", help="Comma-separated seq_XXXX or train_sim/seq_XXXX names.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of discovered sequences to evaluate.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "train_bfov_eval.json"))
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "results" / "train_bfov_eval.csv"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sequences already present in the output JSON (crash-safe re-launch).",
    )
    return parser.parse_args()


def _selected_name(display_name: str, selected: set[str]) -> bool:
    if not selected:
        return True
    return display_name in selected or display_name.split("/", 1)[-1] in selected


def discover_sequences(train_root: Path, splits: Iterable[str], selected: set[str]) -> list[tuple[str, Path]]:
    split_dirs = {"real": "train_real", "sim": "train_sim"}
    sequences: list[tuple[str, Path]] = []
    for split in splits:
        base = train_root / split_dirs[split]
        if not base.is_dir():
            continue
        for seq_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            display_name = f"{base.name}/{seq_dir.name}"
            if _selected_name(display_name, selected):
                sequences.append((display_name, seq_dir))
    return sequences


def video_size(seq_dir: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(seq_dir / "video.mp4"))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read first frame from {seq_dir / 'video.mp4'}")
        height, width = frame.shape[:2]
        return int(width), int(height)
    finally:
        capture.release()


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


def bfov_rows_to_erp(rows: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    return np.asarray(
        [bfov_to_erp_bbox(*row, image_width, image_height) for row in rows],
        dtype=np.float32,
    )


def evaluate_sequence(name: str, seq_dir: Path, mode: str, max_frames: int) -> dict[str, object]:
    image_width, image_height = video_size(seq_dir)
    init_bfov = load_init_bfov(seq_dir)
    init_box = bfov_to_erp_bbox(*init_bfov, image_width, image_height)

    if mode == "auto":
        use_deep = should_use_deep(init_bfov, image_width, image_height)
    else:
        use_deep = mode == "deep"
    tracker, actual_mode = make_tracker(use_deep)

    start = time.perf_counter()
    predictions = tracker.track_sequence(iter_frames(seq_dir, max_frames), init_box)
    elapsed = time.perf_counter() - start

    labels_bfov = load_boxes(seq_dir / "groundtruth.txt")[: len(predictions)]
    targets = bfov_rows_to_erp(labels_bfov, image_width, image_height)
    pred_arr = np.asarray(predictions, dtype=np.float32)
    metrics = otb_metrics(pred_arr, targets, float(image_width))

    return {
        "name": name,
        "frames": len(predictions),
        "mode": actual_mode,
        "elapsed_sec": elapsed,
        "fps": len(predictions) / elapsed if elapsed > 0 else 0.0,
        "success_rate": metrics["success_rate"],
        "auc": metrics["auc"],
        "mean_iou": metrics["mean_iou"],
        "runtime_stats": tracker.get_runtime_stats(),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "total": 0,
            "avg_success_rate": None,
            "avg_auc": None,
            "avg_mean_iou": None,
            "avg_fps": None,
            "total_elapsed_sec": 0.0,
        }
    return {
        "total": len(rows),
        "avg_success_rate": float(np.mean([float(row["success_rate"]) for row in rows])),
        "avg_auc": float(np.mean([float(row["auc"]) for row in rows])),
        "avg_mean_iou": float(np.mean([float(row["mean_iou"]) for row in rows])),
        "avg_fps": float(np.mean([float(row["fps"]) for row in rows])),
        "total_elapsed_sec": float(np.sum([float(row["elapsed_sec"]) for row in rows])),
    }


def write_report(output_path: Path, csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Write the current rows to JSON + CSV (crash-safe incremental save)."""
    report = {"summary": summarize(rows), "sequences": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        {key: value for key, value in row.items() if key != "runtime_stats"}
        for row in rows
    ]
    if flat_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)


def load_done_rows(output_path: Path) -> tuple[list[dict[str, object]], set[str]]:
    """Load previously written rows so a re-launch can skip finished sequences."""
    if not output_path.is_file():
        return [], set()
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        rows = list(existing.get("sequences", []))
        done = {str(row.get("name", "")) for row in rows}
        return rows, done
    except Exception:
        return [], set()


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

    for index, (name, seq_dir) in enumerate(sequences, 1):
        if name in done_names:
            print(f"[{index}/{len(sequences)}] {name} ... (cached)", flush=True)
            continue
        print(f"[{index}/{len(sequences)}] {name} ...", flush=True)
        row = evaluate_sequence(name, seq_dir, args.mode, args.max_frames)
        rows.append(row)
        print(
            f"  mode={row['mode']} frames={row['frames']} "
            f"SR={float(row['success_rate']):.4f} "
            f"AUC={float(row['auc']):.4f} "
            f"mIoU={float(row['mean_iou']):.4f} "
            f"FPS={float(row['fps']):.2f}",
            flush=True,
        )
        # Crash-safe: persist after every sequence so a long run never loses progress.
        write_report(output_path, csv_path, rows)

    write_report(output_path, csv_path, rows)
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"Saved {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
