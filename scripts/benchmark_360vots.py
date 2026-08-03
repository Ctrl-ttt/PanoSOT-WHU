from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker
from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image, save_boxes
from panosot.metrics import otb_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PanoSOT on all local 360VOTS sequences.")
    p.add_argument("--dataset-root", default="data/360VOTS_unpacked")
    p.add_argument("--output-root", default="results/360vots_benchmark")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--cache-dir", default=".cache/torch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = seq_dir / "image"
        init_path = seq_dir / "init.txt"
        gt_path = seq_dir / "gt.txt"
        if not image_dir.is_dir() or not init_path.is_file():
            continue
        frames = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if args.max_frames > 0:
            frames = frames[: args.max_frames]
        init_box = load_boxes(init_path)[0]
        tracker = build_tracker(
            use_deep_features=args.deep,
            backbone_name=args.backbone,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        start = time.perf_counter()
        predictions = tracker.track_sequence((load_image(p) for p in frames), init_box)
        elapsed = time.perf_counter() - start
        pred_path = out_root / f"{seq_dir.name}.txt"
        save_boxes(pred_path, predictions)
        result: dict[str, object] = {
            "sequence": seq_dir.name,
            "frames": len(frames),
            "seconds": round(elapsed, 4),
            "fps": round(len(frames) / elapsed, 4) if elapsed else 0.0,
        }
        if gt_path.is_file():
            gt = load_boxes(gt_path)[: len(predictions)]
            pred = np.asarray(predictions, dtype=np.float32)
            if len(gt) == len(pred) and len(gt):
                width = float(load_image(frames[0]).shape[1])
                result.update(otb_metrics(pred, gt, width))
        result.update(tracker.get_runtime_stats())
        rows.append(result)
        print(result, flush=True)

    if rows:
        fields = sorted({key for row in rows for key in row})
        with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
