from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker
from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import circular_iou_xywh, otb_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect score/PSR/APCE distributions for threshold calibration.")
    parser.add_argument("--sequence", default="data/person_test", help="Directory containing ordered RGB frames.")
    parser.add_argument("--init-box", default="data/init_person.txt", help="Initial target box file.")
    parser.add_argument("--gt", default="data/gt_person.txt", help="Groundtruth box file.")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum frames, 0 for all frames.")
    parser.add_argument("--device", default="cuda", help="Device for deep feature extraction.")
    parser.add_argument("--output", default="results/threshold_calibration.json", help="JSON output path.")
    parser.add_argument("--csv", default="results/threshold_calibration.csv", help="Per-frame CSV output path.")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "min": percentile(values, 0),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "max": percentile(values, 100),
    }


def main() -> None:
    args = parse_args()
    sequence = Path(args.sequence)
    frame_paths = sorted(p for p in sequence.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    if len(frame_paths) < 2:
        raise SystemExit(f"Need at least 2 frames in {sequence}")

    init_box = load_boxes(args.init_box)[0]
    gt_boxes = load_boxes(args.gt)[: len(frame_paths)] if args.gt else None
    image_width = float(load_image(frame_paths[0]).shape[1])

    tracker = build_tracker(
        use_deep_features=True,
        device=args.device,
        deep_template_enlarge=4.0,
    )

    first_frame = load_image(frame_paths[0])
    predictions = [tracker.initialize(first_frame, init_box)]
    rows: list[dict[str, float | int]] = []

    t0 = time.perf_counter()
    for frame_idx, frame_path in enumerate(frame_paths[1:], start=1):
        pred = tracker.track(load_image(frame_path))
        predictions.append(pred)
        stats = tracker.get_runtime_stats()
        row: dict[str, float | int] = {
            "frame": frame_idx,
            "score": float(stats["last_score"]),
            "peak": float(stats["last_peak"]),
            "psr": float(stats["last_psr"]),
            "apce": float(stats["last_apce"]),
            "template_updates": int(stats["template_updates"]),
            "relocalizations": int(stats["relocalizations"]),
            "deep_forward_calls": int(stats["deep_forward_calls"]),
        }
        if gt_boxes is not None and frame_idx < len(gt_boxes):
            row["iou"] = circular_iou_xywh(pred, gt_boxes[frame_idx], image_width=image_width)
        rows.append(row)
    elapsed = time.perf_counter() - t0

    scores = [float(r["score"]) for r in rows]
    psrs = [float(r["psr"]) for r in rows]
    apces = [float(r["apce"]) for r in rows]
    good_rows = [r for r in rows if float(r.get("iou", 0.0)) >= 0.3]
    bad_rows = [r for r in rows if "iou" in r and float(r["iou"]) < 0.1]

    recommended: dict[str, float | None] = {
        "high_confidence": percentile([float(r["score"]) for r in good_rows], 25),
        "update_quality_threshold": percentile([float(r["score"]) for r in good_rows], 50),
        "relocalize_confidence_threshold": percentile([float(r["score"]) for r in bad_rows], 90),
    }

    metrics = None
    if gt_boxes is not None:
        n = min(len(predictions), len(gt_boxes))
        metrics = otb_metrics(predictions[:n], gt_boxes[:n], image_width=image_width)

    report = {
        "sequence": str(sequence),
        "frames": len(predictions),
        "elapsed_sec": elapsed,
        "fps": len(predictions) / elapsed if elapsed > 0 else 0.0,
        "metrics": metrics,
        "score": summarize(scores),
        "psr": summarize(psrs),
        "apce": summarize(apces),
        "recommended": recommended,
        "runtime": tracker.get_runtime_stats(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
