from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker
from panosot.io import load_boxes, save_boxes
from panosot.metrics import circular_iou_xywh, otb_metrics


def decode_image(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as stream:
        image = Image.open(io.BytesIO(stream.read())).convert("RGB")
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def label_box(entry: dict) -> np.ndarray:
    box = entry["bbox"]
    return np.asarray(
        [float(box["cx"]) - 0.5 * float(box["w"]),
         float(box["cy"]) - 0.5 * float(box["h"]),
         float(box["w"]), float(box["h"])],
        dtype=np.float32,
    )


def _safe_metric_boxes(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return metric-safe boxes and the rows that describe valid rectangles."""
    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError("Boxes must be a two-dimensional array with four columns.")
    array = array[:, :4]
    valid = np.isfinite(array).all(axis=1) & (array[:, 2] > 0.0) & (array[:, 3] > 0.0)
    safe = array.copy()
    # A zero-area box gives an IoU of zero while retaining the original frame
    # alignment.  This makes malformed outputs visible in the diagnostics
    # instead of letting NaNs contaminate all sequence metrics.
    safe[~valid] = 0.0
    return safe.astype(np.float32), valid


def prediction_diagnostics(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_width: float,
) -> dict[str, float | int]:
    """Compute robust output-quality diagnostics alongside OTB accuracy."""
    pred = np.asarray(predictions, dtype=np.float64)[:, :4]
    target = np.asarray(targets, dtype=np.float64)[:, :4]
    valid_pred = np.isfinite(pred).all(axis=1) & (pred[:, 2] > 0.0) & (pred[:, 3] > 0.0)
    valid_target = np.isfinite(target).all(axis=1) & (target[:, 2] > 0.0) & (target[:, 3] > 0.0)
    valid = valid_pred & valid_target
    total = max(len(pred), 1)
    result: dict[str, float | int] = {
        "invalid_predictions": int((~valid_pred).sum()),
        "invalid_prediction_rate": float((~valid_pred).sum() / total),
    }
    if not np.any(valid):
        return result

    area_ratio = (pred[valid, 2] * pred[valid, 3]) / (target[valid, 2] * target[valid, 3])
    pred_center = pred[valid, :2] + 0.5 * pred[valid, 2:4]
    target_center = target[valid, :2] + 0.5 * target[valid, 2:4]
    # Longitude is circular in ERP images, so measure the shortest horizontal
    # displacement across the left/right boundary.
    delta_x = (pred_center[:, 0] - target_center[:, 0] + 0.5 * image_width) % image_width - 0.5 * image_width
    delta_y = pred_center[:, 1] - target_center[:, 1]
    center_error = np.hypot(delta_x, delta_y)
    return {
        **result,
        "area_ratio_median": float(np.median(area_ratio)),
        "area_ratio_p90": float(np.quantile(area_ratio, 0.90)),
        "center_error_px_median": float(np.median(center_error)),
        "center_error_px_p90": float(np.quantile(center_error, 0.90)),
    }


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_width: float,
) -> dict[str, float | int]:
    """Evaluate a sequence and useful early/late drift slices consistently."""
    pred, _ = _safe_metric_boxes(predictions)
    target, _ = _safe_metric_boxes(targets)
    frame_count = min(len(pred), len(target))
    pred = pred[:frame_count]
    target = target[:frame_count]
    if frame_count == 0:
        return {
            "success_rate": 0.0,
            "auc": 0.0,
            "mean_iou": 0.0,
            "auc_prefix_20": 0.0,
            "auc_prefix_80": 0.0,
            "auc_suffix_20": 0.0,
            "scored_frames": 0,
            "invalid_predictions": 0,
            "invalid_prediction_rate": 0.0,
        }

    def slice_auc(end: int | None = None, start: int = 0) -> float:
        part_pred = pred[start:end]
        part_target = target[start:end]
        return float(otb_metrics(part_pred, part_target, image_width)["auc"])

    prefix_20 = max(1, int(np.ceil(frame_count * 0.20)))
    prefix_80 = max(1, int(np.ceil(frame_count * 0.80)))
    suffix_start = max(0, frame_count - prefix_20)
    ious = np.asarray(
        [circular_iou_xywh(prediction, target_box, image_width) for prediction, target_box in zip(pred, target)],
        dtype=np.float32,
    )
    half_window = max(1, int(np.ceil(frame_count * 0.10)))
    early_iou = float(np.mean(ious[:half_window]))
    late_iou = float(np.mean(ious[-half_window:]))
    metrics = otb_metrics(pred, target, image_width)
    return {
        **metrics,
        "auc_prefix_20": slice_auc(prefix_20),
        "auc_prefix_80": slice_auc(prefix_80),
        "auc_suffix_20": slice_auc(start=suffix_start),
        "mean_iou_prefix_10": early_iou,
        "mean_iou_suffix_10": late_iou,
        "mean_iou_drop_prefix10_to_suffix10": early_iou - late_iou,
        "scored_frames": frame_count,
        **prediction_diagnostics(predictions[:frame_count], targets[:frame_count], image_width),
    }


def _atomic_save_boxes(path: Path, boxes: list[np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_boxes(temporary, boxes)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _result_is_complete(result: dict[str, Any], expected_frames: int) -> bool:
    return (
        int(result.get("frames", -1)) == expected_frames
        and int(result.get("requested_frames", -1)) == expected_frames
        and int(result.get("scored_frames", -1)) == expected_frames
    )


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def write_summary(out_root: Path) -> int:
    """Rebuild the CSV from durable per-sequence records after every result."""
    result_root = out_root / "sequence_results"
    rows = [
        result
        for path in sorted(result_root.glob("*.json"))
        if (result := _read_result(path)) is not None
    ]
    if not rows:
        return 0
    preferred = [
        "sequence", "frames", "requested_frames", "tracker_frames", "scored_frames", "seconds", "fps",
        "runtime_source", "success_rate", "auc", "mean_iou", "auc_prefix_20",
        "auc_prefix_80", "auc_suffix_20", "mean_iou_prefix_10", "mean_iou_suffix_10",
        "mean_iou_drop_prefix10_to_suffix10", "invalid_predictions",
        "invalid_prediction_rate", "area_ratio_median", "area_ratio_p90",
        "center_error_px_median", "center_error_px_p90",
    ]
    fields = [key for key in preferred if any(key in row for row in rows)]
    fields += sorted({key for row in rows for key in row}.difference(fields))
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write_json(out_root / "summary.json", {"sequences": rows})
    return len(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PanoSOT directly from 360VOTS zip archives.")
    p.add_argument("--zip-root", default="D:/360VOTS/360VOT-test")
    p.add_argument("--output-root", default="results/360vots_120_cuda")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--cache-dir", default=".cache/torch")
    p.add_argument(
        "--tracker-config-json",
        default="",
        help="JSON object of TrackerConfig overrides for reproducible ablations.",
    )
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=120)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Skip sequences with complete durable result records.")
    p.add_argument(
        "--reuse-predictions", action="store_true",
        help="Write metrics for matching existing prediction files without rerunning the tracker.",
    )
    p.add_argument("--batch-name", default="", help="Optional label stored in each result record.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        tracker_overrides = json.loads(args.tracker_config_json) if args.tracker_config_json else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --tracker-config-json: {exc}") from exc
    if not isinstance(tracker_overrides, dict):
        raise SystemExit("--tracker-config-json must be a JSON object.")
    zip_root = Path(args.zip_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    result_root = out_root / "sequence_results"
    result_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        write_summary(out_root)

    for zip_path in sorted(zip_root.glob("*.zip")):
        try:
            sequence_id = int(zip_path.stem)
        except ValueError:
            continue
        if not args.start <= sequence_id <= args.end:
            continue
        with zipfile.ZipFile(zip_path) as zf:
            label_name = next(name for name in zf.namelist() if name.endswith("label.json"))
            labels = json.loads(zf.read(label_name))
            image_names = sorted(
                (name for name in zf.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))),
                key=lambda name: int(Path(name).stem),
            )
            if args.max_frames > 0:
                image_names = image_names[: args.max_frames]
            frame_keys = [Path(name).name for name in image_names]
            gt = np.asarray([label_box(labels[key]) for key in frame_keys], dtype=np.float32)
            result_path = result_root / f"{sequence_id:04d}.json"
            existing = _read_result(result_path) if args.resume else None
            expected_frames = len(image_names)
            prediction_path = out_root / f"{sequence_id:04d}.txt"
            if existing is not None and _result_is_complete(existing, expected_frames) and prediction_path.exists():
                print({"sequence": f"{sequence_id:04d}", "status": "skipped", "frames": expected_frames}, flush=True)
                continue
            first = decode_image(zf, image_names[0])
            if args.reuse_predictions and prediction_path.exists():
                try:
                    existing_predictions = load_boxes(prediction_path)
                except (OSError, ValueError):
                    existing_predictions = np.empty((0, 4), dtype=np.float32)
                if len(existing_predictions) == expected_frames:
                    metrics = evaluate_predictions(existing_predictions, gt, float(first.shape[1]))
                    result: dict[str, object] = {
                        "sequence": f"{sequence_id:04d}",
                        "frames": expected_frames,
                        "requested_frames": expected_frames,
                        "seconds": 0.0,
                        "fps": 0.0,
                        "runtime_source": "reused_prediction",
                        "batch_name": args.batch_name,
                        "tracker_config_json": args.tracker_config_json,
                        "tracker_frames": max(expected_frames - 1, 0),
                        **metrics,
                    }
                    _atomic_write_json(result_path, result)
                    write_summary(out_root)
                    print({"sequence": f"{sequence_id:04d}", "status": "reused", **metrics}, flush=True)
                    continue
            tracker = build_tracker(
                use_deep_features=args.deep,
                backbone_name=args.backbone,
                device=args.device,
                cache_dir=args.cache_dir,
                **tracker_overrides,
            )
            start = time.perf_counter()
            def frames():
                yield first
                for name in image_names[1:]:
                    yield decode_image(zf, name)
            predictions = tracker.track_sequence(frames(), gt[0])
            elapsed = time.perf_counter() - start
            pred = np.asarray(predictions, dtype=np.float32)
            metrics = evaluate_predictions(pred, gt[: len(pred)], float(first.shape[1]))
            _atomic_save_boxes(prediction_path, predictions)
            runtime_stats = tracker.get_runtime_stats()
            tracker_frames = int(runtime_stats.pop("frames", max(len(predictions) - 1, 0)))
            result: dict[str, object] = {
                "sequence": f"{sequence_id:04d}", "frames": len(predictions),
                "requested_frames": expected_frames,
                "seconds": round(elapsed, 4),
                "fps": round(len(predictions) / elapsed, 4) if elapsed else 0.0,
                "runtime_source": "cuda" if str(args.device).lower() == "cuda" else str(args.device),
                "batch_name": args.batch_name,
                "tracker_config_json": args.tracker_config_json,
                "tracker_frames": tracker_frames,
                **metrics,
                **runtime_stats,
            }
            _atomic_write_json(result_path, result)
            write_summary(out_root)
            print(result, flush=True)

    count = write_summary(out_root)
    if count:
        print(f"Wrote {out_root / 'summary.csv'} ({count} sequences)")


if __name__ == "__main__":
    main()
