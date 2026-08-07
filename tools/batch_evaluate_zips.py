"""Evaluate 360VOTS zip sequences without extracting the 58GB dataset.

The archive format is the official 360VOTS layout: <id>/image/*.jpg and
<id>/label.json. Frames are decoded on demand and the deep extractor is
constructed once for the whole run.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import queue
import threading
import time
import zipfile

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import save_boxes
from panosot.metrics import otb_metrics
from panosot.tracker import PanoSOTTracker, TrackerConfig


def decode_image(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as stream:
        image = Image.open(stream).convert("RGB")
        return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def boxes_from_label(label: dict, frame_names: list[str]) -> np.ndarray:
    rows = []
    for name in frame_names:
        item = label[Path(name).name]
        box = item["bbox"]
        rows.append([
            float(box["cx"] - 0.5 * box["w"]),
            float(box["cy"] - 0.5 * box["h"]),
            float(box["w"]),
            float(box["h"]),
        ])
    return np.asarray(rows, dtype=np.float32)


def evaluate_zip(
    zip_path: Path,
    config: TrackerConfig,
    *,
    deep_extractor=None,
    similarity_head=None,
    max_frames: int = 0,
    prefetch: int = 0,
) -> dict:
    t0 = time.perf_counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = zip_path.stem + "/"
        label = json.loads(zf.read(prefix + "label.json"))
        frame_names = sorted(
            name for name in zf.namelist()
            if name.startswith(prefix + "image/")
            and name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        frame_names = frame_names[:max_frames] if max_frames > 0 else frame_names
        if not frame_names:
            return {"name": zip_path.stem, "num_frames": 0, "error": "no_frames"}
        gt = boxes_from_label(label, frame_names)
        tracker = PanoSOTTracker(
            config=config,
            deep_extractor=deep_extractor,
            similarity_head=similarity_head,
        )
        first = decode_image(zf, frame_names[0])
        init = gt[0]

        def frames():
            yield first
            if prefetch <= 0:
                for name in frame_names[1:]:
                    yield decode_image(zf, name)
                return
            buffer: queue.Queue[object] = queue.Queue(maxsize=max(int(prefetch), 1))
            sentinel = object()

            def producer() -> None:
                try:
                    for name in frame_names[1:]:
                        buffer.put(decode_image(zf, name))
                except BaseException as exc:  # propagate reader failures
                    buffer.put(exc)
                finally:
                    buffer.put(sentinel)

            thread = threading.Thread(target=producer, daemon=True)
            thread.start()
            while True:
                item = buffer.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
            thread.join()

        predictions = np.asarray(tracker.track_sequence(frames(), init), dtype=np.float32)
        n = min(len(predictions), len(gt))
        metrics = otb_metrics(predictions[:n], gt[:n], image_width=float(first.shape[1]))
        elapsed = time.perf_counter() - t0
        return {
            "name": zip_path.stem,
            "num_frames": len(frame_names),
            "elapsed_sec": elapsed,
            "fps": len(frame_names) / max(elapsed, 1e-6),
            **metrics,
            "runtime_stats": tracker.get_runtime_stats(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-root", type=Path, default=Path(r"D:\360VOTS\360VOT-test"))
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deep-feature-layer", type=int, default=12)
    parser.add_argument(
        "--tracking-adapter",
        type=Path,
        default=None,
        help="AirSim360-trained projection checkpoint; requires --deep.",
    )
    parser.add_argument("--small-target-max-scale-step", type=float, default=None)
    parser.add_argument(
        "--disable-small-target-bootstrap-flow",
        action="store_true",
        help="Use deep search from frame 1 and disable tiny-target flow fallback.",
    )
    parser.add_argument(
        "--enable-compact-fallback-flow",
        action="store_true",
        help="Enable optical-flow fallback for compact tiny targets after warm-up.",
    )
    parser.add_argument(
        "--enable-compact-persistent-probe",
        action="store_true",
        help="Run deep probes every frame for compact tiny targets.",
    )
    parser.add_argument("--compact-deep-probe-interval", type=int, default=0)
    parser.add_argument("--compact-flow-position-blend", type=float, default=0.0)
    parser.add_argument("--enable-ncc-identity-gate", action="store_true")
    parser.add_argument("--early-semantic-recovery", action="store_true")
    parser.add_argument("--subset", default=None)
    parser.add_argument(
        "--fast-scan",
        type=int,
        default=0,
        help="Evaluate only the first N frames of every zip for rapid failure screening.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.tracking_adapter is not None and args.deep and args.tracking_adapter.is_file():
        try:
            import torch
            adapter_meta = torch.load(args.tracking_adapter, map_location="cpu", weights_only=True)
            adapter_layer = adapter_meta.get("feature_layer")
            if adapter_layer is not None and args.deep_feature_layer == 12:
                args.deep_feature_layer = int(adapter_layer)
                print(f"Using adapter feature layer: {args.deep_feature_layer}")
        except Exception as exc:
            print(f"Warning: unable to read adapter metadata: {exc}")

    zips = sorted(args.zip_root.glob("*.zip"))
    if args.subset:
        wanted = {x.strip() for x in args.subset.split(",")}
        zips = [p for p in zips if p.stem in wanted]
    if not zips:
        raise SystemExit(f"No sequence zip files found under {args.zip_root}")

    max_frames = args.fast_scan if args.fast_scan > 0 else args.max_frames
    config = TrackerConfig(
        use_deep_features=args.deep,
        device=args.device,
        polar_erp_recovery_enabled=bool(args.deep),
        polar_erp_recovery_adaptive_height=bool(args.deep),
        small_target_bootstrap_flow_enabled=not args.disable_small_target_bootstrap_flow,
        deep_fallback_flow_tiny_enabled=not args.disable_small_target_bootstrap_flow,
        deep_fallback_flow_tiny_compact_enabled=bool(args.enable_compact_fallback_flow),
        compact_target_persistent_deep_probe_enabled=bool(args.enable_compact_persistent_probe),
        compact_target_deep_probe_interval=max(int(args.compact_deep_probe_interval), 0),
        compact_fallback_flow_position_blend=float(np.clip(args.compact_flow_position_blend, 0.0, 1.0)),
        ncc_short_update_identity_gate_enabled=bool(args.enable_ncc_identity_gate),
    )
    if args.small_target_max_scale_step is not None:
        config.small_target_max_scale_step = float(args.small_target_max_scale_step)
    config.early_semantic_recovery_enabled = bool(args.early_semantic_recovery)
    extractor = head = None
    if args.deep:
        if args.tracking_adapter is not None and not args.tracking_adapter.is_file():
            raise SystemExit(f"Tracking adapter is missing: {args.tracking_adapter}")
        from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
        from panosot.models import build_similarity_head
        extractor = DeepFeatureExtractor(FeatureConfig(
            device=args.device,
            use_amp=args.device.startswith("cuda"),
            use_channels_last=args.device.startswith("cuda"),
            cudnn_benchmark=args.device.startswith("cuda"),
            feature_layer=args.deep_feature_layer,
            cache_dir=str(PROJECT_ROOT / ".cache" / "torch"),
            tracking_adapter_path=(
                str(args.tracking_adapter) if args.tracking_adapter is not None else None
            ),
        ))
        head = build_similarity_head("depthwise_xcorr")

    print(f"Found {len(zips)} zip sequences; streaming evaluation started")
    results = []
    for index, path in enumerate(zips, 1):
        print(f"[{index}/{len(zips)}] {path.stem} ...", end=" ", flush=True)
        try:
            result = evaluate_zip(
                path, config, deep_extractor=extractor,
                similarity_head=head, max_frames=max_frames,
                prefetch=args.prefetch,
            )
            print(
                f"frames={result['num_frames']} fps={result.get('fps', 0.0):.2f} "
                f"AUC={result.get('auc', 0.0):.4f} mIoU={result.get('mean_iou', 0.0):.4f}"
            )
        except Exception as exc:
            result = {"name": path.stem, "num_frames": 0, "error": repr(exc)}
            print(f"ERROR {exc}")
        results.append(result)

    valid = [r for r in results if "error" not in r]
    summary = {
        "total": len(results),
        "succeeded": len(valid),
        "failed": len(results) - len(valid),
        "avg_auc": float(np.mean([r["auc"] for r in valid])) if valid else None,
        "avg_mean_iou": float(np.mean([r["mean_iou"] for r in valid])) if valid else None,
        "avg_fps": float(np.mean([r["fps"] for r in valid])) if valid else None,
        "total_elapsed_sec": float(sum(r.get("elapsed_sec", 0.0) for r in valid)),
    }
    payload = {"summary": summary, "sequences": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
