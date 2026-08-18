"""Evaluate 360VOTS zip sequences without extracting the 58GB dataset.

The archive format is the official 360VOTS layout: <id>/image/*.jpg and
<id>/label.json. Frames are decoded on demand and the deep extractor is
constructed once for the whole run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    adaptive_deep_tiny: bool = False,
    adaptive_max_short_pixels: float = 160.0,
    adaptive_max_aspect: float = 2.5,
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
        use_tiny_deep = bool(
            adaptive_deep_tiny
            and deep_extractor is not None
            and min(float(gt[0, 2]), float(gt[0, 3])) <= float(adaptive_max_short_pixels)
            and max(float(gt[0, 2]), float(gt[0, 3]))
                / max(min(float(gt[0, 2]), float(gt[0, 3])), 1e-6)
                <= float(adaptive_max_aspect)
        )
        tracker_config = (
            replace(
                config,
                use_deep_features=True,
                tiny_deep_strict_psr_enabled=True,
            )
            if use_tiny_deep else config
        )
        tracker = PanoSOTTracker(
            config=tracker_config,
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
        valid_gt = (gt[:n, 2] > 0.0) & (gt[:n, 3] > 0.0)
        if np.any(valid_gt):
            valid_metrics = otb_metrics(
                predictions[:n][valid_gt],
                gt[:n][valid_gt],
                image_width=float(first.shape[1]),
            )
            metrics.update({
                "valid_frame_count": int(np.sum(valid_gt)),
                "invalid_gt_frame_count": int(np.sum(~valid_gt)),
                "valid_auc": float(valid_metrics["auc"]),
                "valid_mean_iou": float(valid_metrics["mean_iou"]),
            })
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
    parser.add_argument(
        "--adaptive-deep-tiny",
        action="store_true",
        default=True,
        help="Use deep tracking for very tiny near-square initial targets (default).",
    )
    parser.add_argument(
        "--disable-adaptive-deep-tiny",
        dest="adaptive_deep_tiny",
        action="store_false",
        help="Disable the narrow automatic tiny-target deep branch.",
    )
    parser.add_argument(
        "--adaptive-deep-tiny-multiscale",
        action="store_true",
        help="Use the layer-7/layer-12 multiscale extractor for adaptive tiny targets.",
    )
    parser.add_argument("--adaptive-deep-tiny-max-short-pixels", type=float, default=24.0)
    parser.add_argument("--adaptive-deep-tiny-max-aspect", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / ".cache" / "torch"),
        help="Torch model cache directory for deep backbones.",
    )
    parser.add_argument("--deep-feature-layer", type=int, default=12)
    parser.add_argument(
        "--deep-multiscale",
        action="store_true",
        help="Fuse MobileNet layer-7 spatial detail with layer-12 semantics.",
    )
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
        "--disable-deep-fallback-flow",
        action="store_true",
        help="Disable optical-flow fallback only in deep mode.",
    )
    parser.add_argument(
        "--disable-deep-fallback-color",
        action="store_true",
        help="Disable colour fallback only in deep mode.",
    )
    parser.add_argument(
        "--enable-deep-fallback-color",
        action="store_true",
        help="Re-enable colour fallback in deep mode for comparison runs.",
    )
    parser.add_argument(
        "--enable-deep-fallback-ncc-before-color",
        action="store_true",
        help="Try ERP-NCC before colour in deep fallback mode.",
    )
    parser.add_argument(
        "--disable-deep-flow-appearance-score",
        action="store_true",
        help="Use legacy fixed flow confidence in deep fallback mode.",
    )
    parser.add_argument(
        "--enable-compact-fallback-flow",
        action="store_true",
        help="Enable optical-flow fallback for compact tiny targets after warm-up.",
    )
    parser.add_argument(
        "--enable-compact-deep-keep",
        action="store_true",
        help="Keep bounded low-PSR deep proposals during compact-target warm-up.",
    )
    parser.add_argument("--compact-deep-keep-max-aspect", type=float, default=None)
    parser.add_argument(
        "--enable-long-thin-flow",
        dest="enable_long_thin_flow",
        action="store_true",
        default=True,
        help="Enable bounded optical flow for small elongated targets (default).",
    )
    parser.add_argument(
        "--disable-long-thin-flow",
        dest="enable_long_thin_flow",
        action="store_false",
        help="Disable the long-thin optical-flow branch for ablation.",
    )
    parser.add_argument("--enable-long-thin-preprobe", action="store_true")
    parser.add_argument("--enable-long-thin-preprobe-color", action="store_true")
    parser.add_argument(
        "--long-thin-disable-ncc",
        action="store_true",
        help="For long-thin targets, keep foreground LK motion from being overwritten by ERP-NCC.",
    )
    parser.add_argument("--long-thin-protect-position", action="store_true")
    parser.add_argument("--long-thin-protect-frames", type=int, default=None)
    parser.add_argument("--long-thin-relocalize-direction-gate", action="store_true")
    parser.add_argument("--long-thin-disable-relocalize", action="store_true")
    parser.add_argument("--disable-long-thin-velocity-preserve", action="store_true")
    parser.add_argument("--disable-long-thin-velocity-hold", action="store_true")
    parser.add_argument(
        "--enable-long-thin-flow-scale",
        dest="enable_long_thin_flow_scale",
        action="store_true",
        default=True,
        help="Enable bounded horizontal scale growth for compact elongated targets (default).",
    )
    parser.add_argument(
        "--disable-long-thin-flow-scale",
        dest="enable_long_thin_flow_scale",
        action="store_false",
        help="Disable compact elongated target scale growth for ablation.",
    )
    parser.add_argument("--enable-long-thin-flow-height-scale", action="store_true")
    parser.add_argument("--long-thin-preprobe-frames", type=int, default=None)
    parser.add_argument(
        "--long-thin-full-width-ncc",
        action="store_true",
        help="Experimental: search full ERP width for elongated tiny-target NCC.",
    )
    parser.add_argument(
        "--enable-long-thin-global-ncc",
        action="store_true",
        help="Experimental: low-frequency full-ERP NCC recovery for elongated targets.",
    )
    parser.add_argument(
        "--enable-long-thin-semantic-recovery",
        action="store_true",
        help="Periodically use the deep semantic keyframe history to recover long-thin targets.",
    )
    parser.add_argument("--enable-long-thin-global-color", action="store_true")
    parser.add_argument("--enable-long-thin-early-global-ncc", action="store_true")
    parser.add_argument(
        "--enable-compact-persistent-probe",
        action="store_true",
        help="Run deep probes every frame for compact tiny targets.",
    )
    parser.add_argument("--compact-deep-probe-interval", type=int, default=0)
    parser.add_argument(
        "--deep-probe-interval",
        type=int,
        default=None,
        help="Override periodic deep-probe interval for ablation runs.",
    )
    parser.add_argument("--compact-flow-position-blend", type=float, default=0.0)
    parser.add_argument("--enable-ncc-identity-gate", action="store_true")
    parser.add_argument(
        "--disable-handcrafted-ncc",
        action="store_true",
        help="Disable handcrafted ERP-NCC for direct optical-flow ablations.",
    )
    parser.add_argument(
        "--deep-adapter-compact-only",
        action="store_true",
        help="Apply a loaded tracking adapter only to compact initial targets.",
    )
    parser.add_argument("--deep-adapter-compact-max-aspect", type=float, default=None)
    parser.add_argument("--deep-adapter-compact-min-area", type=float, default=None)
    parser.add_argument(
        "--enable-ncc-quarantine",
        action="store_true",
        help="Quarantine adaptive NCC after repeated low-PSR probes.",
    )
    parser.add_argument(
        "--enable-fallback-budget",
        action="store_true",
        help="Bound identity-free deep fallbacks after sustained low-PSR probes.",
    )
    parser.add_argument("--fallback-budget-frames", type=int, default=None)
    parser.add_argument("--fallback-budget-min-frames", type=int, default=None)
    parser.add_argument("--fallback-budget-start-frame", type=int, default=None)
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
        deep_fallback_flow_enabled=not args.disable_deep_fallback_flow,
        deep_fallback_color_enabled=(
            bool(args.enable_deep_fallback_color)
            and not args.disable_deep_fallback_color
        ),
        deep_fallback_ncc_before_color_enabled=bool(
            args.enable_deep_fallback_ncc_before_color
        ),
        deep_fallback_flow_use_appearance_score=not args.disable_deep_flow_appearance_score,
        deep_fallback_flow_tiny_compact_enabled=bool(args.enable_compact_fallback_flow),
        small_target_compact_deep_bootstrap_enabled=bool(args.enable_compact_deep_keep),
        small_target_compact_deep_keep_enabled=bool(args.enable_compact_deep_keep),
        deep_fallback_flow_long_thin_enabled=bool(args.enable_long_thin_flow),
        deep_fallback_flow_long_thin_preprobe_enabled=(
            bool(args.enable_long_thin_preprobe)
            and not args.disable_deep_fallback_flow
        ),
        deep_fallback_flow_long_thin_preprobe_use_color=bool(args.enable_long_thin_preprobe_color),
        deep_fallback_flow_long_thin_disable_ncc=bool(args.long_thin_disable_ncc),
        deep_fallback_flow_long_thin_protect_position=bool(args.long_thin_protect_position),
        deep_fallback_flow_long_thin_relocalize_direction_gate=bool(args.long_thin_relocalize_direction_gate),
        deep_fallback_flow_long_thin_disable_relocalize=bool(args.long_thin_disable_relocalize),
        deep_fallback_flow_long_thin_preserve_velocity_on_relocalize=not args.disable_long_thin_velocity_preserve,
        deep_fallback_flow_long_thin_velocity_hold_enabled=not args.disable_long_thin_velocity_hold,
        deep_fallback_flow_long_thin_scale_enabled=bool(args.enable_long_thin_flow_scale),
        deep_fallback_flow_long_thin_height_scale_enabled=bool(args.enable_long_thin_flow_height_scale),
        deep_fallback_flow_long_thin_ncc_full_width=bool(args.long_thin_full_width_ncc),
        deep_fallback_flow_long_thin_global_ncc_enabled=bool(args.enable_long_thin_global_ncc),
        deep_fallback_flow_long_thin_global_color_enabled=bool(args.enable_long_thin_global_color),
        deep_fallback_flow_long_thin_early_global_ncc_enabled=bool(args.enable_long_thin_early_global_ncc),
        deep_semantic_long_thin_recovery_enabled=bool(args.enable_long_thin_semantic_recovery),
        compact_target_persistent_deep_probe_enabled=bool(args.enable_compact_persistent_probe),
        compact_target_deep_probe_interval=max(int(args.compact_deep_probe_interval), 0),
        compact_fallback_flow_position_blend=float(np.clip(args.compact_flow_position_blend, 0.0, 1.0)),
        ncc_short_update_identity_gate_enabled=bool(args.enable_ncc_identity_gate),
        handcrafted_ncc_enabled=not args.disable_handcrafted_ncc,
        deep_adapter_compact_only=bool(args.deep_adapter_compact_only),
        ncc_quarantine_enabled=bool(args.enable_ncc_quarantine),
        fallback_reliability_budget_enabled=bool(args.enable_fallback_budget),
    )
    if args.fallback_budget_frames is not None:
        config.fallback_reliability_budget_frames = max(int(args.fallback_budget_frames), 1)
    if args.fallback_budget_min_frames is not None:
        config.fallback_reliability_budget_min_frames = max(int(args.fallback_budget_min_frames), 1)
    if args.fallback_budget_start_frame is not None:
        config.fallback_reliability_budget_start_frame = max(int(args.fallback_budget_start_frame), 1)
    if args.compact_deep_keep_max_aspect is not None:
        config.small_target_compact_deep_keep_max_aspect_ratio = max(
            float(args.compact_deep_keep_max_aspect), 1.0
        )
    if args.deep_probe_interval is not None:
        config.deep_probe_interval = max(int(args.deep_probe_interval), 1)
    if args.long_thin_preprobe_frames is not None:
        config.deep_fallback_flow_long_thin_preprobe_frames = max(int(args.long_thin_preprobe_frames), 0)
    if args.long_thin_protect_frames is not None:
        config.deep_fallback_flow_long_thin_protect_frames = max(int(args.long_thin_protect_frames), 0)
    if args.small_target_max_scale_step is not None:
        config.small_target_max_scale_step = float(args.small_target_max_scale_step)
    if args.deep_adapter_compact_max_aspect is not None:
        config.deep_adapter_compact_max_aspect_ratio = max(
            float(args.deep_adapter_compact_max_aspect), 1.0
        )
    if args.deep_adapter_compact_min_area is not None:
        config.deep_adapter_compact_min_init_area = max(
            float(args.deep_adapter_compact_min_area), 0.0
        )
    config.early_semantic_recovery_enabled = bool(args.early_semantic_recovery)
    extractor = head = None
    if args.deep or args.adaptive_deep_tiny:
        if args.tracking_adapter is not None and not args.tracking_adapter.is_file():
            raise SystemExit(f"Tracking adapter is missing: {args.tracking_adapter}")
        from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
        from panosot.models import build_similarity_head
        extractor = DeepFeatureExtractor(FeatureConfig(
            backbone_name=("mobilenet_v3_small_multiscale" if (args.deep_multiscale or args.adaptive_deep_tiny_multiscale) else "mobilenet_v3_small"),
            device=args.device,
            use_amp=args.device.startswith("cuda"),
            use_channels_last=args.device.startswith("cuda"),
            cudnn_benchmark=args.device.startswith("cuda"),
            feature_layer=args.deep_feature_layer,
            cache_dir=args.cache_dir,
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
                adaptive_deep_tiny=args.adaptive_deep_tiny,
                adaptive_max_short_pixels=args.adaptive_deep_tiny_max_short_pixels,
                adaptive_max_aspect=args.adaptive_deep_tiny_max_aspect,
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
