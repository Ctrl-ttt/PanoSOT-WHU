"""Evaluate the submission tracker configuration on local ERP sequences.

The public Arena data is not available locally, but this runner gives every
candidate the same reproducible long-sequence regression gate before it is
submitted.  Sequences use the repository's ``image/``, ``init.txt`` and
``gt.txt`` layout.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import otb_metrics
from panosot.models import build_similarity_head
from panosot.tracker import PanoSOTTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/360VOTS_sample_6")
    parser.add_argument("--subset", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirmation-frames", type=int, default=2)
    parser.add_argument("--template-ema", type=float, default=0.08)
    parser.add_argument("--template-background", type=float, default=0.02)
    parser.add_argument("--motion-momentum", type=float, default=0.5)
    parser.add_argument("--polar-recovery", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--polar-adaptive-height", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deep-probe-interval", type=int, default=20)
    parser.add_argument("--anchor-verify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ncc-quarantine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fallback-budget", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--budget-frames", type=int, default=24)
    parser.add_argument("--budget-after-low-probes", type=int, default=3)
    parser.add_argument("--budget-start-frame", type=int, default=0)
    return parser.parse_args()


def make_tracker(args: argparse.Namespace) -> PanoSOTTracker:
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrackerConfig(
        use_deep_features=True,
        device=device,
        deep_confirmation_frames=args.confirmation_frames,
        deep_template_update_ema=args.template_ema,
        deep_template_update_background=args.template_background,
        deep_motion_momentum=args.motion_momentum,
        polar_erp_recovery_enabled=args.polar_recovery,
        polar_erp_recovery_adaptive_height=args.polar_adaptive_height,
        deep_probe_interval=args.deep_probe_interval,
        deep_relocalize_hand_anchor_verify_enabled=args.anchor_verify,
        ncc_quarantine_enabled=args.ncc_quarantine,
        fallback_reliability_budget_enabled=args.fallback_budget,
        fallback_reliability_budget_frames=args.budget_frames,
        fallback_reliability_budget_after_low_probes=args.budget_after_low_probes,
        fallback_reliability_budget_start_frame=args.budget_start_frame,
    )
    features = DeepFeatureExtractor(
        FeatureConfig(
            backbone_name=config.backbone_name,
            device=device,
            use_amp=device.startswith("cuda"),
            feature_layer=config.deep_feature_layer,
            normalize_features=config.normalize_deep_features,
            template_size=config.deep_template_size,
            coarse_search_size=config.coarse_search_size,
            refine_search_size=config.refine_search_size,
            cache_dir=str(ROOT / "submission" / "weights"),
        )
    )
    return PanoSOTTracker(config, features, build_similarity_head("depthwise_xcorr"))


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    selected = {x.strip() for x in args.subset.split(",") if x.strip()}
    rows: list[float] = []
    for seq in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected and seq.name not in selected:
            continue
        image_dir, init_path, gt_path = seq / "image", seq / "init.txt", seq / "gt.txt"
        if not (image_dir.is_dir() and init_path.is_file() and gt_path.is_file()):
            continue
        frames = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if args.max_frames:
            frames = frames[: args.max_frames]
        tracker = make_tracker(args)
        start = time.perf_counter()
        predictions = tracker.track_sequence((load_image(path) for path in frames), load_boxes(init_path)[0])
        gt = load_boxes(gt_path)[: len(predictions)]
        metrics = otb_metrics(np.asarray(predictions, dtype=np.float32), gt, float(load_image(frames[0]).shape[1]))
        rows.append(float(metrics["auc"]))
        print(f"{seq.name}: auc={metrics['auc']:.4f}, fps={len(frames) / (time.perf_counter() - start):.2f}", flush=True)
    if rows:
        print(f"macro_auc={float(np.mean(rows)):.4f}")


if __name__ == "__main__":
    main()
