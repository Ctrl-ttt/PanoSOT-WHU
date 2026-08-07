"""Print NCC state around a selected 360VOTS transition for diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.geometry import state_to_erp_bbox
from panosot.io import load_boxes, load_image
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head
from panosot.tracker import PanoSOTTracker, TrackerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start", type=int, default=75)
    parser.add_argument("--end", type=int, default=92)
    parser.add_argument("--deep-probe-interval", type=int, default=20)
    parser.add_argument("--deep-ncc-max-jump-ratio", type=float, default=None)
    parser.add_argument("--print-rejections", action="store_true")
    args = parser.parse_args()

    sequence = Path(args.sequence)
    frames = sorted((sequence / "image").iterdir())[: args.end]
    ground_truth = load_boxes(sequence / "gt.txt")[: args.end]
    feature_config = FeatureConfig(
        backbone_name="mobilenet_v3_small",
        device="cuda",
        use_amp=True,
        feature_layer=12,
        cache_dir=str(PROJECT_ROOT / ".cache" / "torch"),
    )
    tracker = PanoSOTTracker(TrackerConfig(
        use_deep_features=True,
        device="cuda",
        deep_probe_interval=args.deep_probe_interval,
        deep_ncc_max_jump_ratio=(
            args.deep_ncc_max_jump_ratio
            if args.deep_ncc_max_jump_ratio is not None else TrackerConfig.deep_ncc_max_jump_ratio
        ),
    ), deep_extractor=DeepFeatureExtractor(feature_config), similarity_head=build_similarity_head("depthwise_xcorr"))
    tracker.initialize(load_image(frames[0]), ground_truth[0])
    original_ncc = tracker._predict_with_ncc
    original_local_search = tracker._local_search

    def traced_local_search(frame: np.ndarray, predicted):
        state, score = original_local_search(frame, predicted)
        if args.start <= tracker._frame_count <= args.end:
            bbox = state_to_erp_bbox(state, 3840, 1920)
            print(
                f"DEEP {tracker._frame_count + 1:03d} score={score:.3f} "
                f"psr={tracker.runtime_stats.last_psr:.3f} "
                f"box={np.rint(bbox).astype(int).tolist()}"
            )
        return state, score

    def traced_ncc(frame: np.ndarray, gray: np.ndarray | None = None):
        rejects_before = tracker.runtime_stats.ncc_flow_disagreement_rejects
        state, score, reliable = original_ncc(frame, gray)
        rejected = tracker.runtime_stats.ncc_flow_disagreement_rejects > rejects_before
        if rejected and args.print_rejections:
            bbox = state_to_erp_bbox(state, 3840, 1920)
            print(
                f"REJECT {tracker._frame_count + 1:03d} score={score:.3f} "
                f"box={np.rint(bbox).astype(int).tolist()} "
                f"velocity={np.round(tracker._ncc_velocity, 1).tolist()}"
            )
        if args.start <= tracker._frame_count <= args.end:
            bbox = state_to_erp_bbox(state, 3840, 1920)
            print(
                f"NCC {tracker._frame_count + 1:03d} reliable={reliable} "
                f"score={score:.3f} box={np.rint(bbox).astype(int).tolist()} "
                f"velocity={np.round(tracker._ncc_velocity, 1).tolist()} "
                f"flow={tracker._ncc_flow_was_reliable}"
            )
        return state, score, reliable

    tracker._predict_with_ncc = traced_ncc
    tracker._local_search = traced_local_search
    for index, frame_path in enumerate(frames[1:], 1):
        prediction = tracker.track(load_image(frame_path))
        if args.start <= index <= args.end:
            print(
                f"OUT {index + 1:03d} box={np.rint(prediction).astype(int).tolist()} "
                f"gt={np.rint(ground_truth[index]).astype(int).tolist()} "
                f"score={tracker.runtime_stats.last_score:.3f} "
                f"psr={tracker.runtime_stats.last_psr:.3f}"
            )


if __name__ == "__main__":
    main()
