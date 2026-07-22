from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes, load_sequence, save_boxes
from panosot.tracker import PanoSOTTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PanoSOT baseline tracker.")
    parser.add_argument("--sequence", required=True, help="Directory containing ordered RGB frames.")
    parser.add_argument(
        "--init-box",
        required=True,
        help="Initial target box file containing one line: x,y,w,h",
    )
    parser.add_argument("--output", required=True, help="Prediction output path.")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use deep features (MobileNetV3 + depthwise cross-correlation) instead of handcrafted.",
    )
    parser.add_argument(
        "--backbone",
        default="mobilenet_v3_small",
        help="Backbone name for deep features (default: mobilenet_v3_small).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for deep feature extraction (cpu or cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_sequence(args.sequence)
    init_box = load_boxes(args.init_box)[0]

    config = TrackerConfig(use_deep_features=args.deep, backbone_name=args.backbone, device=args.device)

    deep_extractor = None
    similarity_head = None
    if args.deep:
        from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
        from panosot.models import build_similarity_head

        feat_config = FeatureConfig(
            backbone_name=args.backbone,
            device=args.device,
            use_amp=(args.device.startswith("cuda")),
        )
        deep_extractor = DeepFeatureExtractor(feat_config)
        similarity_head = build_similarity_head("depthwise_xcorr")
        print(f"Deep feature mode: backbone={args.backbone}, device={deep_extractor.device}")

    tracker = PanoSOTTracker(config=config, deep_extractor=deep_extractor, similarity_head=similarity_head)
    predictions = tracker.track_sequence(frames, init_box)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_boxes(output_path, predictions)


if __name__ == "__main__":
    main()
