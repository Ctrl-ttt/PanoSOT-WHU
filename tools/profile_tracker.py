from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker
from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import otb_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile tracker accuracy and runtime counters on one sequence.")
    parser.add_argument("--sequence", default="data/person_test", help="Directory containing ordered RGB frames.")
    parser.add_argument("--init-box", default="data/init_person.txt", help="Initial target box file.")
    parser.add_argument("--gt", default="data/gt_person.txt", help="Groundtruth box file. Leave empty to skip metrics.")
    parser.add_argument("--no-gt", action="store_true", help="Skip groundtruth loading and metric calculation.")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to evaluate, 0 for all frames.")
    parser.add_argument("--device", default="cuda", help="Device for deep feature extraction.")
    parser.add_argument("--feature-layer", type=int, default=12, help="MobileNetV3 feature layer index.")
    parser.add_argument("--normalize-features", action="store_true", help="Apply L2 normalization to deep features.")
    parser.add_argument("--template-enlarge", type=float, default=4.0, help="Template context enlarge factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = Path(args.sequence)
    frame_paths = sorted(p for p in sequence.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise SystemExit(f"No frames found in {sequence}")

    init_box = load_boxes(args.init_box)[0]
    tracker = build_tracker(
        use_deep_features=True,
        device=args.device,
        deep_feature_layer=args.feature_layer,
        normalize_deep_features=args.normalize_features,
        deep_template_enlarge=args.template_enlarge,
    )

    t0 = time.perf_counter()
    predictions = tracker.track_sequence((load_image(p) for p in frame_paths), init_box)
    elapsed = time.perf_counter() - t0

    result: dict[str, object] = {
        "sequence": str(sequence),
        "frames": len(predictions),
        "elapsed_sec": round(elapsed, 3),
        "fps": round(len(predictions) / elapsed, 3) if elapsed > 0 else 0.0,
        "runtime": tracker.get_runtime_stats(),
    }

    if args.gt and not args.no_gt:
        gt_boxes = load_boxes(args.gt)
        n = min(len(predictions), len(gt_boxes))
        image_width = float(load_image(frame_paths[0]).shape[1])
        result["metrics"] = otb_metrics(predictions[:n], gt_boxes[:n], image_width=image_width)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
