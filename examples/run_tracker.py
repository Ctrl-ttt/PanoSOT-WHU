from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image, save_boxes
from panosot import build_tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PanoSOT tracker.")
    parser.add_argument("--sequence", required=True, help="Directory containing ordered RGB frames.")
    parser.add_argument("--init-box", required=True, help="Initial target box file (x,y,w,h).")
    parser.add_argument("--output", required=True, help="Prediction output path.")
    parser.add_argument("--deep", action="store_true", help="Use deep features.")
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="Backbone name.")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda).")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit the run to the first N frames (0 = all).")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".cache" / "torch"), help="Weight cache directory.")
    parser.add_argument("--num-templates", type=int, default=3, help="Number of templates (1-5).")
    parser.add_argument("--template-weight-init", type=float, default=0.40, help="Weight for init template.")
    parser.add_argument("--template-weight-short", type=float, default=0.35, help="Weight for short template.")
    parser.add_argument("--template-weight-long", type=float, default=0.25, help="Weight for long template.")
    parser.add_argument("--occlusion-threshold", type=float, default=0.35, help="Occlusion confidence threshold.")
    parser.add_argument("--relocalize-threshold", type=float, default=0.35, help="Relocalize confidence threshold.")
    parser.add_argument("--relocalize-interval", type=int, default=10, help="Min frames between relocalizations.")
    parser.add_argument("--debug-dir", default="", help="Directory for per-frame debug artifacts.")
    parser.add_argument("--debug-start-frame", type=int, default=0, help="First tracked frame index to capture.")
    parser.add_argument("--debug-max-frames", type=int, default=20, help="Maximum number of tracked frames to capture.")
    parser.add_argument("--debug-frame-stride", type=int, default=1, help="Capture every Nth tracked frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_dir = Path(args.sequence)
    frame_paths = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    init_box = load_boxes(args.init_box)[0]

    print(f"Loading {len(frame_paths)} frames from {seq_dir}")
    print(f"Initial box: {init_box}")

    tracker = build_tracker(
        use_deep_features=args.deep,
        backbone_name=args.backbone,
        device=args.device,
        cache_dir=args.cache_dir,
        num_templates=args.num_templates,
        template_weight_init=args.template_weight_init,
        template_weight_short=args.template_weight_short,
        template_weight_long=args.template_weight_long,
        occlusion_threshold=args.occlusion_threshold,
        relocalize_confidence_threshold=args.relocalize_threshold,
        relocalize_min_interval=args.relocalize_interval,
        debug_dir=args.debug_dir or None,
        debug_start_frame=args.debug_start_frame,
        debug_max_frames=args.debug_max_frames,
        debug_frame_stride=args.debug_frame_stride,
    )

    if args.deep:
        print(f"Deep feature mode: backbone={args.backbone}, device={tracker.config.device}")
    else:
        print("Handcrafted feature mode")

    print(f"Using {tracker.num_templates} templates with weights:")
    print(f"  - init: {tracker.config.template_weight_init}")
    print(f"  - short: {tracker.config.template_weight_short}")
    print(f"  - long: {tracker.config.template_weight_long}")
    if tracker.config.debug_dir:
        print(f"Debug artifacts: {tracker.config.debug_dir}")

    start_time = time.time()
    frames = (load_image(p) for p in frame_paths)
    predictions = tracker.track_sequence(frames, init_box)
    elapsed = time.time() - start_time

    fps = len(frame_paths) / elapsed if elapsed > 0 else 0
    print(f"Tracking completed in {elapsed:.2f}s ({fps:.2f} FPS)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_boxes(output_path, predictions)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
