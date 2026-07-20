from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes, load_sequence, save_boxes
from panosot.tracker import PanoSOTTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PanoSOT baseline tracker.")
    parser.add_argument("--sequence", required=True, help="Directory containing ordered RGB frames.")
    parser.add_argument(
        "--init-box",
        required=True,
        help="Initial target box file containing one line: x,y,w,h",
    )
    parser.add_argument("--output", required=True, help="Prediction output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_sequence(args.sequence)
    init_box = load_boxes(args.init_box)[0]
    tracker = PanoSOTTracker()
    predictions = tracker.track_sequence(frames, init_box)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_boxes(output_path, predictions)


if __name__ == "__main__":
    main()
