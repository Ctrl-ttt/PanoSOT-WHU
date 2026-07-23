from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes, load_image
from panosot.metrics import otb_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions with OTB-style metrics.")
    parser.add_argument("--pred", required=True, help="Prediction file path.")
    parser.add_argument("--gt", required=True, help="Ground-truth file path.")
    parser.add_argument(
        "--first-frame",
        required=True,
        help="A frame image from the sequence, used only to infer ERP width.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = load_boxes(args.pred)
    gt = load_boxes(args.gt)
    width = load_image(args.first_frame).shape[1]
    metrics = otb_metrics(pred, gt, image_width=width)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
