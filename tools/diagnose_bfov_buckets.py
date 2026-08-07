"""Bucket saved predictions by BFoV latitude and projected ERP width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes
from panosot.metrics import circular_iou_xywh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()

    labels = json.loads((args.sequence / "label.json").read_text(encoding="utf-8"))
    gt = load_boxes(args.sequence / "gt.txt")
    pred = load_boxes(args.pred)
    n = min(len(gt), len(pred))
    width = 3840.0
    buckets: dict[str, list[float]] = {}
    for index in range(n):
        item = labels.get(f"{index:06d}.jpg", {})
        bfov = item.get("bfov", {})
        lat = abs(float(bfov.get("clat", 0.0)))
        gt_width = float(gt[index, 2])
        if gt_width >= 0.95 * width:
            bucket = "full_width"
        elif lat >= 55.0:
            bucket = "polar_55"
        elif lat >= 35.0:
            bucket = "polar_35"
        else:
            bucket = "ordinary"
        buckets.setdefault(bucket, []).append(circular_iou_xywh(pred[index], gt[index], width))

    for name in ("ordinary", "polar_35", "polar_55", "full_width"):
        values = buckets.get(name, [])
        if values:
            values_np = np.asarray(values, dtype=np.float32)
            print(f"{name}: frames={len(values)} mean_iou={values_np.mean():.6f} auc_proxy={(values_np > 0.5).mean():.6f}")
        else:
            print(f"{name}: frames=0")


if __name__ == "__main__":
    main()
