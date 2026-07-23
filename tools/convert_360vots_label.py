from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ERP_WIDTH = 3840.0
ERP_HEIGHT = 1920.0


def wrap_x(x: float, width: float = ERP_WIDTH) -> float:
    return x % width


def axis_aligned_from_bbox(entry: dict) -> list[float]:
    bbox = entry["bbox"]
    return [
        float(bbox["cx"]) - 0.5 * float(bbox["w"]),
        float(bbox["cy"]) - 0.5 * float(bbox["h"]),
        float(bbox["w"]),
        float(bbox["h"]),
    ]


def axis_aligned_from_rbbox(entry: dict) -> list[float]:
    rbbox = entry["rbbox"]
    angle = math.radians(float(rbbox["rotation"]))
    width = abs(float(rbbox["w"]) * math.cos(angle)) + abs(float(rbbox["h"]) * math.sin(angle))
    height = abs(float(rbbox["w"]) * math.sin(angle)) + abs(float(rbbox["h"]) * math.cos(angle))
    return [
        float(rbbox["cx"]) - 0.5 * width,
        float(rbbox["cy"]) - 0.5 * height,
        width,
        height,
    ]


def axis_aligned_from_fov(entry: dict, key: str) -> list[float]:
    fov = entry[key]
    width = float(fov["fov_h"]) / 360.0 * ERP_WIDTH
    height = float(fov["fov_v"]) / 180.0 * ERP_HEIGHT
    cx = (float(fov["clon"]) + 180.0) / 360.0 * ERP_WIDTH
    cy = (90.0 - float(fov["clat"])) / 180.0 * ERP_HEIGHT
    return [cx - 0.5 * width, cy - 0.5 * height, width, height]


def convert_entry(entry: dict, mode: str) -> list[float]:
    if mode == "bbox":
        return axis_aligned_from_bbox(entry)
    if mode == "rbbox":
        return axis_aligned_from_rbbox(entry)
    if mode == "bfov":
        return axis_aligned_from_fov(entry, "bfov")
    if mode == "rbfov":
        return axis_aligned_from_fov(entry, "rbfov")
    raise ValueError(f"Unsupported mode: {mode}")


def save_box(path: Path, box: list[float]) -> None:
    x, y, w, h = box
    path.write_text(f"{x:.6f},{y:.6f},{w:.6f},{h:.6f}\n", encoding="utf-8")


def save_boxes(path: Path, boxes: list[list[float]]) -> None:
    lines = []
    for x, y, w, h in boxes:
        lines.append(f"{x:.6f},{y:.6f},{w:.6f},{h:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 360VOTS label.json into init/gt axis-aligned ERP boxes.",
    )
    parser.add_argument("--label-json", required=True, help="Path to label.json.")
    parser.add_argument(
        "--mode",
        default="bbox",
        choices=("bbox", "rbbox", "bfov", "rbfov"),
        help="Which 360VOTS annotation field to convert.",
    )
    parser.add_argument("--init-out", required=True, help="Output path for the first-frame init box.")
    parser.add_argument("--gt-out", required=True, help="Output path for per-frame GT boxes.")
    parser.add_argument(
        "--wrap-x",
        action="store_true",
        help="Wrap x into [0, ERP_WIDTH) after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_path = Path(args.label_json)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    boxes = [convert_entry(entry, args.mode) for _, entry in sorted(data.items())]

    if args.wrap_x:
        for box in boxes:
            box[0] = wrap_x(box[0])

    init_box = boxes[0]
    save_box(Path(args.init_out), init_box)
    save_boxes(Path(args.gt_out), boxes)


if __name__ == "__main__":
    main()
