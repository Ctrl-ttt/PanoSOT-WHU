from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker
from panosot.io import save_boxes
from panosot.metrics import otb_metrics


def decode_image(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as stream:
        image = Image.open(io.BytesIO(stream.read())).convert("RGB")
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def label_box(entry: dict) -> np.ndarray:
    box = entry["bbox"]
    return np.asarray(
        [float(box["cx"]) - 0.5 * float(box["w"]),
         float(box["cy"]) - 0.5 * float(box["h"]),
         float(box["w"]), float(box["h"])],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PanoSOT directly from 360VOTS zip archives.")
    p.add_argument("--zip-root", default="D:/360VOTS/360VOT-test")
    p.add_argument("--output-root", default="results/360vots_120_cuda")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--cache-dir", default=".cache/torch")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=120)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    zip_root = Path(args.zip_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for zip_path in sorted(zip_root.glob("*.zip")):
        try:
            sequence_id = int(zip_path.stem)
        except ValueError:
            continue
        if not args.start <= sequence_id <= args.end:
            continue
        with zipfile.ZipFile(zip_path) as zf:
            label_name = next(name for name in zf.namelist() if name.endswith("label.json"))
            labels = json.loads(zf.read(label_name))
            image_names = sorted(
                (name for name in zf.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))),
                key=lambda name: int(Path(name).stem),
            )
            if args.max_frames > 0:
                image_names = image_names[: args.max_frames]
            frame_keys = [Path(name).name for name in image_names]
            gt = np.asarray([label_box(labels[key]) for key in frame_keys], dtype=np.float32)
            first = decode_image(zf, image_names[0])
            tracker = build_tracker(
                use_deep_features=args.deep,
                backbone_name=args.backbone,
                device=args.device,
                cache_dir=args.cache_dir,
            )
            start = time.perf_counter()
            def frames():
                yield first
                for name in image_names[1:]:
                    yield decode_image(zf, name)
            predictions = tracker.track_sequence(frames(), gt[0])
            elapsed = time.perf_counter() - start
            pred = np.asarray(predictions, dtype=np.float32)
            metrics = otb_metrics(pred, gt[: len(pred)], float(first.shape[1]))
            save_boxes(out_root / f"{sequence_id:04d}.txt", predictions)
            result: dict[str, object] = {
                "sequence": f"{sequence_id:04d}", "frames": len(predictions),
                "seconds": round(elapsed, 4),
                "fps": round(len(predictions) / elapsed, 4) if elapsed else 0.0,
                **metrics, **tracker.get_runtime_stats(),
            }
            rows.append(result)
            print(result, flush=True)

    if rows:
        fields = sorted({key for row in rows for key in row})
        with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
