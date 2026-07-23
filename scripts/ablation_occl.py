"""遮挡抑制参数消融：confirmation_frames × update_quality_threshold 网格搜索。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import otb_metrics
from panosot.tracker import PanoSOTTracker, TrackerConfig
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head

SEQ = Path("data/360VOTS")
IMAGE_DIR = SEQ / "image"
INIT_BOX = SEQ / "init_box.txt"
GT = SEQ / "groundtruth.txt"
MAX_FRAMES = 50

frame_paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:MAX_FRAMES]
init_box = load_boxes(str(INIT_BOX))[0]
gt_boxes = load_boxes(str(GT))[:MAX_FRAMES]
img_w = float(load_image(frame_paths[0]).shape[1])

variants = []
for cf in [1, 2, 3, 4]:
    for qt in [0.58, 0.65, 0.72, 0.78]:
        variants.append((cf, qt))

print(f"帧数: {len(frame_paths)}, 变体数: {len(variants)}")
results = []

for conf_frames, qual_thresh in variants:
    label = f"确认{conf_frames}帧_阈值{qual_thresh:.2f}"
    print(f"\n[{label}] 运行中...", end=" ", flush=True)

    cfg = TrackerConfig(
        use_deep_features=True,
        confirmation_frames=conf_frames,
        update_quality_threshold=qual_thresh,
    )
    fe = DeepFeatureExtractor(FeatureConfig())
    sh = build_similarity_head("depthwise_xcorr")
    tracker = PanoSOTTracker(config=cfg, deep_extractor=fe, similarity_head=sh)

    t0 = time.perf_counter()
    predictions = tracker.track_sequence((load_image(p) for p in frame_paths), init_box)
    elapsed = time.perf_counter() - t0

    n = min(len(predictions), len(gt_boxes))
    m = otb_metrics(predictions[:n], gt_boxes[:n], image_width=img_w)

    entry = {
        "confirmation_frames": conf_frames,
        "update_quality_threshold": qual_thresh,
        "label": label,
        "num_frames": n,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(n / elapsed, 2),
        "success_rate": m["success_rate"],
        "auc": m["auc"],
        "mean_iou": m["mean_iou"],
    }
    results.append(entry)
    print(f"AUC={m['auc']:.4f}  mIoU={m['mean_iou']:.4f}")

# 输出
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

print("\n" + "=" * 70)
print(f"{'变体':<22} {'SR':>8} {'AUC':>8} {'mIoU':>8} {'FPS':>6}")
print("-" * 55)
best = max(results, key=lambda r: r["auc"])
for r in sorted(results, key=lambda r: r["auc"], reverse=True):
    marker = " **" if r is best else ""
    print(f"{r['label']:<22} {r['success_rate']:>8.4f} {r['auc']:>8.4f} {r['mean_iou']:>8.4f} {r['fps']:>6.2f}{marker}")
print("-" * 55)
print(f"最优: {best['label']}  AUC={best['auc']:.4f}  mIoU={best['mean_iou']:.4f}")

csv_path = out_dir / "ablation_occl.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("label,num_frames,elapsed_sec,fps,success_rate,auc,mean_iou,conf_frames,qual_thresh\n")
    for r in sorted(results, key=lambda r: r["auc"], reverse=True):
        f.write(f"{r['label']},{r['num_frames']},{r['elapsed_sec']},{r['fps']},{r['success_rate']},{r['auc']},{r['mean_iou']},{r['confirmation_frames']},{r['update_quality_threshold']}\n")

json_path = out_dir / "ablation_occl.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存至 {csv_path} / {json_path}")
