"""模板上下文扩展倍率消融：扫描 deep_template_enlarge 值。"""

from __future__ import annotations

import json, sys, time
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

enlarges = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
print(f"帧数: {MAX_FRAMES}, 变体: {enlarges}")

results = []
for enlarge in enlarges:
    label = f"enlarge={enlarge}"
    print(f"[{label}] ", end="", flush=True)

    cfg = TrackerConfig(use_deep_features=True, deep_template_enlarge=enlarge)
    fe = DeepFeatureExtractor(FeatureConfig())
    sh = build_similarity_head("depthwise_xcorr")
    tracker = PanoSOTTracker(config=cfg, deep_extractor=fe, similarity_head=sh)

    t0 = time.perf_counter()
    predictions = tracker.track_sequence((load_image(p) for p in frame_paths), init_box)
    elapsed = time.perf_counter() - t0

    n = min(len(predictions), len(gt_boxes))
    m = otb_metrics(predictions[:n], gt_boxes[:n], image_width=img_w)
    results.append({"enlarge": enlarge, "label": label, "elapsed_sec": elapsed, "fps": n/elapsed,
                     "success_rate": m["success_rate"], "auc": m["auc"], "mean_iou": m["mean_iou"]})
    print(f"AUC={m['auc']:.4f}  mIoU={m['mean_iou']:.4f}  SR={m['success_rate']:.4f}")

best = max(results, key=lambda r: r["auc"])
print(f"\n{'变体':<16} {'SR':>8} {'AUC':>8} {'mIoU':>8} {'FPS':>6}")
print("-" * 50)
for r in sorted(results, key=lambda r: r["auc"], reverse=True):
    m = " **" if r is best else ""
    print(f"{r['label']:<16} {r['success_rate']:>8.4f} {r['auc']:>8.4f} {r['mean_iou']:>8.4f} {r['fps']:>6.2f}{m}")
print(f"\n最优: {best['label']}  AUC={best['auc']:.4f}")

out = Path("results/ablation_enlarge.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"已保存 {out}")
