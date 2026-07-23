"""完整551帧评测：深度特征 + scale_factors=(0.83, 1.0, 1.20)。

使用：python scripts/eval_best.py
"""

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

frame_paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
print(f"帧数: {len(frame_paths)}")

init_box = load_boxes(str(INIT_BOX))[0]

cfg = TrackerConfig(
    use_deep_features=True,
    deep_template_enlarge=4.0,  # 小目标模板扩大4倍上下文
)

fe = DeepFeatureExtractor(FeatureConfig())
sh = build_similarity_head("depthwise_xcorr")

tracker = PanoSOTTracker(config=cfg, deep_extractor=fe, similarity_head=sh)

print("跟踪中...")
t0 = time.perf_counter()
predictions = tracker.track_sequence((load_image(p) for p in frame_paths), init_box)
elapsed = time.perf_counter() - t0

gt_boxes = load_boxes(str(GT))
n = min(len(predictions), len(gt_boxes))
metrics = otb_metrics(predictions[:n], gt_boxes[:n],
                      image_width=float(load_image(frame_paths[0]).shape[1]))

result = {
    "frames": len(frame_paths),
    "time_sec": round(elapsed, 1),
    "fps": round(len(frame_paths) / elapsed, 2),
    "success_rate": metrics["success_rate"],
    "auc": metrics["auc"],
    "mean_iou": metrics["mean_iou"],
}

print(json.dumps(result, indent=2, ensure_ascii=False))
(SEQ / "best_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
print("\n已保存至 data/360VOTS/best_result.json")
