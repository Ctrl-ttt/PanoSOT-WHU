"""P3 极区自适应快速测试：0029 前 50 帧验证。"""
from __future__ import annotations

import sys, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import otb_metrics
from panosot.tracker import PanoSOTTracker, TrackerConfig
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head

seq = Path("data/0029")
frame_paths = sorted(p for p in (seq / "image").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:50]
init_box = load_boxes(str(seq / "init_box.txt"))[0]
gt_boxes = load_boxes(str(seq / "groundtruth.txt"))

cfg = TrackerConfig(
    use_deep_features=True,
    deep_template_enlarge=4.0,
    confirmation_frames=2,
    update_quality_threshold=0.65,
    # 极区参数使用默认值（lat>55°自动启用更多旋转角）
)
fe = DeepFeatureExtractor(FeatureConfig())
sh = build_similarity_head("depthwise_xcorr")
tracker = PanoSOTTracker(config=cfg, deep_extractor=fe, similarity_head=sh)

t0 = time.perf_counter()
predictions = tracker.track_sequence((load_image(p) for p in frame_paths), init_box)
elapsed = time.perf_counter() - t0

n = min(len(predictions), len(gt_boxes))
img_w = float(load_image(frame_paths[0]).shape[1])
metrics = otb_metrics(predictions[:n], gt_boxes[:n], image_width=img_w)
print(f"0029 50帧 (极区自适应): {elapsed:.1f}s, SR={metrics['success_rate']:.4f}, AUC={metrics['auc']:.4f}, mIoU={metrics['mean_iou']:.4f}")
