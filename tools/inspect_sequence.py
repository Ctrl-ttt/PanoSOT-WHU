"""Small diagnostic for printing per-frame 360VOTS tracker states."""
from __future__ import annotations

import argparse, json, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panosot.tracker import PanoSOTTracker, TrackerConfig
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--long-thin", action="store_true")
    args = ap.parse_args()
    with zipfile.ZipFile(args.zip) as zf:
        prefix = args.zip.stem + "/"
        labels = json.loads(zf.read(prefix + "label.json"))
        names = sorted(n for n in zf.namelist() if n.startswith(prefix + "image/") and n.endswith(".jpg"))[:args.frames]
        frames, gt = [], []
        for name in names:
            with zf.open(name) as stream:
                frames.append(np.asarray(Image.open(stream).convert("RGB"), dtype=np.float32) / 255.0)
            b = labels[Path(name).name]["bbox"]
            gt.append(np.asarray([b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["w"], b["h"]], dtype=np.float32))
    extractor = head = None
    if args.adapter:
        extractor = DeepFeatureExtractor(FeatureConfig(device="cuda", feature_layer=12, tracking_adapter_path=str(args.adapter), cache_dir=str(ROOT / ".cache" / "torch")))
        head = build_similarity_head("depthwise_xcorr")
    config = TrackerConfig(use_deep_features=extractor is not None, device="cuda", deep_adapter_compact_only=False,
                           small_target_bootstrap_flow_enabled=not args.long_thin,
                           deep_fallback_flow_tiny_enabled=not args.long_thin,
                           deep_fallback_flow_enabled=True,
                           deep_fallback_flow_long_thin_enabled=args.long_thin,
                           deep_fallback_flow_long_thin_ncc_full_width=args.long_thin,
                           deep_fallback_flow_long_thin_early_global_ncc_enabled=args.long_thin,
                           deep_fallback_flow_long_thin_early_global_ncc_min_score=0.0,
                           deep_fallback_flow_long_thin_early_global_ncc_verify_deep=False)
    tracker = PanoSOTTracker(config, extractor, head)
    for i, (frame, target) in enumerate(zip(frames, gt)):
        pred = tracker.initialize(frame, target) if i == 0 else tracker.track(frame)
        if i < 120:
            print(f"{i:4d} gt=({target[0]+target[2]/2:7.1f},{target[1]+target[3]/2:6.1f},{target[2]:5.1f},{target[3]:5.1f}) pred=({pred[0]+pred[2]/2:7.1f},{pred[1]+pred[3]/2:6.1f},{pred[2]:5.1f},{pred[3]:5.1f}) score={tracker.runtime_stats.last_score:5.3f} psr={tracker.runtime_stats.last_psr:5.2f} lost={tracker.lost_frames}")

if __name__ == "__main__":
    main()
