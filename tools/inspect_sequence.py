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
    ap.add_argument("--deep", action="store_true", help="Run the same CUDA deep tracker path as batch evaluation.")
    ap.add_argument("--feature-layer", type=int, default=12)
    ap.add_argument("--long-thin", action="store_true")
    ap.add_argument("--preprobe", action="store_true")
    ap.add_argument("--preprobe-frames", type=int, default=12)
    ap.add_argument("--disable-ncc", action="store_true")
    ap.add_argument("--diag-flow", action="store_true")
    ap.add_argument("--protect-position", action="store_true")
    ap.add_argument("--protect-frames", type=int, default=20)
    ap.add_argument("--direction-gate", action="store_true")
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
    if args.deep or args.adapter:
        extractor = DeepFeatureExtractor(FeatureConfig(device="cuda", feature_layer=args.feature_layer, tracking_adapter_path=str(args.adapter) if args.adapter else None, cache_dir=str(ROOT / ".cache" / "torch"), use_amp=True, use_channels_last=True, cudnn_benchmark=True))
        head = build_similarity_head("depthwise_xcorr")
    config = TrackerConfig(use_deep_features=bool(args.deep or args.adapter), device="cuda", deep_adapter_compact_only=False,
                           small_target_bootstrap_flow_enabled=not args.long_thin,
                           deep_fallback_flow_tiny_enabled=not args.long_thin,
                           deep_fallback_flow_enabled=True,
                           deep_fallback_flow_long_thin_enabled=args.long_thin,
                           deep_fallback_flow_long_thin_ncc_full_width=args.long_thin,
                           deep_fallback_flow_long_thin_early_global_ncc_enabled=args.long_thin,
                           deep_fallback_flow_long_thin_early_global_ncc_min_score=0.0,
                           deep_fallback_flow_long_thin_early_global_ncc_verify_deep=False,
                           deep_fallback_flow_long_thin_preprobe_enabled=bool(args.preprobe),
                           deep_fallback_flow_long_thin_preprobe_frames=max(int(args.preprobe_frames), 0))
    config.deep_fallback_flow_long_thin_disable_ncc = bool(args.disable_ncc)
    config.deep_fallback_flow_long_thin_protect_position = bool(args.protect_position)
    config.deep_fallback_flow_long_thin_protect_frames = max(int(args.protect_frames), 0)
    config.deep_fallback_flow_long_thin_relocalize_direction_gate = bool(args.direction_gate)
    tracker = PanoSOTTracker(config, extractor, head)
    print('initial flow/ncc diagnostics:')
    for i, (frame, target) in enumerate(zip(frames, gt)):
        if i == 0:
            pred = tracker.initialize(frame, target)
        else:
            if args.diag_flow and tracker._previous_frame_gray is not None:
                gray = tracker._handcrafted_gray(frame, assume_normalized=True)
                anchor_state = tracker._hand_state or tracker.state
                anchor_bbox = tracker._state_to_output_bbox(anchor_state, 3840, 1920)
                flow_state, flow_ok = tracker._predict_with_optical_flow(frame, anchor_state, current_gray=gray)
                bbox = tracker._state_to_output_bbox(flow_state, 3840, 1920)
                print(f"diag {i:3d} flow={flow_ok} thin={tracker._last_flow_long_thin} asp={tracker._last_flow_adaptive_spread:.1f} anchor_x={anchor_bbox[0]+anchor_bbox[2]/2:.1f} flow_x={bbox[0]+bbox[2]/2:.1f} anchor_y={anchor_bbox[1]+anchor_bbox[3]/2:.1f} flow_y={bbox[1]+bbox[3]/2:.1f} inlier={tracker._last_flow_inlier_ratio:.2f} spread={tracker._last_flow_spread:.2f} reason={tracker._last_flow_reject_reason}")
            before = tracker.runtime_stats
            snapshot = (before.fallback_flow_results, before.fallback_ncc_results, before.relocalizations, before.relocalization_accepts, tracker.get_runtime_stats().get("deep_forward_calls", 0), before.local_searches)
            pred = tracker.track(frame)
            after = tracker.runtime_stats
            now = (after.fallback_flow_results, after.fallback_ncc_results, after.relocalizations, after.relocalization_accepts, tracker.get_runtime_stats().get("deep_forward_calls", 0), after.local_searches)
            if args.deep:
                diff = tuple(int(b - a) for a, b in zip(snapshot, now))
                print(f"branch {i:3d} flow+{diff[0]} ncc+{diff[1]} rel+{diff[2]}/{diff[3]} deep+{diff[4]} local+{diff[5]}")
        if i < 120:
            print(f"{i:4d} gt=({target[0]+target[2]/2:7.1f},{target[1]+target[3]/2:6.1f},{target[2]:5.1f},{target[3]:5.1f}) pred=({pred[0]+pred[2]/2:6.1f},{pred[1]+pred[3]/2:6.1f},{pred[2]:5.1f},{pred[3]:5.1f}) score={tracker.runtime_stats.last_score:5.3f} psr={tracker.runtime_stats.last_psr:5.2f} lost={tracker.lost_frames} rel={tracker.runtime_stats.relocalizations} trust={tracker._last_state_trust:.2f} gate={tracker._last_local_jump_gated} flow={tracker._last_flow_reliable} hv=({np.degrees(tracker._hand_velocity[0]):.2f},{np.degrees(tracker._hand_velocity[1]):.2f}) vv=({np.degrees(tracker.velocity[0]):.2f},{np.degrees(tracker.velocity[1]):.2f})")

if __name__ == "__main__":
    main()
