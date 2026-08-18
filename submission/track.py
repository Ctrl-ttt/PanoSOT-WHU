#!/usr/bin/env python3
"""Competition submission entrypoint for PanoSOT-WHU.

The submission uses a conservative router:
- default to the tuned handcrafted tracker
- enable the deep tracker only for medium-sized, elongated targets where it
  was the best local-validation branch
"""
from __future__ import annotations

import glob
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "panosot").is_dir() else SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head
from panosot.tracker import PanoSOTTracker, TrackerConfig

DATASET_DIR = os.environ.get("DATASET_DIR", "/mnt/dataset")
RESULT_DIR = os.environ.get("RESULT_DIR", "/mnt/result")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
D2R = np.pi / 180.0
DEEP_ADAPTER_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "checkpoints", "local_bfov_adapter_layer12_t4_reg_cpu_1300.pt"
)
ISOTROPIC_HANDCRAFTED_SCALE_PAIRS = (
    (0.85, 0.85),
    (0.93, 0.93),
    (1.0, 1.0),
    (1.08, 1.08),
    (1.16, 1.16),
)

# The router is intentionally conservative. On the local validation split it
# selected only the sequence where the deep branch was clearly superior.
DEEP_ROUTER_MIN_AREA_PX = 10000.0
DEEP_ROUTER_MAX_AREA_PX = 24000.0
DEEP_ROUTER_MIN_SHORT_PX = 80.0
DEEP_ROUTER_MAX_SHORT_PX = 120.0
DEEP_ROUTER_MIN_ASPECT = 1.8
DEEP_ROUTER_TINY_MAX_AREA_PX = 7000.0
DEEP_ROUTER_TINY_MAX_SHORT_PX = 70.0
DEEP_ROUTER_TINY_MIN_ASPECT = 1.4
DEEP_ROUTER_TINY_MAX_ASPECT = 1.9
DEEP_ROUTER_MAX_ABS_LAT_DEG = 20.0


def bfov_to_erp_box(clon, clat, fov_h, fov_v, img_w, img_h):
    coslat = max(np.cos(clat * D2R), 1e-6)
    w = (fov_h / coslat) / 360.0 * img_w
    h = fov_v / 180.0 * img_h
    cx = (clon / 360.0 + 0.5) * img_w
    cy = (0.5 - clat / 180.0) * img_h
    return cx - w / 2.0, cy - h / 2.0, w, h


def erp_box_to_bfov(x, y, w, h, img_w, img_h):
    cx = x + w / 2.0
    cy = y + h / 2.0
    clon = (cx / img_w - 0.5) * 360.0
    clon = ((clon + 180.0) % 360.0) - 180.0
    clat = (0.5 - cy / img_h) * 180.0
    fov_v = float(np.clip(h / img_h * 180.0, 1e-3, 179.0))
    coslat = max(np.cos(clat * D2R), 1e-6)
    fov_h = float(np.clip((w / img_w * 360.0) * coslat, 1e-3, 179.0))
    return clon, clat, fov_h, fov_v


def find_video(seq_dir):
    vids = sorted(glob.glob(os.path.join(seq_dir, "*.mp4")))
    return vids[0] if vids else None


def list_sequences(dataset_dir):
    seqlist = os.path.join(dataset_dir, "seqlist.txt")
    if os.path.isfile(seqlist):
        with open(seqlist, encoding="utf-8-sig") as f:
            return [ln.strip().lstrip("\ufeff") for ln in f if ln.strip().lstrip("\ufeff")]
    return sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and find_video(os.path.join(dataset_dir, d))
    )


def load_init_bfov(seq_dir):
    with open(os.path.join(seq_dir, "init.txt"), encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.replace(",", " ").split()
            if len(tokens) < 4:
                break
            return [float(v) for v in tokens[:4]]
    raise ValueError(f"No valid init BFoV found in {os.path.join(seq_dir, 'init.txt')}")


def _should_use_deep_box(box_xywh: np.ndarray, clat_deg: float) -> bool:
    x, y, w, h = [float(v) for v in np.asarray(box_xywh).reshape(4)]
    if not np.all(np.isfinite([x, y, w, h])) or w <= 0.0 or h <= 0.0:
        return False
    short_side = min(w, h)
    long_side = max(w, h)
    area = w * h
    aspect = long_side / max(short_side, 1e-6)
    medium_elongated = (
        area >= DEEP_ROUTER_MIN_AREA_PX
        and area <= DEEP_ROUTER_MAX_AREA_PX
        and short_side >= DEEP_ROUTER_MIN_SHORT_PX
        and short_side <= DEEP_ROUTER_MAX_SHORT_PX
        and aspect >= DEEP_ROUTER_MIN_ASPECT
        and h >= w
        and abs(float(clat_deg)) <= DEEP_ROUTER_MAX_ABS_LAT_DEG
    )
    tiny_moderate_aspect = (
        area <= DEEP_ROUTER_TINY_MAX_AREA_PX
        and short_side <= DEEP_ROUTER_TINY_MAX_SHORT_PX
        and aspect >= DEEP_ROUTER_TINY_MIN_ASPECT
        and aspect <= DEEP_ROUTER_TINY_MAX_ASPECT
        and abs(float(clat_deg)) <= DEEP_ROUTER_MAX_ABS_LAT_DEG
    )
    return medium_elongated or tiny_moderate_aspect


def should_use_deep(init_bfov, img_w, img_h):
    box = np.asarray(bfov_to_erp_box(*init_bfov, img_w, img_h), dtype=np.float32)
    return _should_use_deep_box(box, float(init_bfov[1]))


def _tracker_config(use_deep: bool, device: str) -> TrackerConfig:
    config = TrackerConfig()
    config.use_deep_features = bool(use_deep)
    config.device = device

    if use_deep:
        config.deep_confirmation_frames = 2
        config.deep_update_quality_threshold = 0.65
        config.deep_template_update_ema = 0.08
        config.deep_template_update_background = 0.02
        config.deep_motion_momentum = 0.5
    else:
        config.confirmation_frames = 2
        config.update_quality_threshold = 0.65
        config.handcrafted_ncc_scale_pairs = ISOTROPIC_HANDCRAFTED_SCALE_PAIRS
        config.handcrafted_ncc_flow_every_frame = True
        config.handcrafted_ncc_search_factor = 5.0
    return config


@lru_cache(maxsize=2)
def _shared_deep_components(device: str) -> tuple[DeepFeatureExtractor, object]:
    feat_config = FeatureConfig(
        backbone_name="mobilenet_v3_small",
        device=device,
        use_amp=(device == "cuda"),
        feature_layer=12,
        normalize_features=False,
        template_size=112,
        coarse_search_size=224,
        refine_search_size=160,
        cache_dir=WEIGHTS_DIR,
        tracking_adapter_path=DEEP_ADAPTER_CHECKPOINT,
        use_channels_last=(device == "cuda"),
        cudnn_benchmark=(device == "cuda"),
    )
    deep_extractor = DeepFeatureExtractor(feat_config)
    similarity_head = build_similarity_head("depthwise_xcorr")
    return deep_extractor, similarity_head


def make_tracker(use_deep: bool):
    if use_deep:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = _tracker_config(True, device)
            deep_extractor, similarity_head = _shared_deep_components(device)
            return PanoSOTTracker(
                config=config,
                deep_extractor=deep_extractor,
                similarity_head=similarity_head,
            ), "deep"
        except Exception as exc:
            print(
                f"[warn] deep tracker unavailable, falling back to handcrafted: {exc}",
                file=sys.stderr,
            )

    device = "cpu"
    config = _tracker_config(False, device)
    return PanoSOTTracker(config=config, deep_extractor=None, similarity_head=None), (
        "hand-fallback" if use_deep else "hand"
    )


def frame_from_cv(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def track_one_sequence(seq_dir):
    """Return per-frame BFoV predictions for one sequence."""
    video = find_video(seq_dir)
    if not video:
        return [], "missing_video"

    cap = cv2.VideoCapture(video)
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return [], "missing_first_frame"

    h, w = first.shape[:2]
    init_bfov = load_init_bfov(seq_dir)
    x, y, bw, bh = bfov_to_erp_box(*init_bfov, w, h)
    init_box = np.array([x, y, bw, bh], dtype=np.float64)
    use_deep = should_use_deep(init_bfov, w, h)

    def frames():
        yield frame_from_cv(first)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame_from_cv(frame)

    tracker, actual_mode = make_tracker(use_deep)
    boxes = tracker.track_sequence(frames(), init_box)
    cap.release()

    results = [list(init_bfov)]
    for box in boxes[1:]:
        if (
            box is None
            or not np.all(np.isfinite(box))
            or box[2] <= 1e-3
            or box[3] <= 1e-3
        ):
            results.append([0.0, 0.0, 0.0, 0.0])
        else:
            results.append(list(erp_box_to_bfov(*box, w, h)))
    return results, actual_mode


def write_results(path, bfovs):
    with open(path, "w", encoding="utf-8") as f:
        for clon, clat, fh, fv in bfovs:
            f.write(f"{clon:.3f},{clat:.3f},{fh:.3f},{fv:.3f}\n")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    seqs = list_sequences(DATASET_DIR)
    if not seqs:
        print(f"[error] no sequences found in {DATASET_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[PanoSOT-WHU] processing {len(seqs)} sequences from {DATASET_DIR}")
    total_frames, t0 = 0, time.time()
    for idx, name in enumerate(seqs, 1):
        seq_dir = os.path.join(DATASET_DIR, name)
        ts = time.time()
        bfovs, mode = track_one_sequence(seq_dir)
        write_results(os.path.join(RESULT_DIR, f"{name}.txt"), bfovs)
        total_frames += len(bfovs)
        dt = time.time() - ts
        fps = len(bfovs) / dt if dt > 0 else 0
        print(
            f"  [{idx}/{len(seqs)}] {name}: {len(bfovs)} frames, "
            f"mode={mode}, {dt:.2f}s ({fps:.1f} FPS)"
        )

    total_dt = time.time() - t0
    avg = total_frames / total_dt if total_dt > 0 else 0
    print(
        f"[PanoSOT-WHU] done: {total_frames} frames / {total_dt:.2f}s, "
        f"avg {avg:.1f} FPS, results written to {RESULT_DIR}"
    )


if __name__ == "__main__":
    main()
