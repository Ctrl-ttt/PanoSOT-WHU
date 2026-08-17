#!/usr/bin/env python3
"""PanoSOT-WHU 比赛提交入口。

评测约定（与官方 demo 完全一致）：
- 输入: DATASET_DIR(默认 /mnt/dataset), 每序列 {name}/{video.mp4, init.txt(BFoV)}
- 输出: RESULT_DIR(默认 /mnt/result), 每序列 {name}.txt, 每行 clon,clat,fov_h,fov_v
- 丢失帧输出 0,0,0,0 占位, 行号与帧号严格对应, 不跳过不留空行
- 容器启动即自动跑完全部序列, 无需参数, 退出码 0

内部: 在 ERP 平面用 PanoSOT-WHU 混合跟踪器(深度特征 + NCC/光流手工特征),
仅在读入初始框与写出结果时做 BFoV <-> ERP 像素框转换(公式与官方 demo 一致)。
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATASET_DIR = os.environ.get("DATASET_DIR", "/mnt/dataset")
RESULT_DIR = os.environ.get("RESULT_DIR", "/mnt/result")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
D2R = np.pi / 180.0


# ---------- BFoV <-> ERP 像素框 转换（与官方 demo 一致） ----------
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


# ---------- 数据读取（与官方 demo 一致） ----------
def find_video(seq_dir):
    vids = sorted(glob.glob(os.path.join(seq_dir, "*.mp4")))
    return vids[0] if vids else None


def list_sequences(dataset_dir):
    seqlist = os.path.join(dataset_dir, "seqlist.txt")
    if os.path.isfile(seqlist):
        with open(seqlist) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and find_video(os.path.join(dataset_dir, d))
    )


def load_init_bfov(seq_dir):
    with open(os.path.join(seq_dir, "init.txt")) as f:
        p = f.readline().strip().replace(" ", "").split(",")
    return [float(v) for v in p[:4]]  # clon, clat, fov_h, fov_v


# ---------- tracker ----------
def make_tracker():
    import torch

    from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
    from panosot.models import build_similarity_head
    from panosot.tracker import PanoSOTTracker, TrackerConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrackerConfig()
    config.device = device
    feat_config = FeatureConfig(
        backbone_name=config.backbone_name,
        device=device,
        use_amp=(device == "cuda"),
        feature_layer=config.deep_feature_layer,
        normalize_features=config.normalize_deep_features,
        template_size=config.deep_template_size,
        coarse_search_size=config.coarse_search_size,
        refine_search_size=config.refine_search_size,
        cache_dir=WEIGHTS_DIR,  # 断网环境: 权重已随镜像打包
    )
    deep_extractor = DeepFeatureExtractor(feat_config)
    similarity_head = build_similarity_head("depthwise_xcorr")
    return PanoSOTTracker(
        config=config,
        deep_extractor=deep_extractor,
        similarity_head=similarity_head,
    )


def frame_from_cv(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def track_one_sequence(seq_dir):
    """返回逐帧 BFoV 列表 [[clon,clat,fov_h,fov_v], ...]。"""
    video = find_video(seq_dir)
    if not video:
        return []

    cap = cv2.VideoCapture(video)
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return []

    h, w = first.shape[:2]
    init_bfov = load_init_bfov(seq_dir)
    x, y, bw, bh = bfov_to_erp_box(*init_bfov, w, h)
    init_box = np.array([x, y, bw, bh], dtype=np.float64)

    def frames():
        yield frame_from_cv(first)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame_from_cv(frame)

    tracker = make_tracker()
    boxes = tracker.track_sequence(frames(), init_box)
    cap.release()

    # 首帧强制用 init BFoV（erp_bbox_to_state/state_to_erp_bbox 对高纬
    # 宽扁框非精确互逆，往返会引入像素偏差，首帧对齐 init 保证 IoU=1）。
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
    return results


def write_results(path, bfovs):
    with open(path, "w") as f:
        for clon, clat, fh, fv in bfovs:
            f.write(f"{clon:.3f},{clat:.3f},{fh:.3f},{fv:.3f}\n")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    seqs = list_sequences(DATASET_DIR)
    if not seqs:
        print(f"[错误] 在 {DATASET_DIR} 未找到任何序列", file=sys.stderr)
        sys.exit(1)

    print(f"[PanoSOT-WHU] 待处理序列 {len(seqs)} 条,数据集={DATASET_DIR}")
    total_frames, t0 = 0, time.time()
    for idx, name in enumerate(seqs, 1):
        seq_dir = os.path.join(DATASET_DIR, name)
        ts = time.time()
        bfovs = track_one_sequence(seq_dir)
        write_results(os.path.join(RESULT_DIR, f"{name}.txt"), bfovs)
        total_frames += len(bfovs)
        dt = time.time() - ts
        fps = len(bfovs) / dt if dt > 0 else 0
        print(
            f"  [{idx}/{len(seqs)}] {name}: {len(bfovs)} 帧, "
            f"{dt:.2f}s ({fps:.1f} FPS)"
        )

    total_dt = time.time() - t0
    avg = total_frames / total_dt if total_dt > 0 else 0
    print(
        f"[PanoSOT-WHU] 全部完成: {total_frames} 帧 / {total_dt:.2f}s, "
        f"平均 {avg:.1f} FPS,结果写入 {RESULT_DIR}"
    )


if __name__ == "__main__":
    main()
