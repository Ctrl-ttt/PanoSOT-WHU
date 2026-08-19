"""训练集转换：mp4 + BFoV 标签 -> batch_evaluate 可评测目录格式。

训练集格式（ys_panotracking_train/train/{train_real,train_sim}/seq_XXXX/）：
- video.mp4        : 360 ERP 全景视频, 1440x720
- init.txt         : 初始 BFoV (clon,clat,fov_h,fov_v), 单位度
- groundtruth.txt  : 逐帧 BFoV, 每行一行, 与视频帧对应

输出格式（与 data/ 下评测序列一致）：
- image/000000.jpg : 逐帧图像（与视频帧序号对应）
- init_box.txt     : 初始像素框 x,y,w,h
- groundtruth.txt  : 逐帧像素框 x,y,w,h

BFoV -> ERP 像素框（分辨率 W x H）：
- w = fov_h / 360 * W
- h = fov_v / 180 * H
- cx = (clon + 180) / 360 * W
- cy = (90 - clat) / 180 * H

使用方式：
  python tools/convert_train_dataset.py --count 6 --max-frames 500
  python tools/convert_train_dataset.py --splits real sim --count 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRAIN_ROOT = Path(r"e:\影石暑期竞赛项目\ys_panotracking_train\train")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "train_eval"


def load_bfov_boxes(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        rows.append([float(value) for value in tokens[:4]])
    return np.asarray(rows, dtype=np.float64)


def bfov_to_pixel_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """BFoV (clon,clat,fov_h,fov_v) -> 轴对齐 ERP 像素框 x,y,w,h。"""
    clon = boxes[:, 0]
    clat = boxes[:, 1]
    fov_h = boxes[:, 2]
    fov_v = boxes[:, 3]
    cx = (clon + 180.0) / 360.0 * width
    cy = (90.0 - clat) / 180.0 * height
    # In ERP, a fixed spherical horizontal FoV occupies more pixels as the
    # target approaches either pole.  Keep this conversion identical to the
    # runtime submission and local-BFoV training code; the old conversion
    # omitted 1/cos(latitude), producing systematically narrow validation
    # boxes for high-latitude targets.
    cos_lat = np.maximum(np.cos(np.deg2rad(clat)), 1e-6)
    w = fov_h / cos_lat / 360.0 * width
    h = fov_v / 180.0 * height
    w = np.minimum(w, width)
    out = np.empty_like(boxes)
    out[:, 0] = cx - 0.5 * w
    out[:, 1] = cy - 0.5 * h
    out[:, 2] = w
    out[:, 3] = h
    return out


def convert_sequence(
    seq_dir: Path,
    out_dir: Path,
    max_frames: int = 0,
    overwrite: bool = False,
) -> int:
    """转换一个训练序列，返回实际转换的帧数。"""
    video_path = seq_dir / "video.mp4"
    init_path = seq_dir / "init.txt"
    gt_path = seq_dir / "groundtruth.txt"
    if not (video_path.is_file() and init_path.is_file() and gt_path.is_file()):
        print(f"[skip] 缺少文件: {seq_dir}")
        return 0

    if out_dir.exists() and not overwrite:
        print(f"[skip] 已存在: {out_dir.name}")
        return -1

    image_dir = out_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    # 读 BFoV 标签并转换
    gt_bfov = load_bfov_boxes(gt_path)
    init_bfov = load_bfov_boxes(init_path)
    if init_bfov.shape[0] < 1:
        print(f"[skip] init.txt 为空: {seq_dir}")
        return 0

    # 逐帧提取视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] 无法打开视频: {video_path}")
        return 0

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_count >= max_frames:
            break
        out_path = image_dir / f"{frame_count:06d}.jpg"
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            out_path, format="JPEG", quality=95
        )
        frame_count += 1
    cap.release()

    if frame_count == 0:
        print(f"[error] 视频无帧: {video_path}")
        return 0

    # 写像素框标签（截取与帧数一致）
    n = min(frame_count, gt_bfov.shape[0])
    gt_pixel = bfov_to_pixel_boxes(gt_bfov[:n], float(width), float(height))
    init_pixel = bfov_to_pixel_boxes(init_bfov[:1], float(width), float(height))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "groundtruth.txt", "w", encoding="utf-8") as f:
        for x, y, w, h in gt_pixel:
            f.write(f"{x:.3f},{y:.3f},{w:.3f},{h:.3f}\n")
    with open(out_dir / "init_box.txt", "w", encoding="utf-8") as f:
        x, y, w, h = init_pixel[0]
        f.write(f"{x:.3f},{y:.3f},{w:.3f},{h:.3f}\n")

    print(f"[ok] {out_dir.name}: {n} 帧 ({width}x{height})")
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help="训练集根目录（含 train_real/train_sim）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="输出目录（每个序列一个子目录）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["real", "sim"],
        choices=["real", "sim"],
        help="要转换的数据子集",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="每个子集转换的序列数（按序号升序取前 N 个）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="每序列最多转换的帧数（0 = 全部）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_names = {"real": "train_real", "sim": "train_sim"}
    total_frames = 0
    converted = 0
    for split in args.splits:
        split_dir = args.train_root / split_names[split]
        if not split_dir.is_dir():
            print(f"[skip] 子集不存在: {split_dir}")
            continue
        seq_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("seq_"))
        chosen = seq_dirs[: args.count]
        print(f"--- {split} ({len(seq_dirs)} 序列，取前 {len(chosen)}) ---")
        for seq_dir in chosen:
            out_dir = args.out_dir / f"{split}_{seq_dir.name}"
            n = convert_sequence(seq_dir, out_dir, max_frames=args.max_frames, overwrite=args.overwrite)
            if n is not None and n > 0:
                converted += 1
                total_frames += n
    print(f"\n完成：转换 {converted} 个序列，共 {total_frames} 帧 -> {args.out_dir}")


if __name__ == "__main__":
    main()
