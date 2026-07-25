"""测试数据生成工具。

支持多种模式：
  - basic:     随机噪声背景 + 移动方块
  - better:   渐变背景 + 移动方块
  - person:   渐变背景 + 移动人像
  - pano:     真实感全景场景 + 移动人像

用法:
  python examples/generate_test_data.py --mode pano
  python examples/generate_test_data.py --mode person --num-frames 30 --output data/person_test
"""
from __future__ import annotations

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


def _save_frame(output_dir: Path, idx: int, frame: np.ndarray):
    path = output_dir / f"{idx + 1:08d}.jpg"
    Image.fromarray(frame).save(path)


def _write_init(output_dir: Path, x: int, y: int, w: int, h: int):
    (output_dir / "init.txt").write_text(f"{x},{y},{w},{h}")


def _draw_person(frame: np.ndarray, cx: int, cy: int, size: int):
    """在人像位置绘制简笔人。"""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    head_r = int(size * 0.15)
    body_w = int(size * 0.25)
    body_h = int(size * 0.5)
    leg_h = int(size * 0.4)
    arm_len = int(size * 0.2)

    head_x, head_y = cx, cy - size // 2 + head_r
    draw.ellipse([head_x - head_r, head_y - head_r,
                  head_x + head_r, head_y + head_r],
                 fill=(255, 220, 180), outline=(139, 90, 43), width=2)

    bt = head_y + head_r
    bb = bt + body_h
    draw.rectangle([cx - body_w, bt, cx + body_w, bb],
                   fill=(0, 100, 200), outline=(0, 50, 100), width=2)

    ay = bt + int(body_h * 0.3)
    draw.line([(cx - body_w, ay), (cx - body_w - arm_len, ay + arm_len)],
              fill=(255, 220, 180), width=3)
    draw.line([(cx + body_w, ay), (cx + body_w + arm_len, ay + arm_len)],
              fill=(255, 220, 180), width=3)

    draw.line([(cx - 5, bb), (cx - body_w, bb + leg_h)],
              fill=(50, 50, 150), width=3)
    draw.line([(cx + 5, bb), (cx + body_w, bb + leg_h)],
              fill=(50, 50, 150), width=3)

    return np.array(img)


# ---------------------------------------------------------------------------
# 各模式生成器
# ---------------------------------------------------------------------------

def generate_basic(output_dir: Path, num_frames: int = 30):
    """随机噪声背景 + 移动方块。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = 720, 1440
    cx, cy = width // 2, height // 2
    tw, th = 200, 200
    vx, vy = 5, 2

    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        y1, y2 = max(0, cy - th // 2), min(height, cy + th // 2)
        x1, x2 = max(0, cx - tw // 2), min(width, cx + tw // 2)
        frame[y1:y2, x1:x2] = [255, 100, 100]
        _save_frame(output_dir, i, frame)
        cx = (cx + vx) % width
        cy = max(th // 2, min(height - th // 2, cy + vy))

    _write_init(output_dir, cx - tw // 2, cy - th // 2, tw, th)
    print(f"basic 模式: {num_frames} 帧已生成到 {output_dir}")


def generate_better(output_dir: Path, num_frames: int = 30):
    """渐变背景 + 移动方块。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = 720, 1440
    cx, cy = width // 2, height // 2
    tw, th = 200, 200
    vx, vy = 8, 3

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 50 + np.random.randint(0, 30, (height, width))
        frame[:, :, 1] = 60 + np.random.randint(0, 30, (height, width))
        frame[:, :, 2] = 70 + np.random.randint(0, 30, (height, width))

        y1, y2 = max(0, cy - th // 2), min(height, cy + th // 2)
        x1, x2 = max(0, cx - tw // 2), min(width, cx + tw // 2)
        frame[y1:y2, x1:x2] = [255, 200, 50]
        _save_frame(output_dir, i, frame)
        cx = (cx + vx) % width
        cy = max(th // 2, min(height - th // 2, cy + vy))

    _write_init(output_dir, cx - tw // 2, cy - th // 2, tw, th)
    print(f"better 模式: {num_frames} 帧已生成到 {output_dir}")


def generate_person(output_dir: Path, num_frames: int = 30):
    """渐变背景 + 移动人像。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = 720, 1440
    cx, cy = width // 2, height // 2
    size = 200
    vx, vy = 6, 2

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + np.random.randint(0, 20, (height, width))
        frame[:, :, 1] = 50 + np.random.randint(0, 20, (height, width))
        frame[:, :, 2] = 60 + np.random.randint(0, 20, (height, width))

        frame = _draw_person(frame, cx, cy, size)
        _save_frame(output_dir, i, frame)
        cx = (cx + vx) % width
        cy = max(size // 2, min(height - size // 2, cy + vy))

    tw, th = size, int(size * 1.8)
    _write_init(output_dir, cx - tw // 2, cy - th // 2, tw, th)
    print(f"person 模式: {num_frames} 帧已生成到 {output_dir}")


def generate_pano(output_dir: Path, num_frames: int = 50):
    """真实感全景场景 + 移动人像。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = 720, 1440
    cx, cy = width // 2, height // 2
    tw, th = 120, 200
    vx, vy = 15, 2

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 天空渐变
        sky = np.linspace([135, 206, 235], [70, 130, 180], height)
        for y in range(height):
            frame[y, :] = sky[y]

        # 地面
        ground_y = int(height * 0.6)
        ground = np.linspace([101, 67, 33], [80, 50, 20], height - ground_y)
        for y in range(ground_y, height):
            frame[y, :] = ground[y - ground_y]

        # 添加细节
        np.random.seed(42 + i)
        for _ in range(8):
            tree_x = np.random.randint(0, width)
            tree_y = ground_y - np.random.randint(20, 100)
            trunk_w, trunk_h = 8, np.random.randint(30, 80)
            cv2.rectangle(frame, (tree_x - trunk_w, tree_y),
                          (tree_x + trunk_w, tree_y + trunk_h), (101, 67, 33), -1)
            crown_r = np.random.randint(15, 35)
            cv2.circle(frame, (tree_x, tree_y - crown_r), crown_r, (34, 139, 34), -1)

        # 目标人像
        frame = _draw_person(frame, cx, cy, th)
        _save_frame(output_dir, i, frame)
        cx = (cx + vx) % width
        cy = max(th // 2, min(height - th // 2, cy + vy))

    _write_init(output_dir, cx - tw // 2, cy - th // 2, tw, th)
    print(f"pano 模式: {num_frames} 帧已生成到 {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODES = {
    "basic": generate_basic,
    "better": generate_better,
    "person": generate_person,
    "pano": generate_pano,
}


def main():
    parser = argparse.ArgumentParser(description="测试数据生成工具")
    parser.add_argument("--mode", choices=list(_MODES.keys()), default="pano",
                        help="生成模式 (默认: pano)")
    parser.add_argument("--num-frames", type=int, default=50, help="帧数")
    parser.add_argument("--output", default="data/pano_test", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    _MODES[args.mode](output_dir, args.num_frames)
    print(f"✅ 初始化框: {(output_dir / 'init.txt').read_text().strip()}")


if __name__ == "__main__":
    main()
