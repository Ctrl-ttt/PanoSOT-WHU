"""一站式视频跟踪工具。

整合视频转帧、目标选择、跟踪、可视化全流程，只需一行命令。

用法:
  # 自动模式：自动检测目标并跟踪
  python tools/run_video_tracking.py --video data/video.mp4 --output result.mp4

  # 指定初始框
  python tools/run_video_tracking.py --video data/video.mp4 --output result.mp4 --init-box "x,y,w,h"

  # 交互式选框
  python tools/run_video_tracking.py --video data/video.mp4 --output result.mp4 --interactive

  # 跳过可视化（仅生成预测结果）
  python tools/run_video_tracking.py --video data/video.mp4 --output result.mp4 --no-visualize

  # 使用深度特征模式
  python tools/run_video_tracking.py --video data/video.mp4 --output result.mp4 --deep
"""
from __future__ import annotations

import argparse
import cv2
import numpy as np
import subprocess
import sys
import tempfile
from pathlib import Path


def video_to_frames(video_path: Path, output_dir: Path, step: int = 1):
    """将视频转换为帧序列。"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out_path = output_dir / f"{saved_idx + 1:08d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
            if saved_idx % 20 == 0:
                print(f"\r  已提取 {saved_idx}/{total_frames} 帧", end="")
        frame_idx += 1

    cap.release()
    print(f"\r  已提取 {saved_idx}/{total_frames} 帧")
    return saved_idx, fps, width, height


def detect_target(frame: np.ndarray, max_targets: int = 8):
    """从首帧检测候选目标。"""
    candidates = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 10000 < area < 800000 and 0.3 < w / max(h, 1) < 3.0:
            candidates.append((x, y, w, h, area))

    candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates[:max_targets]


def interactive_select(frame: np.ndarray):
    """交互式选框。"""
    h, w = frame.shape[:2]
    clone = frame.copy()
    start = None
    box = None

    def on_mouse(event, x, y, flags, param):
        nonlocal start, box
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start:
            cv2.imshow("Select Target", clone.copy())
            cv2.rectangle(clone, start, (x, y), (0, 0, 255), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            x1, y1 = min(start[0], x), min(start[1], y)
            x2, y2 = max(start[0], x), max(start[1], y)
            box = (x1, y1, x2 - x1, y2 - y1)

    cv2.namedWindow("Select Target", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Target", min(w, 1200), min(h, 800))
    cv2.setMouseCallback("Select Target", on_mouse)

    print("操作说明: 鼠标拖拽选框 -> 按 s 保存 -> 按 q 退出")
    while True:
        cv2.imshow("Select Target", clone)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s") and box:
            cv2.destroyAllWindows()
            return box
        elif key == ord("r"):
            clone = frame.copy()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None
    return None


def run_tracker(sequence_dir: Path, init_box: tuple[int, int, int, int], output_path: Path, use_deep: bool):
    """运行跟踪器。"""
    init_path = sequence_dir.parent / "init.txt"
    init_path.write_text(f"{init_box[0]},{init_box[1]},{init_box[2]},{init_box[3]}")

    cmd = [sys.executable, str(Path(__file__).parents[1] / "examples" / "run_tracker.py"),
           "--sequence", str(sequence_dir),
           "--init-box", str(init_path),
           "--output", str(output_path)]
    if use_deep:
        cmd.append("--deep")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ 跟踪失败: {result.stderr[:200]}")
        return False
    return True


def visualize_tracking(frames_dir: Path, pred_path: Path, output_video: Path, fps: float = 30):
    """生成跟踪可视化视频。"""
    cmd = [sys.executable, str(Path(__file__).parents[1] / "examples" / "visualize_tracking.py"),
           "--frames", str(frames_dir),
           "--pred", str(pred_path),
           "--output", str(output_video)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ 可视化失败: {result.stderr[:200]}")
        return False
    return True


def analyze_tracking(pred_path: Path, init_box: tuple[int, int, int, int]):
    """分析跟踪质量。"""
    preds = []
    with open(pred_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                preds.append([float(p) for p in parts])

    if not preds:
        return

    centers = [(p[0] + p[2] / 2, p[1] + p[3] / 2) for p in preds]
    x_vals = [c[0] for c in centers]
    y_vals = [c[1] for c in centers]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    print(f"\n📊 跟踪质量分析:")
    print(f"  初始框: ({init_box[0]}, {init_box[1]}, {init_box[2]}, {init_box[3]})")
    print(f"  跟踪帧数: {len(preds)}")
    print(f"  X漂移: {x_min:.0f} ~ {x_max:.0f}  (共 {x_max-x_min:.0f}px)")
    print(f"  Y漂移: {y_min:.0f} ~ {y_max:.0f}  (共 {y_max-y_min:.0f}px)")

    # 检测跳变
    for i in range(1, len(preds)):
        dx = abs(preds[i][0] - preds[i-1][0])
        dy = abs(preds[i][1] - preds[i-1][1])
        if dx > 100 or dy > 100:
            print(f"  ⚠️ 第{i}帧发生跳变: ({int(preds[i-1][0])},{int(preds[i-1][1])}) -> ({int(preds[i][0])},{int(preds[i][1])})")


def main():
    parser = argparse.ArgumentParser(description="一站式视频跟踪工具")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--output", required=True, help="输出跟踪视频路径")
    parser.add_argument("--init-box", type=str, help="初始框 'x,y,w,h'")
    parser.add_argument("--interactive", action="store_true", help="交互式选框")
    parser.add_argument("--no-visualize", action="store_true", help="跳过可视化")
    parser.add_argument("--deep", action="store_true", help="使用深度特征模式")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pano_tracking_") as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / "frames"
        pred_path = temp_path / "pred.txt"

        print("=" * 50)
        print("📹 步骤1/4: 视频转帧")
        print("=" * 50)
        num_frames, fps, width, height = video_to_frames(video_path, frames_dir)
        print(f"  视频信息: {width}x{height}, {fps:.1f} FPS, {num_frames} 帧")

        print("\n" + "=" * 50)
        print("🎯 步骤2/4: 目标选择")
        print("=" * 50)

        if args.init_box:
            parts = args.init_box.split(",")
            init_box = tuple(int(p.strip()) for p in parts)
            print(f"  使用指定初始框: {init_box}")
        elif args.interactive:
            first_frame = cv2.imread(str(sorted(frames_dir.iterdir())[0]))
            init_box = interactive_select(first_frame)
            if not init_box:
                print("  ❌ 用户取消选择")
                return
            print(f"  选中初始框: {init_box}")
        else:
            first_frame = cv2.imread(str(sorted(frames_dir.iterdir())[0]))
            targets = detect_target(first_frame)
            if not targets:
                print("  ❌ 未检测到目标，请使用 --interactive 模式")
                return
            init_box = targets[0][:4]
            print(f"  自动检测到 {len(targets)} 个候选目标")
            print(f"  自动选择: {init_box}")

        print("\n" + "=" * 50)
        print("🔍 步骤3/4: 目标跟踪")
        print("=" * 50)
        print(f"  模式: {'深度特征' if args.deep else '手工特征'}")
        if not run_tracker(frames_dir, init_box, pred_path, args.deep):
            return

        if not args.no_visualize:
            print("\n" + "=" * 50)
            print("🎬 步骤4/4: 生成可视化视频")
            print("=" * 50)
            if not visualize_tracking(frames_dir, pred_path, output_path, fps):
                return

        analyze_tracking(pred_path, init_box)

        print(f"\n✅ 跟踪完成! 输出视频: {output_path}")

        if args.keep_temp:
            import shutil
            shutil.copytree(temp_dir, "data/temp_tracking", dirs_exist_ok=True)
            print(f"  临时文件已保留: data/temp_tracking")


if __name__ == "__main__":
    main()
