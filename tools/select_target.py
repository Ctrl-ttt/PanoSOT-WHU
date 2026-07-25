"""视频目标选择工具。

从视频第一帧中自动检测候选目标，或交互式手动选择。

用法:
  # 自动检测候选目标
  python tools/select_target.py --video data/video.mp4

  # 交互式手动选框
  python tools/select_target.py --video data/video.mp4 --interactive

  # 指定已有帧图片
  python tools/select_target.py --frame data/first_frame.jpg
"""
from __future__ import annotations

import argparse
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 自动目标检测
# ---------------------------------------------------------------------------

def detect_candidates(frame: np.ndarray, max_targets: int = 8):
    """使用边缘检测 + 轮廓分析 + 颜色检测找出候选目标。"""
    candidates = []

    # 1. 边缘轮廓检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 10000 < area < 800000 and 0.3 < w / max(h, 1) < 3.0:
            candidates.append((x, y, w, h, area, "edge"))

    # 2. 颜色检测（红/橙/黄/绿/蓝/紫）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = [
        ((0, 50, 50), (10, 255, 255), "red"),
        ((170, 50, 50), (180, 255, 255), "red"),
        ((10, 50, 50), (30, 255, 255), "orange"),
        ((80, 50, 50), (130, 255, 255), "green"),
        ((130, 50, 50), (160, 255, 255), "blue"),
    ]
    for lower, upper, name in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if 5000 < area < 500000:
                candidates.append((x, y, w, h, area, name))

    # 按面积排序，去重
    candidates.sort(key=lambda c: c[4], reverse=True)
    unique = []
    used = set()
    for x, y, w, h, area, label in candidates:
        key = (round(x / 100), round(y / 100))
        if key not in used:
            used.add(key)
            unique.append((x, y, w, h, area, label))

    return unique[:max_targets]


def find_stable_targets(video_path: Path, num_samples: int = 5, max_targets: int = 8):
    """跨多帧检测，返回出现频率最高的稳定目标。"""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // num_samples)

    frame_detections = []
    for i in range(num_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ok, frame = cap.read()
        if ok:
            frame_detections.append(detect_candidates(frame))
    cap.release()

    if not frame_detections:
        return []

    # 统计每个位置出现的次数
    from collections import Counter
    pos_counter = Counter()
    pos_best = {}
    for dets in frame_detections:
        for x, y, w, h, area, label in dets:
            key = (round(x / 100), round(y / 100))
            pos_counter[key] += 1
            if key not in pos_best or area > pos_best[key][4]:
                pos_best[key] = (x, y, w, h, area, label)

    stable = []
    for key, count in pos_counter.most_common(max_targets):
        if count >= 2:
            x, y, w, h, area, label = pos_best[key]
            stable.append((x, y, w, h, area, label, count))

    return stable


# ---------------------------------------------------------------------------
# 交互式选框
# ---------------------------------------------------------------------------

class InteractiveSelector:
    """鼠标拖拽选择初始框。"""

    def __init__(self, image: np.ndarray):
        self.image = image.copy()
        self.clone = image.copy()
        self.start = None
        self.box = None

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.image = self.clone.copy()
            cv2.rectangle(self.image, self.start, (x, y), (0, 0, 255), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            x1, y1 = min(self.start[0], x), min(self.start[1], y)
            x2, y2 = max(self.start[0], x), max(self.start[1], y)
            self.box = (x1, y1, x2 - x1, y2 - y1)

    def run(self) -> tuple[int, int, int, int] | None:
        h, w = self.image.shape[:2]
        cv2.namedWindow("Select Target", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Target", min(w, 1200), min(h, 800))
        cv2.setMouseCallback("Select Target", self._on_mouse)

        print("操作说明:")
        print("  鼠标拖拽选框 -> 按 s 保存 -> 按 q 退出")
        print("  按 r 重置")

        while True:
            cv2.imshow("Select Target", self.image)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s") and self.box:
                cv2.destroyAllWindows()
                return self.box
            elif key == ord("r"):
                self.image = self.clone.copy()
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return None
        return None


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="视频目标选择工具")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--frame", type=str, help="已有帧图片路径（替代视频）")
    parser.add_argument("--interactive", action="store_true", help="交互式手动选框")
    parser.add_argument("--output", default="data/init.txt", help="输出 init.txt 路径")
    args = parser.parse_args()

    # 读取第一帧
    if args.frame:
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"无法读取图片: {args.frame}")
            return
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"无法读取视频: {args.video}")
            return
    else:
        print("请指定 --video 或 --frame")
        return

    h, w = frame.shape[:2]
    print(f"帧尺寸: {w}x{h}")

    # 保存第一帧
    first_frame_path = Path("data/first_frame.jpg")
    first_frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(first_frame_path), frame)
    print(f"首帧已保存: {first_frame_path}")

    if args.interactive:
        # 交互式模式
        selector = InteractiveSelector(frame)
        box = selector.run()
        if box:
            x, y, bw, bh = box
            print(f"选中区域: x={x}, y={y}, w={bw}, h={bh}")
            Path(args.output).write_text(f"{x},{y},{bw},{bh}")
            print(f"已保存到: {args.output}")
        else:
            print("未选择目标")
    else:
        # 自动检测模式
        print("\n自动检测候选目标...")
        if args.video:
            targets = find_stable_targets(Path(args.video))
        else:
            targets = detect_candidates(frame)

        if not targets:
            print("未检测到候选目标，请尝试 --interactive 模式")
            return

        # 标记并保存
        overlay = frame.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
                  (255, 255, 0), (255, 0, 255), (128, 0, 255), (255, 128, 0)]

        print(f"\n检测到 {len(targets)} 个候选目标:")
        for i, item in enumerate(targets):
            if len(item) == 7:
                x, y, bw, bh, area, label, count = item
                print(f"  T{i} ({label}, 出现{count}次): ({x}, {y}, {bw}, {bh})")
            else:
                x, y, bw, bh, area, label = item
                count = 1
                print(f"  T{i} ({label}): ({x}, {y}, {bw}, {bh})")

            color = colors[i % len(colors)]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 6)
            cv2.putText(overlay, f"T{i}", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

        marked_path = Path("data/target_candidates.jpg")
        cv2.imwrite(str(marked_path), overlay)
        print(f"\n标记图已保存: {marked_path}")

        # 自动选择最佳目标
        best = targets[0]
        x, y, bw, bh = best[0], best[1], best[2], best[3]
        Path(args.output).write_text(f"{x},{y},{bw},{bh}")
        print(f"自动选择 T0，已保存到: {args.output}")
        print("如需选择其他目标，请手动修改该文件或使用 --interactive 模式")


if __name__ == "__main__":
    main()
