from __future__ import annotations

import argparse
import cv2
from pathlib import Path


def video_to_frames(video_path: Path, output_dir: Path, max_frames: int = 100, step: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    frame_count = 0
    saved_count = 0
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if (frame_count - 1) % max(step, 1) != 0:
            continue

        frame_path = output_dir / f"{saved_count + 1:08d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_count += 1

        if saved_count % 20 == 0:
            print(f"Extracted {saved_count} frames")
    
    cap.release()
    print(f"Total frames extracted: {saved_count}")

    return saved_count


def create_init_file(output_dir: Path, init_box: str):
    init_path = output_dir.parent / "init_vtest.txt"
    init_path.write_text(init_box)
    print(f"Initial box saved to: {init_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ordered JPG frames from a video.")
    parser.add_argument("--video", default="data/OpenCV_vtest.avi", help="Input video path.")
    parser.add_argument("--output-dir", default="data/vtest_frames", help="Output frame directory.")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum saved frames.")
    parser.add_argument("--step", type=int, default=1, help="Save one frame every N source frames.")
    parser.add_argument("--init-box", default=None, help="Optional init box string, e.g. x,y,w,h.")
    parser.add_argument("--init-output", default=None, help="Optional init box output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    frame_count = video_to_frames(video_path, output_dir, max_frames=args.max_frames, step=args.step)

    if frame_count and args.init_box:
        init_path = Path(args.init_output) if args.init_output else output_dir.parent / "init_vtest.txt"
        init_path.write_text(args.init_box, encoding="utf-8")
        print(f"Initial box saved to: {init_path}")


if __name__ == "__main__":
    main()
