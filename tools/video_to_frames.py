from __future__ import annotations

import cv2
import os
from pathlib import Path


def video_to_frames(video_path: Path, output_dir: Path, max_frames: int = 100):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = output_dir / f"{frame_count + 1:08d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_count += 1
        
        if frame_count % 20 == 0:
            print(f"Extracted {frame_count} frames")
    
    cap.release()
    print(f"Total frames extracted: {frame_count}")
    
    return frame_count


def create_init_file(output_dir: Path, init_box: str):
    init_path = output_dir.parent / "init_vtest.txt"
    init_path.write_text(init_box)
    print(f"Initial box saved to: {init_path}")


if __name__ == "__main__":
    video_path = Path("data/OpenCV_vtest.avi")
    output_dir = Path("data/vtest_frames")
    
    frame_count = video_to_frames(video_path, output_dir, max_frames=60)
    
    if frame_count:
        init_box = "280,200,60,80"
        create_init_file(output_dir, init_box)
        print(f"\nRun tracking with:")
        print(f"python examples/run_tracker.py --sequence data/vtest_frames --init-box data/init_vtest.txt --output data/pred_vtest.txt")