from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def analyze_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Size: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    success, first_frame = cap.read()
    if not success:
        print("Failed to read first frame")
        return
    
    cv2.imwrite("data/first_frame_real.jpg", first_frame)
    print(f"\nFirst frame saved to: data/first_frame_real.jpg")
    print(f"Open this image to find a visible target to track.")
    print(f"Then run:")
    print(f"  python tools/video_to_frames.py --video '{video_path}' --output data/real_video_frames")
    print(f"  python examples/run_tracker.py --sequence data/real_video_frames --init-box data/init_real.txt --output data/pred_real.txt")
    
    cap.release()


if __name__ == "__main__":
    video_path = Path(r"d:\yingshi\PanoSOT-WHU\data\external_videos\tracking_pexels_360_36157408.mp4")
    analyze_video(video_path)