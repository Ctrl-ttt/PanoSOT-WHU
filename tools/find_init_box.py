from __future__ import annotations

import cv2
from pathlib import Path


def show_first_frame(frames_dir: Path):
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if not frame_paths:
        print("No frames found")
        return
    
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    
    cv2.imwrite("data/first_frame.jpg", first_frame)
    print("First frame saved to: data/first_frame.jpg")
    print("\nPlease open this image and find the target bounding box.")
    print("Format: x,y,width,height")
    print("Example: If the target starts at (200, 150) with size 80x120, then init box is: 200,150,80,120")


if __name__ == "__main__":
    show_first_frame(Path("data/vtest_frames"))