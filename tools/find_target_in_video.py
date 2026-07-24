from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def find_target_in_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    success, first_frame = cap.read()
    if not success:
        print("Failed to read first frame")
        return
    
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if 5000 < area < 500000 and h > w * 0.8 and h < w * 3:
            candidates.append((x, y, w, h, area))
    
    candidates.sort(key=lambda c: c[4], reverse=True)
    
    overlay = first_frame.copy()
    
    print(f"\nFound {len(candidates)} candidate objects:")
    for i, (x, y, w, h, area) in enumerate(candidates[:10]):
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(overlay, f"T{i}: {x},{y},{w},{h}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"  Target {i}: ({x}, {y}, {w}, {h}) area={area}")
    
    cv2.imwrite("data/target_candidates.jpg", overlay)
    print(f"\nSaved to: data/target_candidates.jpg")
    
    if candidates:
        best = candidates[0]
        init_box = f"{best[0]},{best[1]},{best[2]},{best[3]}"
        print(f"\nRecommended init box: {init_box}")
        (Path("data") / "init_real.txt").write_text(init_box)
        print("Saved to data/init_real.txt")
    
    cap.release()


if __name__ == "__main__":
    video_path = Path(r"d:\yingshi\PanoSOT-WHU\data\external_videos\tracking_pexels_360_36157408.mp4")
    find_target_in_video(video_path)