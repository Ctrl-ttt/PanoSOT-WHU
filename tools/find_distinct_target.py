from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def find_colorful_objects(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    color_ranges = [
        ((0, 50, 50), (10, 255, 255), "red"),
        ((170, 50, 50), (180, 255, 255), "red"),
        ((10, 50, 50), (30, 255, 255), "orange"),
        ((30, 50, 50), (60, 255, 255), "yellow"),
        ((80, 50, 50), (130, 255, 255), "green"),
        ((100, 50, 50), (130, 255, 255), "cyan"),
        ((130, 50, 50), (160, 255, 255), "blue"),
        ((140, 50, 50), (170, 255, 255), "purple"),
    ]
    
    targets = []
    for (lower, upper, color_name) in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if 5000 < area < 500000:
                targets.append((x, y, w, h, area, color_name))
    
    targets.sort(key=lambda t: t[4], reverse=True)
    return targets


def find_high_contrast_objects(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast = np.absolute(laplacian)
    
    ret, thresh = cv2.threshold(contrast, 20, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    targets = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 10000 < area < 800000:
            targets.append((x, y, w, h, area))
    
    targets.sort(key=lambda t: t[4], reverse=True)
    return targets


def main():
    video_path = Path(r"d:\yingshi\PanoSOT-WHU\data\external_videos\tracking_pexels_360_36157408.mp4")
    
    cap = cv2.VideoCapture(str(video_path))
    success, first_frame = cap.read()
    cap.release()
    
    if not success:
        print("Failed to read first frame")
        return
    
    print("Finding colorful objects...")
    colorful = find_colorful_objects(first_frame)
    print(f"Found {len(colorful)} colorful objects")
    
    print("\nFinding high-contrast objects...")
    contrast = find_high_contrast_objects(first_frame)
    print(f"Found {len(contrast)} high-contrast objects")
    
    overlay = first_frame.copy()
    height, width = overlay.shape[:2]
    
    scale = min(1200 / width, 800 / height)
    small = cv2.resize(overlay, None, fx=scale, fy=scale)
    
    all_targets = []
    used_positions = set()
    
    for t in colorful[:5]:
        x, y, w, h, area, color_name = t
        pos_key = (round(x/100), round(y/100))
        if pos_key not in used_positions:
            used_positions.add(pos_key)
            all_targets.append((x, y, w, h, area, color_name))
    
    for t in contrast[:10]:
        x, y, w, h, area = t
        pos_key = (round(x/100), round(y/100))
        if pos_key not in used_positions:
            used_positions.add(pos_key)
            all_targets.append((x, y, w, h, area, "contrast"))
    
    print("\n=== Recommended Targets ===")
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
              (255, 255, 0), (255, 0, 255), (128, 0, 255), (255, 128, 0)]
    
    for i, (x, y, w, h, area, label) in enumerate(all_targets[:8]):
        cv2.rectangle(overlay, (x, y), (x+w, y+h), colors[i], 8)
        cv2.putText(overlay, f"T{i}:{label}", (x, y-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[i], 4)
        print(f"  Target {i} ({label}): ({x}, {y}, {w}, {h}) area={area}")
    
    output_path = Path("data/distinct_targets.jpg")
    cv2.imwrite(str(output_path), overlay)
    print(f"\nSaved marked image to: {output_path}")
    
    if all_targets:
        best = all_targets[0]
        init_box = f"{best[0]},{best[1]},{best[2]},{best[3]}"
        (Path("data") / "init_real.txt").write_text(init_box)
        print(f"\nAuto-selected: {init_box} (type: {best[5]})")
        print("\nRun tracking:")
        print(f"  python examples/run_tracker.py --sequence data/real_video_frames --init-box data/init_real.txt --output data/pred_real.txt --deep")


if __name__ == "__main__":
    main()