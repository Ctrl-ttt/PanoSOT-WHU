from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def detect_stable_targets(video_path: Path, num_frames: int = 10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    all_detections = []
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        success, frame = cap.read()
        if not success:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = float(w) / max(h, 1)
            
            if 5000 < area < 1000000 and 0.3 < aspect_ratio < 3.0:
                detections.append((x, y, w, h, area))
        
        detections.sort(key=lambda d: d[4], reverse=True)
        all_detections.append(detections[:20])
    
    cap.release()
    
    stable_targets = []
    for frame_idx, detections in enumerate(all_detections):
        for det in detections[:10]:
            x, y, w, h, area = det
            stable = True
            for other_frame, other_dets in enumerate(all_detections):
                if other_frame == frame_idx:
                    continue
                found = False
                for other_det in other_dets[:10]:
                    ox, oy, ow, oh, oarea = other_det
                    if abs(x - ox) < 200 and abs(y - oy) < 200 and abs(area - oarea) < area * 0.5:
                        found = True
                        break
                if not found:
                    stable = False
                    break
            if stable:
                stable_targets.append((x, y, w, h, area))
    
    return stable_targets


def main():
    video_path = Path(r"d:\yingshi\PanoSOT-WHU\data\external_videos\tracking_pexels_360_36157408.mp4")
    
    print("Analyzing video for stable targets...")
    targets = detect_stable_targets(video_path)
    
    cap = cv2.VideoCapture(str(video_path))
    success, first_frame = cap.read()
    cap.release()
    
    if not success:
        print("Failed to read first frame")
        return
    
    overlay = first_frame.copy()
    
    print(f"\nFound {len(targets)} stable targets across frames:")
    for i, (x, y, w, h, area) in enumerate(targets[:5]):
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 6)
        cv2.putText(overlay, f"T{i}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        print(f"  Target {i}: ({x}, {y}, {w}, {h}) area={area}")
    
    output_path = Path("data/stable_targets.jpg")
    cv2.imwrite(str(output_path), overlay)
    print(f"\nSaved to: {output_path}")
    
    if targets:
        best = targets[0]
        init_box = f"{best[0]},{best[1]},{best[2]},{best[3]}"
        (Path("data") / "init_real.txt").write_text(init_box)
        print(f"\nRecommended init box saved to data/init_real.txt: {init_box}")
        print("\nRun tracking with:")
        print(f"  python examples/run_tracker.py --sequence data/real_video_frames --init-box data/init_real.txt --output data/pred_real.txt --deep")
        print(f"  python examples/visualize_tracking.py --frames data/real_video_frames --pred data/pred_real.txt --output data/real_video_tracking.mp4")


if __name__ == "__main__":
    main()