from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes


def trace_tracking_path():
    frames_dir = Path("data/vtest_frames")
    pred_path = Path("data/pred_vtest.txt")
    init_path = Path("data/init_vtest.txt")
    
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    predictions = load_boxes(pred_path)
    init_box = load_boxes(init_path)[0]
    
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    
    centers = []
    for pred in predictions:
        cx = int(pred[0] + pred[2] / 2)
        cy = int(pred[1] + pred[3] / 2)
        centers.append((cx, cy))
    
    init_cx = int(init_box[0] + init_box[2] / 2)
    init_cy = int(init_box[1] + init_box[3] / 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/trace_path.mp4', fourcc, 10.0, (width, height))
    
    print("Creating tracking path video...")
    
    for i, (frame_path, pred) in enumerate(zip(frame_paths, predictions)):
        frame = cv2.imread(str(frame_path))
        
        cv2.rectangle(frame, 
                      (int(pred[0]), int(pred[1])), 
                      (int(pred[0] + pred[2]), int(pred[1] + pred[3])), 
                      (0, 0, 255), 3)
        
        cx, cy = centers[i]
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        if i > 0:
            for j in range(max(0, i-20), i):
                alpha = (j - max(0, i-20)) / 20.0
                prev_cx, prev_cy = centers[j]
                cv2.line(frame, (prev_cx, prev_cy), (cx, cy), 
                         (0, int(255*alpha), 0), 2)
        
        cv2.circle(frame, (init_cx, init_cy), 8, (0, 0, 255), 2)
        cv2.putText(frame, f"Init", (init_cx+10, init_cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Center: ({cx},{cy})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"\nSaved tracking path video to data/trace_path.mp4")
    
    print("\n=== Tracking Path Analysis ===")
    print(f"Initial center: ({init_cx}, {init_cy})")
    print(f"Final center: ({centers[-1][0]}, {centers[-1][1]})")
    print(f"Total movement: {int(np.sqrt((centers[-1][0]-init_cx)**2 + (centers[-1][1]-init_cy)**2))} pixels")
    
    dx = centers[-1][0] - init_cx
    dy = centers[-1][1] - init_cy
    print(f"Direction: ({dx:+.0f}, {dy:+.0f}) pixels")
    
    print("\n=== Frame-by-frame movement (every 10 frames) ===")
    for i in range(0, len(centers), 10):
        if i > 0:
            prev = centers[i-10]
            curr = centers[i]
            dist = int(np.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2))
            print(f"  Frame {i-10}→{i}: ({prev[0]},{prev[1]}) → ({curr[0]},{curr[1]}), dist={dist}px")


if __name__ == "__main__":
    trace_tracking_path()