from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes


def analyze_tracking_video():
    frames_dir = Path("data/vtest_frames")
    pred_path = Path("data/pred_vtest.txt")
    init_path = Path("data/init_vtest.txt")
    
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    predictions = load_boxes(pred_path)
    init_box = load_boxes(init_path)[0]
    
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/debug_analysis.mp4', fourcc, 10.0, (width, height))
    
    print("Analyzing tracking frames...")
    
    for i, (frame_path, pred) in enumerate(zip(frame_paths, predictions)):
        frame = cv2.imread(str(frame_path))
        
        cv2.rectangle(frame, 
                      (int(pred[0]), int(pred[1])), 
                      (int(pred[0] + pred[2]), int(pred[1] + pred[3])), 
                      (0, 0, 255), 3)
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Pred: ({int(pred[0])},{int(pred[1])},{int(pred[2])},{int(pred[3])})", 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if i == 0:
            cv2.rectangle(frame, 
                          (int(init_box[0]), int(init_box[1])), 
                          (int(init_box[0] + init_box[2]), int(init_box[1] + init_box[3])), 
                          (0, 255, 0), 3)
            cv2.putText(frame, f"Init: ({int(init_box[0])},{int(init_box[1])})", 
                        (int(init_box[0]), int(init_box[1]) - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"\nSaved analysis video to data/debug_analysis.mp4")
    
    print("\n=== Tracking Statistics ===")
    print(f"Initial box: {init_box}")
    print(f"Frame count: {len(predictions)}")
    
    box_sizes = []
    for pred in predictions:
        box_sizes.append(pred[2] * pred[3])
    
    print(f"Box area: min={int(min(box_sizes))}, max={int(max(box_sizes))}, avg={int(sum(box_sizes)/len(box_sizes))}")
    print(f"Box area change: {int(box_sizes[-1] / box_sizes[0] * 100)}% of initial")
    
    if max(box_sizes) > min(box_sizes) * 3:
        print("\n⚠️ WARNING: Box size increased by more than 3x - tracker likely lost!")
    
    print("\n=== Last 10 predictions ===")
    for i in range(max(0, len(predictions)-10), len(predictions)):
        pred = predictions[i]
        print(f"  Frame {i}: ({int(pred[0])},{int(pred[1])},{int(pred[2])},{int(pred[3])})")


if __name__ == "__main__":
    analyze_tracking_video()