from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes


def debug_tracking():
    frames_dir = Path("data/vtest_frames")
    pred_path = Path("data/pred_vtest.txt")
    init_path = Path("data/init_vtest.txt")
    
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    predictions = load_boxes(pred_path)
    
    init_box = load_boxes(init_path)[0]
    print(f"Initial box: {init_box}")
    print(f"Frame count: {len(frame_paths)}")
    print(f"Prediction count: {len(predictions)}")
    
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    
    overlay = first_frame.copy()
    
    cv2.rectangle(overlay, 
                  (int(init_box[0]), int(init_box[1])), 
                  (int(init_box[0] + init_box[2]), int(init_box[1] + init_box[3])), 
                  (0, 255, 0), 4)
    cv2.putText(overlay, f"INIT: ({int(init_box[0])},{int(init_box[1])},{int(init_box[2])},{int(init_box[3])})", 
                (int(init_box[0]), int(init_box[1]) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite("data/debug_initial.jpg", overlay)
    print("\nSaved data/debug_initial.jpg with initial box in GREEN")
    
    print("\n=== Tracking Analysis ===")
    for i in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
        if i < len(predictions):
            pred = predictions[i]
            frame = cv2.imread(str(frame_paths[i]))
            
            cv2.rectangle(frame, 
                          (int(pred[0]), int(pred[1])), 
                          (int(pred[0] + pred[2]), int(pred[1] + pred[3])), 
                          (0, 0, 255), 4)
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Pred: ({int(pred[0])},{int(pred[1])},{int(pred[2])},{int(pred[3])})", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imwrite(str(frames_dir.parent / f"debug_frame_{i}.jpg"), frame)
            print(f"Frame {i}: box=({int(pred[0])},{int(pred[1])},{int(pred[2])},{int(pred[3])})")
    
    print("\n=== What's wrong? ===")
    print("Looking at the tracking path:")
    print("- Frame 0: 510,270,50,140 (initial)")
    print("- Frame 10: 498,286,69,178 (slightly moving)")
    print("- Frame 20: 507,261,97,245 (BOX EXPLODING!)")
    print("- Frame 30: 479,224,137,336 (completely lost)")
    print("\nThe tracker loses the target around frame 20.")
    print("This is likely because:")
    print("1. Initial box doesn't accurately frame the target person")
    print("2. Template update causes drift")
    print("3. The tracker is following background instead of the person")


if __name__ == "__main__":
    debug_tracking()