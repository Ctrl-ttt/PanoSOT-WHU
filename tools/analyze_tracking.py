from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes


def analyze_tracking(frames_dir: Path, pred_path: Path, init_box: tuple[int, int, int, int]):
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    predictions = load_boxes(pred_path)
    
    if len(frame_paths) != len(predictions):
        print(f"Warning: frame count ({len(frame_paths)}) != prediction count ({len(predictions)})")
    
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    print(f"Initial box: {init_box}")
    
    cv2.rectangle(first_frame, 
                  (init_box[0], init_box[1]), 
                  (init_box[0] + init_box[2], init_box[1] + init_box[3]), 
                  (0, 255, 0), 3)
    cv2.putText(first_frame, "Initial", (init_box[0], init_box[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    for i in [0, 10, 20, 30, 40, 50]:
        if i < len(frame_paths):
            frame = cv2.imread(str(frame_paths[i]))
            pred = predictions[i]
            
            cv2.rectangle(frame, 
                          (int(pred[0]), int(pred[1])), 
                          (int(pred[0] + pred[2]), int(pred[1] + pred[3])), 
                          (0, 0, 255), 3)
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Pred: ({int(pred[0])},{int(pred[1])})", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imwrite(str(frames_dir.parent / f"analysis_frame_{i}.jpg"), frame)
            print(f"Saved analysis_frame_{i}.jpg")
    
    cv2.imwrite(str(frames_dir.parent / "analysis_initial.jpg"), first_frame)
    print("\nSaved analysis_initial.jpg with initial box in green")
    
    print("\n=== Tracking Path ===")
    for i in range(0, len(predictions), 5):
        pred = predictions[i]
        print(f"Frame {i}: ({int(pred[0]):4d}, {int(pred[1]):4d}, {int(pred[2]):3d}, {int(pred[3]):3d})")


if __name__ == "__main__":
    analyze_tracking(Path("data/vtest_frames"), Path("data/pred_vtest.txt"), (510, 270, 50, 140))