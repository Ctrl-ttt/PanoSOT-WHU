from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def main():
    first_frame_path = Path("data/first_frame_real.jpg")
    pred_path = Path("data/pred_real.txt")
    marked_path = Path("data/video_targets_marked.jpg")
    
    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        print("Failed to load first frame")
        return
    
    preds = []
    with open(pred_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                preds.append([float(p) for p in parts])
    
    print(f"Video size: {first_frame.shape[1]}x{first_frame.shape[0]}")
    print(f"Total predictions: {len(preds)}")
    print(f"\nInitial box: {preds[0]}")
    print(f"Final box: {preds[-1]}")
    
    center_x_0 = preds[0][0] + preds[0][2] / 2
    center_y_0 = preds[0][1] + preds[0][3] / 2
    center_x_1 = preds[-1][0] + preds[-1][2] / 2
    center_y_1 = preds[-1][1] + preds[-1][3] / 2
    
    print(f"\nCenter movement:")
    print(f"  Start: ({center_x_0:.1f}, {center_y_0:.1f})")
    print(f"  End: ({center_x_1:.1f}, {center_y_1:.1f})")
    print(f"  Delta: ({center_x_1 - center_x_0:.1f}, {center_y_1 - center_y_0:.1f})")
    
    overlay = first_frame.copy()
    x, y, w, h = [int(v) for v in preds[0]]
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 10)
    cv2.putText(overlay, "INITIAL TARGET", (x, y-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    
    cv2.imwrite("data/target_on_first_frame.jpg", overlay)
    print("\nSaved target on first frame: data/target_on_first_frame.jpg")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print("1. Open data/target_on_first_frame.jpg to see what we're tracking")
    print("2. If the target is not visible or wrong, open data/video_targets_marked.jpg")
    print("3. Choose a different target by editing data/init_real.txt")
    print("4. Format: x,y,width,height")
    print("\nTry these alternative targets:")
    print("  Target 2 (right side): 2939,1253,621,806")
    print("  Target 3 (center-right): 1503,1195,591,800")
    print("  Target 4: 2010,1067,612,755")
    print("  Target 6 (left side): 451,271,451,907")


if __name__ == "__main__":
    main()