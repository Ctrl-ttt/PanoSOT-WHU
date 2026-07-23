from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def find_correct_target():
    first_frame = cv2.imread("data/vtest_frames/00000001.jpg")
    height, width = first_frame.shape[:2]
    
    overlay = first_frame.copy()
    
    print("Analyzing first frame...")
    print(f"Frame size: {width}x{height}")
    
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    person_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 40 < w < 150 and 100 < h < 300 and h > w * 1.5:
            person_candidates.append((x, y, w, h))
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    print(f"\nFound {len(person_candidates)} person-like candidates:")
    for i, (x, y, w, h) in enumerate(person_candidates):
        print(f"  Candidate {i}: ({x}, {y}, {w}, {h})")
        cv2.putText(overlay, f"P{i}:{x},{y},{w},{h}", (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite("data/find_target_result.jpg", overlay)
    print(f"\nSaved data/find_target_result.jpg")
    
    if person_candidates:
        print("\nRecommended initial boxes:")
        for i, (x, y, w, h) in enumerate(person_candidates[:5]):
            print(f"  {i}: {x},{y},{w},{h}")
            print(f"    python examples/run_tracker.py --sequence data/vtest_frames --init-box <file_with_this_line> --output data/pred_vtest.txt")


if __name__ == "__main__":
    find_correct_target()