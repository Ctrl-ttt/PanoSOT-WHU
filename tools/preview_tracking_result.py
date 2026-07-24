from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def main():
    frames_dir = Path("data/real_video_frames")
    pred_path = Path("data/pred_real.txt")
    
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    
    preds = []
    with open(pred_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                preds.append([float(p) for p in parts])
    
    key_frames = [0, 15, 30, 45, 59]
    
    for idx in key_frames:
        if idx < len(frame_paths) and idx < len(preds):
            frame = cv2.imread(str(frame_paths[idx]))
            if frame is not None:
                x, y, w, h = [int(v) for v in preds[idx]]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 10)
                cv2.putText(frame, f"Frame {idx}", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
                
                height, width = frame.shape[:2]
                scale = min(800 / width, 600 / height)
                small = cv2.resize(frame, None, fx=scale, fy=scale)
                
                output_path = Path(f"data/tracking_frame_{idx}.jpg")
                cv2.imwrite(str(output_path), small)
                print(f"Saved frame {idx}: {output_path}")
    
    print("\n✅ Done! Check the tracking frames in data/")
    print("Look for the RED rectangle - it should be tracking a visible object!")


if __name__ == "__main__":
    main()