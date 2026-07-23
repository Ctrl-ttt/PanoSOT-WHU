from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker


def generate_test_frames(num_frames=10, height=720, width=1440):
    frames = []
    center_x, center_y = width // 2, height // 2
    target_w, target_h = 200, 200
    velocity_x, velocity_y = 5, 2
    
    cx, cy = center_x, center_y
    
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        for c in range(3):
            frame[cy-target_h//2:cy+target_h//2, cx-target_w//2:cx+target_w//2, c] = [255, 100, 100][c]
        
        cx = (cx + velocity_x) % width
        cy = min(max(cy + velocity_y, target_h), height - target_h)
        if cy <= target_h or cy >= height - target_h:
            velocity_y *= -1
        
        frames.append(frame)
    
    return frames, np.array([center_x-target_w//2, center_y-target_h//2, target_w, target_h], dtype=np.float32)


def main():
    frames, init_box = generate_test_frames(num_frames=10)
    h, w = frames[0].shape[:2]
    
    print(f"=== Testing Deep Tracker - Multi Frame ===")
    tracker = build_tracker(use_deep_features=True, device="cuda")
    
    print(f"\nFrame 0 (init):")
    result = tracker.initialize(frames[0], init_box)
    print(f"  Result: {result}")
    print(f"  State: lon={tracker.state.lon:.4f}, lat={tracker.state.lat:.4f}, ew={tracker.state.equatorial_width:.4f}, ah={tracker.state.angular_height:.4f}")
    
    for i, frame in enumerate(frames[1:], start=1):
        result = tracker.track(frame)
        print(f"\nFrame {i}:")
        print(f"  Result: {result}")
        print(f"  State: lon={tracker.state.lon:.4f}, lat={tracker.state.lat:.4f}, ew={tracker.state.equatorial_width:.4f}, ah={tracker.state.angular_height:.4f}")
        print(f"  Lost frames: {tracker.lost_frames}")
        if result[2] > 1000:
            print(f"  WARNING: Width is too large!")
            break


if __name__ == "__main__":
    main()