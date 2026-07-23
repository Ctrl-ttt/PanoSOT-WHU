from __future__ import annotations

import numpy as np
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker


def generate_test_frames(num_frames: int = 50, height: int = 720, width: int = 1440) -> list:
    frames = []
    center_x, center_y = width // 2, height // 2
    target_w, target_h = 200, 200
    velocity_x, velocity_y = 5, 2
    
    cx, cy = center_x, center_y
    
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        for c in range(3):
            frame[cy - target_h//2:cy + target_h//2, cx - target_w//2:cx + target_w//2, c] = [
                255 if c == 0 else 100,
                100 if c == 1 else 255,
                100 if c == 2 else 100,
            ][c]
        
        cx = (cx + velocity_x) % width
        cy = min(max(cy + velocity_y, target_h), height - target_h)
        
        if cy <= target_h or cy >= height - target_h:
            velocity_y *= -1
        
        frames.append(frame)
    
    return frames, np.array([center_x - target_w//2, center_y - target_h//2, target_w, target_h], dtype=np.float32)


def main() -> None:
    print("Generating test frames...")
    frames, init_box = generate_test_frames(num_frames=30)
    print(f"Generated {len(frames)} frames, init_box: {init_box}")
    
    print("\n=== Testing Handcrafted Feature Mode ===")
    tracker = build_tracker(use_deep_features=False)
    
    start_time = time.time()
    predictions = tracker.track_sequence(iter(frames), init_box)
    elapsed = time.time() - start_time
    
    fps = len(frames) / elapsed
    print(f"Handcrafted mode - FPS: {fps:.2f}, Time: {elapsed:.2f}s")
    
    print("\n=== Testing Deep Feature Mode ===")
    tracker_deep = build_tracker(use_deep_features=True, device="cuda")
    
    start_time = time.time()
    predictions_deep = tracker_deep.track_sequence(iter(frames), init_box)
    elapsed = time.time() - start_time
    
    fps_deep = len(frames) / elapsed
    print(f"Deep mode - FPS: {fps_deep:.2f}, Time: {elapsed:.2f}s")
    print(f"Device: {tracker_deep.config.device}")
    
    print("\n=== Results ===")
    print(f"Handcrafted mode: {len(predictions)} predictions")
    print(f"Deep mode: {len(predictions_deep)} predictions")
    print(f"First prediction (handcrafted): {predictions[0]}")
    print(f"Last prediction (handcrafted): {predictions[-1]}")
    print(f"First prediction (deep): {predictions_deep[0]}")
    print(f"Last prediction (deep): {predictions_deep[-1]}")
    
    print("\n✅ Tracker test completed successfully!")


if __name__ == "__main__":
    main()