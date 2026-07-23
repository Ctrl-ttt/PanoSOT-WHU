from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot import build_tracker
from panosot.geometry import erp_bbox_to_state, state_to_erp_bbox


def generate_test_frame(height: int = 720, width: int = 1440):
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    target_w, target_h = 200, 200
    
    for c in range(3):
        frame[center_y-target_h//2:center_y+target_h//2, center_x-target_w//2:center_x+target_w//2, c] = [255, 100, 100][c]
    
    return frame, np.array([center_x-target_w//2, center_y-target_h//2, target_w, target_h], dtype=np.float32)


def main():
    frame, init_box = generate_test_frame()
    h, w = frame.shape[:2]
    
    print(f"Frame shape: {h}x{w}")
    print(f"Init box: {init_box}")
    
    state = erp_bbox_to_state(init_box, w, h)
    print(f"\nInitial state:")
    print(f"  lon: {state.lon:.4f}")
    print(f"  lat: {state.lat:.4f}")
    print(f"  equatorial_width: {state.equatorial_width:.4f}")
    print(f"  angular_height: {state.angular_height:.4f}")
    
    bbox_back = state_to_erp_bbox(state, w, h)
    print(f"\nBack to bbox: {bbox_back}")
    
    print("\n=== Testing Deep Tracker ===")
    tracker = build_tracker(use_deep_features=True, device="cuda")
    
    tracker.initialize(frame, init_box)
    print(f"\nAfter initialize:")
    print(f"  state.lon: {tracker.state.lon:.4f}")
    print(f"  state.lat: {tracker.state.lat:.4f}")
    print(f"  state.equatorial_width: {tracker.state.equatorial_width:.4f}")
    print(f"  state.angular_height: {tracker.state.angular_height:.4f}")
    
    pred_bbox = state_to_erp_bbox(tracker.state, w, h)
    print(f"  Predicted bbox: {pred_bbox}")
    
    print(f"\nTemplates: {len(tracker._templates)}")
    for i, t in enumerate(tracker._templates):
        print(f"  Template {i}: shape={t.shape}, type={tracker._template_types[i]}")
    
    print("\n=== Tracking one frame ===")
    frame2 = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    target_w, target_h = 200, 200
    center_x, center_y = w // 2 + 50, h // 2 + 20
    for c in range(3):
        frame2[center_y-target_h//2:center_y+target_h//2, center_x-target_w//2:center_x+target_w//2, c] = [255, 100, 100][c]
    
    result_bbox = tracker.track(frame2)
    print(f"Result bbox: {result_bbox}")
    print(f"\nAfter track:")
    print(f"  state.lon: {tracker.state.lon:.4f}")
    print(f"  state.lat: {tracker.state.lat:.4f}")
    print(f"  state.equatorial_width: {tracker.state.equatorial_width:.4f}")
    print(f"  state.angular_height: {tracker.state.angular_height:.4f}")


if __name__ == "__main__":
    main()