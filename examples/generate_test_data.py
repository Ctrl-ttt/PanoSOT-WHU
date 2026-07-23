from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path


def generate_test_sequence(output_dir: Path, num_frames: int = 30):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    height, width = 720, 1440
    center_x, center_y = width // 2, height // 2
    target_w, target_h = 200, 200
    velocity_x, velocity_y = 5, 2
    
    cx, cy = center_x, center_y
    
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        cy_min = max(0, cy - target_h//2)
        cy_max = min(height, cy + target_h//2)
        cx_min = max(0, cx - target_w//2)
        cx_max = min(width, cx + target_w//2)
        
        frame[cy_min:cy_max, cx_min:cx_max, 0] = 255
        frame[cy_min:cy_max, cx_min:cx_max, 1] = 100
        frame[cy_min:cy_max, cx_min:cx_max, 2] = 100
        
        cx = (cx + velocity_x) % width
        cy = min(max(cy + velocity_y, target_h), height - target_h)
        if cy <= target_h or cy >= height - target_h:
            velocity_y *= -1
        
        img = Image.fromarray(frame)
        img.save(output_dir / f"{i+1:08d}.jpg", quality=90)
    
    init_box = f"{center_x-target_w//2},{center_y-target_h//2},{target_w},{target_h}"
    (output_dir.parent / "init.txt").write_text(init_box)
    
    print(f"Generated {num_frames} test frames to {output_dir}")
    print(f"Initial box saved to init.txt: {init_box}")


if __name__ == "__main__":
    generate_test_sequence(Path("data/test"))