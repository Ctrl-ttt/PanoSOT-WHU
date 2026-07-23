from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def draw_person(frame: np.ndarray, cx: int, cy: int, size: int):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    half_size = size // 2
    head_radius = int(size * 0.15)
    body_width = int(size * 0.25)
    body_height = int(size * 0.5)
    arm_length = int(size * 0.2)
    leg_length = int(size * 0.4)
    
    head_x, head_y = cx, cy - half_size + head_radius
    draw.ellipse([head_x - head_radius, head_y - head_radius, 
                  head_x + head_radius, head_y + head_radius], 
                  fill=(255, 220, 180), outline=(139, 90, 43), width=2)
    
    body_top = head_y + head_radius
    body_bottom = body_top + body_height
    draw.rectangle([cx - body_width, body_top, 
                    cx + body_width, body_bottom], 
                    fill=(0, 100, 200), outline=(0, 50, 100), width=2)
    
    arm_y = body_top + int(body_height * 0.3)
    draw.line([cx - body_width, arm_y, cx - body_width - arm_length, arm_y], 
              fill=(255, 220, 180), width=4)
    draw.line([cx + body_width, arm_y, cx + body_width + arm_length, arm_y], 
              fill=(255, 220, 180), width=4)
    
    leg_y_start = body_bottom
    draw.line([cx - body_width // 2, leg_y_start, cx - body_width // 2, leg_y_start + leg_length], 
              fill=(0, 50, 100), width=4)
    draw.line([cx + body_width // 2, leg_y_start, cx + body_width // 2, leg_y_start + leg_length], 
              fill=(0, 50, 100), width=4)
    
    return np.array(img)


def generate_test_sequence(output_dir: Path, num_frames: int = 60):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    height, width = 720, 1440
    center_x, center_y = width // 2, height // 2
    target_w, target_h = 200, 300
    velocity_x, velocity_y = 6, 1
    
    cx, cy = center_x, center_y
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        frame[:, :, 0] = 30 + np.random.randint(0, 20, (height, width))
        frame[:, :, 1] = 50 + np.random.randint(0, 20, (height, width))
        frame[:, :, 2] = 70 + np.random.randint(0, 20, (height, width))
        
        for y in range(0, height, 50):
            for x in range(0, width, 100):
                frame[y:y+2, x:x+50] = [40, 60, 80]
        
        frame = draw_person(frame, cx, cy, max(target_w, target_h))
        
        cx = (cx + velocity_x) % width
        cy = min(max(cy + velocity_y, target_h), height - target_h)
        if cy <= target_h or cy >= height - target_h:
            velocity_y *= -1
        
        img = Image.fromarray(frame)
        img.save(output_dir / f"{i+1:08d}.jpg", quality=90)
    
    init_box = f"{center_x-target_w//2},{center_y-target_h//2},{target_w},{target_h}"
    (output_dir.parent / "init_person.txt").write_text(init_box)
    
    gts = []
    cx, cy = center_x, center_y
    for i in range(num_frames):
        x = cx - target_w//2
        y = cy - target_h//2
        gts.append(f"{x},{y},{target_w},{target_h}")
        
        cx = (cx + velocity_x) % width
        cy = min(max(cy + velocity_y, target_h), height - target_h)
        if cy <= target_h or cy >= height - target_h:
            velocity_y *= -1
    
    (output_dir.parent / "gt_person.txt").write_text("\n".join(gts))
    
    print(f"Generated {num_frames} person test frames to {output_dir}")
    print(f"Initial box: {init_box}")
    print(f"Ground truth saved to gt_person.txt")


if __name__ == "__main__":
    generate_test_sequence(Path("data/person_test"))