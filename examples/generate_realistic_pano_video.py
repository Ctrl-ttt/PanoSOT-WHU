from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import cv2


def generate_pano_video(output_dir: Path, num_frames: int = 50):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    height, width = 720, 1440
    center_x, center_y = width // 2, height // 2
    target_w, target_h = 120, 200
    
    velocity_x = 15
    velocity_y = 2
    
    cx, cy = center_x, center_y
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        sky_gradient = np.linspace([135, 206, 235], [70, 130, 180], height)
        for y in range(height):
            frame[y, :, :] = sky_gradient[y]
        
        ground_y = int(height * 0.6)
        ground_gradient = np.linspace([101, 67, 33], [80, 50, 20], height - ground_y)
        for y in range(ground_y, height):
            frame[y, :, :] = ground_gradient[y - ground_y]
        
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        for x in range(0, width, 100):
            tree_height = 100 + np.random.randint(-20, 20)
            draw.polygon([
                (x + 50, ground_y - tree_height),
                (x, ground_y),
                (x + 100, ground_y)
            ], fill=(34, 139, 34))
        
        for x in range(200, width - 200, 250):
            building_height = 80 + np.random.randint(-15, 15)
            draw.rectangle([x, ground_y - building_height, x + 60, ground_y],
                          fill=(200, 200, 200))
        
        body_color = (255, 80, 80)
        head_color = (255, 220, 180)
        leg_color = (30, 30, 80)
        
        head_radius = 25
        body_w = 30
        body_h = 80
        leg_h = 60
        
        draw.ellipse([cx - head_radius, cy - target_h//2,
                      cx + head_radius, cy - target_h//2 + head_radius*2],
                     fill=head_color)
        
        draw.rectangle([cx - body_w//2, cy - target_h//2 + head_radius*2,
                        cx + body_w//2, cy - target_h//2 + head_radius*2 + body_h],
                       fill=body_color)
        
        arm_swing = int(np.sin(i * 0.3) * 20)
        draw.line([cx - body_w//2, cy - target_h//2 + head_radius*2 + 25,
                   cx - body_w//2 - 35 + arm_swing, cy - target_h//2 + head_radius*2 + 40],
                  fill=head_color, width=8)
        draw.line([cx + body_w//2, cy - target_h//2 + head_radius*2 + 25,
                   cx + body_w//2 + 35 - arm_swing, cy - target_h//2 + head_radius*2 + 40],
                  fill=head_color, width=8)
        
        leg_swing = int(np.sin(i * 0.3 + np.pi) * 15)
        draw.line([cx - 8, cy - target_h//2 + head_radius*2 + body_h,
                   cx - 8 + leg_swing, cy + target_h//2],
                  fill=leg_color, width=8)
        draw.line([cx + 8, cy - target_h//2 + head_radius*2 + body_h,
                   cx + 8 - leg_swing, cy + target_h//2],
                  fill=leg_color, width=8)
        
        frame = np.array(img)
        
        road_y1 = ground_y - 15
        road_y2 = ground_y
        frame[road_y1:road_y2, :, :] = [50, 50, 50]
        
        for x in range(0, width, 150):
            frame[road_y1:road_y2, x:x+75, :] = [255, 255, 255]
        
        cx = (cx + velocity_x) % width
        if cx < target_w or cx > width - target_w:
            velocity_x = -velocity_x
        
        img = Image.fromarray(frame)
        img.save(output_dir / f"{i+1:08d}.jpg", quality=90)
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_frames} frames")
    
    init_box = f"{center_x-target_w//2},{center_y-target_h//2},{target_w},{target_h}"
    (output_dir.parent / "init_pano.txt").write_text(init_box)
    
    gts = []
    cx, cy = center_x, center_y
    velocity_x = 15
    for i in range(num_frames):
        x = cx - target_w//2
        y = cy - target_h//2
        gts.append(f"{x},{y},{target_w},{target_h}")
        
        cx = (cx + velocity_x) % width
        if cx < target_w or cx > width - target_w:
            velocity_x = -velocity_x
    
    (output_dir.parent / "gt_pano.txt").write_text("\n".join(gts))
    
    print(f"\nGenerated {num_frames} realistic panorama test frames to {output_dir}")
    print(f"Initial box: {init_box}")


if __name__ == "__main__":
    generate_pano_video(Path("data/pano_test"))