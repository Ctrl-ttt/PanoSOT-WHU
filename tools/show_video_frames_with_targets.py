from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def extract_key_frames(video_path: Path, num_frames: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        success, frame = cap.read()
        if success:
            frames.append((i * frame_interval, frame))
    
    cap.release()
    return frames


def detect_targets(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    targets = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 10000 < area < 800000:
            targets.append((x, y, w, h, area))
    
    targets.sort(key=lambda t: t[4], reverse=True)
    return targets[:8]


def main():
    video_path = Path(r"d:\yingshi\PanoSOT-WHU\data\external_videos\tracking_pexels_360_36157408.mp4")
    
    print("Extracting key frames...")
    key_frames = extract_key_frames(video_path)
    
    print(f"\nAnalyzing {len(key_frames)} key frames...")
    
    all_targets = []
    for frame_idx, frame in key_frames:
        targets = detect_targets(frame)
        all_targets.extend([(frame_idx, t) for t in targets])
        print(f"  Frame {frame_idx}: {len(targets)} targets found")
    
    all_targets.sort(key=lambda t: t[1][4], reverse=True)
    unique_targets = []
    used_positions = set()
    
    for frame_idx, (x, y, w, h, area) in all_targets:
        pos_key = (round(x/100), round(y/100))
        if pos_key not in used_positions:
            used_positions.add(pos_key)
            unique_targets.append((x, y, w, h, area))
    
    print(f"\nFound {len(unique_targets)} unique targets")
    
    _, first_frame = key_frames[0]
    overlay = first_frame.copy()
    
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), 
              (255, 255, 0), (255, 0, 255), (128, 0, 255), (255, 128, 0)]
    
    print("\nRecommended targets (large, stable objects):")
    for i, (x, y, w, h, area) in enumerate(unique_targets[:8]):
        cv2.rectangle(overlay, (x, y), (x+w, y+h), colors[i], 8)
        cv2.putText(overlay, f"T{i}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 6)
        print(f"  Target {i}: ({x}, {y}, {w}, {h}) area={area}")
    
    output_path = Path("data/video_targets_marked.jpg")
    cv2.imwrite(str(output_path), overlay)
    print(f"\nSaved marked image to: {output_path}")
    
    best_target = unique_targets[0]
    init_box = f"{best_target[0]},{best_target[1]},{best_target[2]},{best_target[3]}"
    (Path("data") / "init_real.txt").write_text(init_box)
    print(f"\nAuto-selected best target: {init_box}")
    print("\nRunning tracking with this target...")
    
    import subprocess
    result = subprocess.run([
        r"d:\yingshi\PanoSOT-WHU\.venv\Scripts\python.exe",
        "examples/run_tracker.py",
        "--sequence", "data/real_video_frames",
        "--init-box", "data/init_real.txt",
        "--output", "data/pred_real.txt",
        "--deep"
    ], capture_output=True, text=True, cwd=r"d:\yingshi\PanoSOT-WHU")
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    result2 = subprocess.run([
        r"d:\yingshi\PanoSOT-WHU\.venv\Scripts\python.exe",
        "examples/visualize_tracking.py",
        "--frames", "data/real_video_frames",
        "--pred", "data/pred_real.txt",
        "--output", "data/real_video_tracking.mp4"
    ], capture_output=True, text=True, cwd=r"d:\yingshi\PanoSOT-WHU")
    
    print(result2.stdout)
    if result2.stderr:
        print("Errors:", result2.stderr)
    
    print("\n✅ Tracking complete! Video saved to: data/real_video_tracking.mp4")
    print("\nIf the target is not correct, open data/video_targets_marked.jpg")
    print("and pick a different target by modifying data/init_real.txt")
    print("Format: x,y,width,height")


if __name__ == "__main__":
    main()