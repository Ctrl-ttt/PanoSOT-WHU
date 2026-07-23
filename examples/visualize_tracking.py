from __future__ import annotations

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import load_boxes


def load_predictions(pred_path: Path) -> list[np.ndarray]:
    return load_boxes(pred_path)


def load_ground_truth(gt_path: Path) -> list[np.ndarray]:
    return load_boxes(gt_path)


def draw_box(frame: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int], label: str = "") -> np.ndarray:
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    if label:
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def create_video(frames_dir: Path, pred_path: Path, gt_path: Path, output_path: Path):
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    predictions = load_predictions(pred_path)
    ground_truth = load_ground_truth(gt_path)
    
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
    
    print(f"Creating visualization video...")
    
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        
        if i < len(ground_truth):
            frame = draw_box(frame, ground_truth[i], (0, 255, 0), "GT")
        
        if i < len(predictions):
            frame = draw_box(frame, predictions[i], (0, 0, 255), "Pred")
        
        out.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frame_paths)} frames")
    
    out.release()
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking results")
    parser.add_argument("--frames", required=True, help="Directory containing frames")
    parser.add_argument("--pred", required=True, help="Prediction file")
    parser.add_argument("--gt", required=True, help="Ground truth file")
    parser.add_argument("--output", default="tracking_visualization.mp4", help="Output video path")
    args = parser.parse_args()
    
    create_video(Path(args.frames), Path(args.pred), Path(args.gt), Path(args.output))


if __name__ == "__main__":
    main()