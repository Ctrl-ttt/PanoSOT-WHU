from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class TargetSelector:
    def __init__(self, image_path: Path):
        self.image = cv2.imread(str(image_path))
        self.height, self.width = self.image.shape[:2]
        self.clone = self.image.copy()
        self.start_x = None
        self.start_y = None
        self.drawing = False
    
    def draw_rect(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x = x
            self.start_y = y
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.image = self.clone.copy()
                cv2.rectangle(self.image, (self.start_x, self.start_y), (x, y), (0, 0, 255), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)
            w, h = x2 - x1, y2 - y1
            self.selected_box = (x1, y1, w, h)
            print(f"Selected: x={x1}, y={y1}, w={w}, h={h}")
    
    def run(self) -> tuple[int, int, int, int] | None:
        cv2.namedWindow('Select Target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Target', min(self.width, 1200), min(self.height, 800))
        cv2.setMouseCallback('Select Target', self.draw_rect)
        
        print("Instructions:")
        print("1. Click and drag to draw a rectangle around the target")
        print("2. Press 's' to save and exit")
        print("3. Press 'r' to reset")
        print("4. Press 'q' to quit without saving")
        
        while True:
            cv2.imshow('Select Target', self.image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and hasattr(self, 'selected_box'):
                cv2.destroyAllWindows()
                return self.selected_box
            elif key == ord('r'):
                self.image = self.clone.copy()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None


def main():
    video_path = Path(r"d:\yingshi\PanoSOT-WHU\data\external_videos\tracking_pexels_360_36157408.mp4")
    first_frame_path = Path("data/first_frame_real.jpg")
    
    if not first_frame_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        success, first_frame = cap.read()
        cap.release()
        if success:
            cv2.imwrite(str(first_frame_path), first_frame)
    
    if not first_frame_path.exists():
        print("Failed to get first frame")
        return
    
    selector = TargetSelector(first_frame_path)
    box = selector.run()
    
    if box:
        x, y, w, h = box
        init_box = f"{x},{y},{w},{h}"
        (Path("data") / "init_real.txt").write_text(init_box)
        print(f"\nSaved init box to data/init_real.txt: {init_box}")
        
        overlay = cv2.imread(str(first_frame_path))
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 4)
        cv2.imwrite("data/selected_target.jpg", overlay)
        print("Saved selected target to data/selected_target.jpg")
        
        print("\nNow run tracking:")
        print(f"  python examples/run_tracker.py --sequence data/real_video_frames --init-box data/init_real.txt --output data/pred_real.txt --deep")
        print(f"  python examples/visualize_tracking.py --frames data/real_video_frames --pred data/pred_real.txt --output data/real_video_tracking.mp4")


if __name__ == "__main__":
    main()