from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class InitialBoxSelector:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        self.start_x, self.start_y = -1, -1
        self.drawing = False
        self.boxes = []
        
        cv2.namedWindow("Select Initial Box", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Initial Box", 1280, 720)
        cv2.setMouseCallback("Select Initial Box", self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp = self.image.copy()
                cv2.rectangle(temp, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Initial Box", temp)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                end_x, end_y = x, y
                x1 = min(self.start_x, end_x)
                y1 = min(self.start_y, end_y)
                w = abs(end_x - self.start_x)
                h = abs(end_y - self.start_y)
                if w > 20 and h > 40:
                    self.boxes.append((x1, y1, w, h))
                    print(f"\nSelected box: {x1},{y1},{w},{h}")
                    self.draw_boxes()
            self.drawing = False
    
    def draw_boxes(self):
        temp = self.image.copy()
        for i, (x, y, w, h) in enumerate(self.boxes):
            cv2.rectangle(temp, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(temp, f"Box {i}: {x},{y},{w},{h}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Select Initial Box", temp)
    
    def run(self):
        print("=== Interactive Initial Box Selector ===")
        print("Instructions:")
        print("1. Click and drag to draw a rectangle around the target")
        print("2. Press 's' to save the last selected box to data/init_vtest.txt")
        print("3. Press 'q' to quit")
        print(f"\nImage size: {self.width}x{self.height}")
        
        cv2.imshow("Select Initial Box", self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and self.boxes:
                x, y, w, h = self.boxes[-1]
                with open("data/init_vtest.txt", 'w') as f:
                    f.write(f"{x},{y},{w},{h}")
                print(f"\nSaved to data/init_vtest.txt: {x},{y},{w},{h}")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if self.boxes:
            print(f"\nSelected boxes:")
            for i, (x, y, w, h) in enumerate(self.boxes):
                print(f"  {i}: {x},{y},{w},{h}")
            return self.boxes[-1]
        return None


if __name__ == "__main__":
    selector = InitialBoxSelector("data/vtest_frames/00000001.jpg")
    selector.run()