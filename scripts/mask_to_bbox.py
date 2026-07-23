"""从 mask 生成 bbox 标注文件 (groundtruth.txt)"""
from PIL import Image
import numpy as np
import os
import sys

def main():
    mask_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/360VOTS/mask'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'data/360VOTS/groundtruth.txt'

    masks = sorted(os.listdir(mask_dir))
    with open(out_file, 'w') as f:
        for m in masks:
            path = os.path.join(mask_dir, m)
            mask = np.array(Image.open(path).convert('L'))
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                f.write('0,0,0,0\n')
            else:
                x = int(xs.min())
                y = int(ys.min())
                w = int(xs.max() - xs.min() + 1)
                h = int(ys.max() - ys.min() + 1)
                f.write(f'{x},{y},{w},{h}\n')
    print(f'Done! {len(masks)} bboxes written to {out_file}')

if __name__ == '__main__':
    main()
