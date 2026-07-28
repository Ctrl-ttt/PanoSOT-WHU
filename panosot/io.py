from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path: str | Path, max_size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if max_size is not None:
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_sequence(sequence_dir: str | Path, max_size: int | None = 960) -> List[np.ndarray]:
    sequence_dir = Path(sequence_dir)
    frame_paths = sorted(
        path for path in sequence_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return [load_image(path, max_size=max_size) for path in frame_paths]


def load_sequence_lazy(
    sequence_dir: str | Path, max_size: int | None = 960
) -> Iterable[np.ndarray]:
    """惰性加载序列帧，不一次性占满内存。

    返回生成器，每次只加载一帧。适用于长序列场景，
    内存占用从 O(N * H * W * 3) 降至 O(H * W * 3)。
    注意：生成器只能迭代一次。
    """
    sequence_dir = Path(sequence_dir)
    frame_paths = sorted(
        path for path in sequence_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for path in frame_paths:
        yield load_image(path, max_size=max_size)


def load_boxes(path: str | Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        rows.append([float(value) for value in tokens[:4]])
    return np.asarray(rows, dtype=np.float32)


def save_boxes(path: str | Path, boxes: Iterable[np.ndarray]) -> None:
    lines = []
    for box in boxes:
        x, y, w, h = [float(v) for v in box]
        lines.append(f"{x:.3f},{y:.3f},{w:.3f},{h:.3f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
