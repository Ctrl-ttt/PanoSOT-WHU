"""Utilities for sampling instance-supervised pairs from AirSim360 ERP data."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import zipfile
from typing import Iterator

import numpy as np


def decode_argb_instance_ids(image: np.ndarray) -> np.ndarray:
    """Decode AirSim's RGBA/ARGB instance colour into one uint32 id per pixel."""
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected AirSim instance image with shape [H, W, 4].")
    rgba = image.astype(np.uint32, copy=False)
    return (
        rgba[..., 0]
        | (rgba[..., 1] << 8)
        | (rgba[..., 2] << 16)
        | (rgba[..., 3] << 24)
    )


@dataclass(frozen=True)
class InstanceBox:
    instance_id: int
    bbox_xywh: tuple[int, int, int, int]
    pixel_count: int


def extract_instance_boxes(
    instance_ids: np.ndarray,
    *,
    min_pixels: int = 128,
    max_fraction: float = 0.25,
) -> list[InstanceBox]:
    """Return usable object boxes while rejecting background and huge regions."""
    if instance_ids.ndim != 2:
        raise ValueError("Expected decoded instance ids with shape [H, W].")
    height, width = instance_ids.shape
    values, counts = np.unique(instance_ids, return_counts=True)
    max_pixels = int(round(height * width * max_fraction))
    boxes: list[InstanceBox] = []
    for value, count in zip(values.tolist(), counts.tolist()):
        # ID zero is reserved for empty/background in AirSim's segmentation.
        if value == 0 or count < min_pixels or count > max_pixels:
            continue
        ys, xs = np.nonzero(instance_ids == value)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        boxes.append(
            InstanceBox(
                instance_id=int(value),
                bbox_xywh=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                pixel_count=int(count),
            )
        )
    return boxes


class AirSim360InstanceArchive:
    """Lazy reader for an official AirSim360 instance zip archive."""

    def __init__(self, archive_path: str | Path) -> None:
        self.archive_path = Path(archive_path)

    def names(self) -> list[str]:
        with zipfile.ZipFile(self.archive_path) as archive:
            return sorted(
                name for name in archive.namelist() if name.lower().endswith(".png")
            )

    def read_rgba(self, member_name: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to read AirSim360 PNG labels.") from exc
        with zipfile.ZipFile(self.archive_path) as archive:
            with Image.open(BytesIO(archive.read(member_name))) as image:
                return np.asarray(image.convert("RGBA"))

    def iter_boxes(
        self, *, min_pixels: int = 128, max_fraction: float = 0.25
    ) -> Iterator[tuple[str, list[InstanceBox]]]:
        for name in self.names():
            ids = decode_argb_instance_ids(self.read_rgba(name))
            yield name, extract_instance_boxes(
                ids, min_pixels=min_pixels, max_fraction=max_fraction
            )
