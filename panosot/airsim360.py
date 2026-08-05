"""Utilities for sampling instance-supervised pairs from AirSim360 ERP data."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import random
import zipfile
from typing import Iterator, Sequence

import numpy as np


def decode_argb_instance_ids(image: np.ndarray) -> np.ndarray:
    """Decode AirSim's ARGB instance colour into one uint32 id per pixel.

    Pillow returns PNG data in RGBA channel order, while AirSim stores the
    instance colour as an AARRGGBB integer.
    """
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected AirSim instance image with shape [H, W, 4].")
    rgba = image.astype(np.uint32, copy=False)
    return (
        (rgba[..., 3] << 24)
        | (rgba[..., 0] << 16)
        | (rgba[..., 1] << 8)
        | rgba[..., 2]
    )


@dataclass(frozen=True)
class InstanceBox:
    instance_id: int
    bbox_xywh: tuple[int, int, int, int]
    pixel_count: int


def _wrapped_x_interval(xs: np.ndarray, image_width: int) -> tuple[int, int]:
    """Return the shortest ERP interval containing horizontal pixel positions.

    The largest empty angular gap is excluded.  The resulting interval may
    cross the ERP seam: x + width can exceed image_width, which is supported
    by the tracker's spherical-box conversion.
    """
    unique_xs = np.unique(xs)
    if len(unique_xs) == 1:
        return int(unique_xs[0]), 1
    gaps = np.diff(unique_xs)
    wrap_gap = int(unique_xs[0]) + image_width - int(unique_xs[-1])
    all_gaps = np.append(gaps, wrap_gap)
    gap_index = int(np.argmax(all_gaps))
    start = int(unique_xs[(gap_index + 1) % len(unique_xs)])
    width = image_width - int(all_gaps[gap_index]) + 1
    return start, width


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
        x0, box_width = _wrapped_x_interval(xs, width)
        y0, y1 = int(ys.min()), int(ys.max())
        boxes.append(
            InstanceBox(
                instance_id=int(value),
                bbox_xywh=(x0, y0, box_width, y1 - y0 + 1),
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


class AirSim360TrainingArchive:
    """Read aligned RGB/instance archive pairs without unpacking multi-GB data."""

    def __init__(self, raw_archive: str | Path, instance_archive: str | Path) -> None:
        self.raw_archive = Path(raw_archive)
        self.instances = AirSim360InstanceArchive(instance_archive)
        with zipfile.ZipFile(self.raw_archive) as archive:
            self._raw_names = {
                Path(name).name: name
                for name in archive.namelist()
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            }

    def paired_names(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for instance_name in self.instances.names():
            raw_name = self._raw_names.get(Path(instance_name).name)
            if raw_name is not None:
                pairs.append((raw_name, instance_name))
        return pairs

    def read_rgb(self, member_name: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to read AirSim360 RGB images.") from exc
        with zipfile.ZipFile(self.raw_archive) as archive:
            with Image.open(BytesIO(archive.read(member_name))) as image:
                return np.asarray(image.convert("RGB"))


def crop_erp_wrapped(
    image: np.ndarray,
    bbox_xywh: Sequence[int | float],
    *,
    context: float = 2.0,
) -> np.ndarray:
    """Crop an ERP target with horizontal wrapping and padded vertical context."""
    x, y, width, height = (float(value) for value in bbox_xywh)
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive.")
    image_height, image_width = image.shape[:2]
    crop_width = max(1, int(round(width * context)))
    crop_height = max(1, int(round(height * context)))
    center_x = x + 0.5 * width
    center_y = y + 0.5 * height
    start_x = int(np.floor(center_x - 0.5 * crop_width))
    start_y = int(np.floor(center_y - 0.5 * crop_height))
    xs = np.mod(np.arange(start_x, start_x + crop_width), image_width)
    ys = np.clip(np.arange(start_y, start_y + crop_height), 0, image_height - 1)
    return image[np.ix_(ys, xs)]


class AirSim360PairDataset:
    """Instance-supervised template/search pairs for tracker pretraining.

    AirSim360 panoramic frames are static scene renders rather than labelled
    target tracks.  We therefore preserve each object identity inside a frame
    and create two independently augmented views of its wrapped ERP crop.
    """

    def __init__(
        self,
        archive: AirSim360TrainingArchive,
        *,
        min_pixels: int = 128,
        max_fraction: float = 0.25,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.archive = archive
        self._rng = random.Random(seed)
        samples: list[tuple[str, InstanceBox]] = []
        paired = {
            instance_name: raw_name
            for raw_name, instance_name in archive.paired_names()
        }
        for instance_name, boxes in archive.instances.iter_boxes(
            min_pixels=min_pixels, max_fraction=max_fraction
        ):
            raw_name = paired.get(instance_name)
            if raw_name is None:
                continue
            samples.extend((raw_name, box) for box in boxes)
            if max_samples is not None and len(samples) >= max_samples:
                break
        self.samples = samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        raw_name, box = self.samples[index]
        image = self.archive.read_rgb(raw_name)
        template = crop_erp_wrapped(image, box.bbox_xywh, context=1.7)
        search = crop_erp_wrapped(image, box.bbox_xywh, context=2.6)
        return template, search
