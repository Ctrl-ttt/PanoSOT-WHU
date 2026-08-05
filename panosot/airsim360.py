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
        self._archive: zipfile.ZipFile | None = None

    def _open(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        return self._archive

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def names(self) -> list[str]:
        return sorted(
            name for name in self._open().namelist() if name.lower().endswith(".png")
        )

    def read_rgba(self, member_name: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to read AirSim360 PNG labels.") from exc
        with Image.open(BytesIO(self._open().read(member_name))) as image:
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
        self._archive = zipfile.ZipFile(self.raw_archive)
        self._raw_names = {
            Path(name).name: name
            for name in self._archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        }

    def close(self) -> None:
        self._archive.close()
        self.instances.close()

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
        with Image.open(BytesIO(self._archive.read(member_name))) as image:
            return np.asarray(image.convert("RGB"))


def crop_erp_wrapped(
    image: np.ndarray,
    bbox_xywh: Sequence[int | float],
    *,
    context: float = 2.0,
    center_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Crop an ERP target with horizontal wrapping and padded vertical context."""
    x, y, width, height = (float(value) for value in bbox_xywh)
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive.")
    image_height, image_width = image.shape[:2]
    crop_width = max(1, int(round(width * context)))
    crop_height = max(1, int(round(height * context)))
    offset_x, offset_y = center_offset_xy
    center_x = x + 0.5 * width + float(offset_x)
    center_y = y + 0.5 * height + float(offset_y)
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

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        raw_name, box = self.samples[index]
        image = self.archive.read_rgb(raw_name)
        # With 112/224 network inputs, these contexts preserve the same
        # target pixel scale in template and search; the target therefore
        # belongs at the centre of the XCorr response map.
        template = crop_erp_wrapped(image, box.bbox_xywh, context=2.0)
        # Move the search crop while retaining the same target.  This supplies
        # the non-centre localization labels absent from static panoramas.
        box_width, box_height = box.bbox_xywh[2:]
        offset_x = self._rng.uniform(-0.70 * box_width, 0.70 * box_width)
        offset_y = self._rng.uniform(-0.70 * box_height, 0.70 * box_height)
        search = crop_erp_wrapped(
            image,
            box.bbox_xywh,
            context=4.0,
            center_offset_xy=(offset_x, offset_y),
        )
        target_fraction = (
            0.5 - offset_x / (4.0 * box_width),
            0.5 - offset_y / (4.0 * box_height),
        )
        return template, search, target_fraction


class AirSim360TemporalPairDataset:
    """Consecutive-frame pairs with persistent AirSim instance identities."""

    def __init__(
        self,
        archive: AirSim360TrainingArchive,
        *,
        min_pixels: int = 128,
        max_fraction: float = 0.25,
        max_samples: int | None = None,
    ) -> None:
        self.archive = archive
        numbered = {
            int(Path(name).stem.rsplit("_", 1)[1]): name
            for name in archive.instances.names()
        }
        paired = {
            instance_name: raw_name
            for raw_name, instance_name in archive.paired_names()
        }
        first_label = archive.instances.read_rgba(numbered[min(numbered)])
        self.image_width = int(first_label.shape[1])
        samples: list[tuple[str, InstanceBox, str, InstanceBox]] = []
        previous_boxes: dict[int, InstanceBox] | None = None
        previous_raw: str | None = None
        for number in sorted(numbered):
            instance_name = numbered[number]
            raw_name = paired.get(instance_name)
            if raw_name is None:
                continue
            current_boxes = {
                box.instance_id: box
                for box in extract_instance_boxes(
                    decode_argb_instance_ids(archive.instances.read_rgba(instance_name)),
                    min_pixels=min_pixels,
                    max_fraction=max_fraction,
                )
            }
            if previous_boxes is not None and previous_raw is not None:
                shared_ids = sorted(previous_boxes.keys() & current_boxes.keys())
                for instance_id in shared_ids:
                    previous_box = previous_boxes[instance_id]
                    current_box = current_boxes[instance_id]
                    target_fraction = temporal_target_fraction(
                        previous_box.bbox_xywh,
                        current_box.bbox_xywh,
                        self.image_width,
                    )
                    # Retain realistic motion, but avoid pairs where the
                    # target leaves the 4x local-search crop.
                    if 0.05 <= target_fraction[0] <= 0.95 and 0.05 <= target_fraction[1] <= 0.95:
                        samples.append(
                            (previous_raw, previous_box, raw_name, current_box)
                        )
                if max_samples is not None and len(samples) >= max_samples:
                    break
            previous_boxes = current_boxes
            previous_raw = raw_name
        self.samples = samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        template_name, template_box, search_name, search_box = self.samples[index]
        template_image = self.archive.read_rgb(template_name)
        search_image = self.archive.read_rgb(search_name)
        template = crop_erp_wrapped(
            template_image, template_box.bbox_xywh, context=2.0
        )
        # Runtime local search is centred on the previous state, not on the
        # unknown next-frame target. Reproduce that condition for supervision.
        search = crop_erp_wrapped(search_image, template_box.bbox_xywh, context=4.0)
        target_fraction = temporal_target_fraction(
            template_box.bbox_xywh, search_box.bbox_xywh, self.image_width
        )
        return template, search, target_fraction


def temporal_target_fraction(
    previous_bbox_xywh: Sequence[int | float],
    current_bbox_xywh: Sequence[int | float],
    image_width: int,
) -> tuple[float, float]:
    """Map next-frame target centre into a 4x previous-state search crop."""
    prev_x, prev_y, prev_w, prev_h = (float(value) for value in previous_bbox_xywh)
    curr_x, curr_y, curr_w, curr_h = (float(value) for value in current_bbox_xywh)
    prev_cx, prev_cy = prev_x + 0.5 * prev_w, prev_y + 0.5 * prev_h
    curr_cx, curr_cy = curr_x + 0.5 * curr_w, curr_y + 0.5 * curr_h
    dx = (curr_cx - prev_cx + 0.5 * image_width) % image_width - 0.5 * image_width
    dy = curr_cy - prev_cy
    return 0.5 + dx / (4.0 * prev_w), 0.5 + dy / (4.0 * prev_h)
