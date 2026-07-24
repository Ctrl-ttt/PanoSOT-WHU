from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _resample_nearest() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.NEAREST
    return Image.NEAREST


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.dtype != np.uint8:
        max_value = float(array.max()) if array.size else 0.0
        scale = 255.0 if max_value <= 1.5 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    return array


def _response_to_rgb(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2:
        raise ValueError("Expected response map to have shape [H, W].")

    resp_min = float(response.min()) if response.size else 0.0
    resp_max = float(response.max()) if response.size else 1.0
    denom = max(resp_max - resp_min, 1e-6)
    normalized = (response - resp_min) / denom

    red = normalized
    green = np.sqrt(normalized)
    blue = 1.0 - normalized
    return np.clip(np.stack([red, green, blue], axis=-1) * 255.0, 0.0, 255.0).astype(np.uint8)


def _save_image(path: Path, image: np.ndarray, upscale: int = 1) -> None:
    rgb = _to_uint8_rgb(image)
    pil_image = Image.fromarray(rgb, mode="RGB")
    if upscale > 1:
        pil_image = pil_image.resize(
            (pil_image.width * upscale, pil_image.height * upscale),
            resample=_resample_nearest(),
        )
    pil_image.save(path)


def _save_response_map(path: Path, response: np.ndarray) -> None:
    rgb = _response_to_rgb(response)
    pil_image = Image.fromarray(rgb, mode="RGB")
    scale = max(1, int(np.ceil(192 / max(pil_image.width, pil_image.height, 1))))
    if scale > 1:
        pil_image = pil_image.resize(
            (pil_image.width * scale, pil_image.height * scale),
            resample=_resample_nearest(),
        )

    draw = ImageDraw.Draw(pil_image)
    peak_y, peak_x = np.unravel_index(int(np.argmax(response)), response.shape)
    peak_x *= scale
    peak_y *= scale
    draw.line((peak_x - 6, peak_y, peak_x + 6, peak_y), fill=(255, 255, 255), width=1)
    draw.line((peak_x, peak_y - 6, peak_x, peak_y + 6), fill=(255, 255, 255), width=1)
    pil_image.save(path)


def _box_segments(bbox_xywh: np.ndarray, image_width: int) -> list[tuple[float, float, float, float]]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    x = x % max(image_width, 1)
    x2 = x + max(w, 0.0)
    if x2 <= image_width:
        return [(x, y, x2, y + max(h, 0.0))]
    return [
        (x, y, float(image_width - 1), y + max(h, 0.0)),
        (0.0, y, x2 - image_width, y + max(h, 0.0)),
    ]


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox_xywh: np.ndarray,
    image_size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    width, height = image_size
    for x1, y1, x2, y2 in _box_segments(bbox_xywh, width):
        draw.rectangle(
            (
                max(0.0, x1),
                max(0.0, y1),
                min(float(width - 1), x2),
                min(float(height - 1), y2),
            ),
            outline=color,
            width=3,
        )
    label_x = int(float(bbox_xywh[0]) % max(width, 1))
    label_y = max(0, int(float(bbox_xywh[1])) - 14)
    draw.text((label_x, label_y), label, fill=color)


class TrackerDebugRecorder:
    def __init__(
        self,
        debug_dir: str,
        start_frame: int = 0,
        max_frames: int = 20,
        frame_stride: int = 1,
        save_response_maps: bool = True,
    ) -> None:
        self.root_dir = Path(debug_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.start_frame = max(0, int(start_frame))
        self.max_frames = max(0, int(max_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.save_response_maps = save_response_maps
        self._captured_frames = 0

    def wants_frame(self, frame_index: int) -> bool:
        if frame_index < self.start_frame:
            return False
        if self._captured_frames >= self.max_frames:
            return False
        return (frame_index - self.start_frame) % self.frame_stride == 0

    def record_initialize(
        self,
        frame: np.ndarray,
        init_bbox_xywh: np.ndarray,
        template_patch: np.ndarray | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        init_dir = self.root_dir / "frame_0000_init"
        init_dir.mkdir(parents=True, exist_ok=True)

        overlay = Image.fromarray(_to_uint8_rgb(frame), mode="RGB")
        draw = ImageDraw.Draw(overlay)
        _draw_bbox(draw, init_bbox_xywh, overlay.size, (0, 255, 0), "init")
        overlay.save(init_dir / "overlay.png")

        if template_patch is not None:
            _save_image(init_dir / "template_patch.png", template_patch, upscale=2)

        lines = ["frame=0", "type=initialize"]
        if metadata:
            for key in sorted(metadata):
                lines.append(f"{key}={metadata[key]}")
        (init_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def record_step(
        self,
        frame_index: int,
        frame: np.ndarray,
        predicted_bbox_xywh: np.ndarray,
        result_bbox_xywh: np.ndarray,
        score: float,
        metadata: dict[str, Any] | None = None,
        artifacts: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not self.wants_frame(frame_index):
            return

        self._captured_frames += 1
        frame_dir = self.root_dir / f"frame_{frame_index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        overlay = Image.fromarray(_to_uint8_rgb(frame), mode="RGB")
        draw = ImageDraw.Draw(overlay)
        _draw_bbox(draw, predicted_bbox_xywh, overlay.size, (255, 215, 0), "predicted")
        _draw_bbox(draw, result_bbox_xywh, overlay.size, (255, 64, 64), "result")
        overlay.save(frame_dir / "overlay.png")

        if artifacts:
            for name in ("template_patch", "match_patch", "coarse_patch", "refine_patch"):
                image = artifacts.get(name)
                if image is not None:
                    _save_image(frame_dir / f"{name}.png", image, upscale=2)

            if self.save_response_maps:
                for name in ("coarse_response", "refine_response"):
                    response = artifacts.get(name)
                    if response is not None:
                        _save_response_map(frame_dir / f"{name}.png", response)

        lines = [
            f"frame={frame_index}",
            "type=track",
            f"score={score:.6f}",
            f"predicted_bbox={np.array2string(predicted_bbox_xywh, precision=3, separator=', ')}",
            f"result_bbox={np.array2string(result_bbox_xywh, precision=3, separator=', ')}",
        ]
        if metadata:
            for key in sorted(metadata):
                lines.append(f"{key}={metadata[key]}")
        (frame_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
