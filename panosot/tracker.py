from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List

import numpy as np

from .geometry import (
    SphereState,
    clamp_lat,
    erp_bbox_to_state,
    lon_distance,
    state_size_to_fov,
    state_to_erp_bbox,
    tangent_patch,
    wrap_lon,
)


@dataclass
class TrackerConfig:
    template_size: int = 48
    search_enlarge: float = 3.0
    local_grid_radius: int = 2
    local_step_factor: float = 0.35
    scale_factors: tuple[float, ...] = (0.93, 1.0, 1.08)
    update_rate: float = 0.05
    high_confidence: float = 0.58
    low_confidence: float = 0.42
    max_lost_frames: int = 4
    relocalize_stride_deg: float = 24.0
    relocalize_lat_stride_deg: float = 18.0
    relocalize_topk: int = 3
    motion_momentum: float = 0.7


def to_gray(image: np.ndarray) -> np.ndarray:
    return (
        0.2989 * image[..., 0]
        + 0.5870 * image[..., 1]
        + 0.1140 * image[..., 2]
    ).astype(np.float32)


def gradient_features(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    mag = np.sqrt(gx * gx + gy * gy)
    return np.stack([gray, gx, gy, mag], axis=0)


def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    flat = feature_map.reshape(feature_map.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-6)
    return flat


def patch_descriptor(patch: np.ndarray) -> np.ndarray:
    gray = to_gray(patch)
    feat = gradient_features(gray)
    return normalize_feature_map(feat)


class PanoSOTTracker:
    """A light-weight 360 tracking baseline with spherical search and re-detection."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.initialized = False
        self.state: SphereState | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.template: np.ndarray | None = None
        self.template_descriptor: np.ndarray | None = None
        self.lost_frames = 0
        self.frame_shape: tuple[int, int] | None = None

    def initialize(self, frame: np.ndarray, init_bbox_xywh: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        self.state = erp_bbox_to_state(init_bbox_xywh, w, h)
        self.template = self._extract_template(frame, self.state)
        self.template_descriptor = patch_descriptor(self.template)
        self.initialized = True
        self.velocity[:] = 0.0
        self.lost_frames = 0
        return state_to_erp_bbox(self.state, w, h)

    def track(self, frame: np.ndarray) -> np.ndarray:
        if not self.initialized or self.state is None or self.frame_shape is None:
            raise RuntimeError("Tracker must be initialized before calling track().")

        h, w = self.frame_shape
        predicted = SphereState(
            lon=float(wrap_lon(self.state.lon + self.velocity[0])),
            lat=float(clamp_lat(self.state.lat + self.velocity[1])),
            equatorial_width=self.state.equatorial_width,
            angular_height=self.state.angular_height,
        )

        best_state, best_score = self._local_search(frame, predicted)
        if best_score < self.config.low_confidence or self.lost_frames >= self.config.max_lost_frames:
            relocalized, relocalized_score = self._global_relocalize(frame, predicted)
            if relocalized_score > best_score:
                best_state, best_score = relocalized, relocalized_score

        lon_delta = lon_distance(best_state.lon, self.state.lon)
        lat_delta = best_state.lat - self.state.lat
        self.velocity[0] = (
            self.config.motion_momentum * self.velocity[0]
            + (1.0 - self.config.motion_momentum) * lon_delta
        )
        self.velocity[1] = (
            self.config.motion_momentum * self.velocity[1]
            + (1.0 - self.config.motion_momentum) * lat_delta
        )
        self.state = best_state

        if best_score >= self.config.high_confidence:
            new_template = self._extract_template(frame, self.state)
            self.template = (
                (1.0 - self.config.update_rate) * self.template
                + self.config.update_rate * new_template
            )
            self.template_descriptor = patch_descriptor(self.template)
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        return state_to_erp_bbox(self.state, w, h)

    def track_sequence(
        self,
        frames: Iterable[np.ndarray],
        init_bbox_xywh: np.ndarray,
    ) -> List[np.ndarray]:
        frames = list(frames)
        if not frames:
            return []
        outputs = [self.initialize(frames[0], init_bbox_xywh)]
        for frame in frames[1:]:
            outputs.append(self.track(frame))
        return outputs

    def _extract_template(self, frame: np.ndarray, state: SphereState) -> np.ndarray:
        size = self.config.template_size
        fov_x, fov_y = state_size_to_fov(state, enlarge=1.25)
        return tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)

    def _score_patch(self, patch: np.ndarray) -> float:
        descriptor = patch_descriptor(patch)
        score = float(np.mean(np.sum(self.template_descriptor * descriptor, axis=1)))
        return score

    def _local_search(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        step_lon = max(
            predicted.equatorial_width * self.config.local_step_factor / max(math.cos(predicted.lat), 1e-3),
            math.radians(2.0),
        )
        step_lat = max(predicted.angular_height * self.config.local_step_factor, math.radians(2.0))
        best_state = predicted
        best_score = -1.0

        for scale in self.config.scale_factors:
            candidate_height = predicted.angular_height * scale
            candidate_width = predicted.equatorial_width * scale
            candidate = SphereState(
                lon=predicted.lon,
                lat=predicted.lat,
                equatorial_width=candidate_width,
                angular_height=candidate_height,
            )
            fov_x, fov_y = state_size_to_fov(candidate, enlarge=self.config.search_enlarge)

            for dx in range(-self.config.local_grid_radius, self.config.local_grid_radius + 1):
                for dy in range(-self.config.local_grid_radius, self.config.local_grid_radius + 1):
                    lon = wrap_lon(predicted.lon + dx * step_lon)
                    lat = clamp_lat(predicted.lat + dy * step_lat)
                    patch = tangent_patch(
                        frame,
                        lon,
                        lat,
                        fov_x,
                        fov_y,
                        self.config.template_size,
                        self.config.template_size,
                    )
                    score = self._score_patch(patch)
                    score -= 0.015 * (abs(dx) + abs(dy))
                    if score > best_score:
                        best_score = score
                        best_state = SphereState(
                            lon=float(lon),
                            lat=float(lat),
                            equatorial_width=float(candidate_width),
                            angular_height=float(candidate_height),
                        )
        return best_state, best_score

    def _global_relocalize(
        self,
        frame: np.ndarray,
        predicted: SphereState,
    ) -> tuple[SphereState, float]:
        lon_values = np.deg2rad(
            np.arange(-180.0, 180.0, self.config.relocalize_stride_deg, dtype=np.float32)
        )
        lat_values = np.deg2rad(
            np.arange(-72.0, 72.1, self.config.relocalize_lat_stride_deg, dtype=np.float32)
        )
        candidates: list[tuple[float, SphereState]] = []

        fov_x, fov_y = state_size_to_fov(predicted, enlarge=self.config.search_enlarge)
        for lon in lon_values:
            for lat in lat_values:
                lat = clamp_lat(float(lat))
                patch = tangent_patch(
                    frame,
                    float(lon),
                    float(lat),
                    fov_x,
                    fov_y,
                    self.config.template_size,
                    self.config.template_size,
                )
                score = self._score_patch(patch)
                score -= 0.01 * abs(lon_distance(float(lon), predicted.lon))
                score -= 0.02 * abs(float(lat) - predicted.lat)
                candidate = SphereState(
                    lon=float(lon),
                    lat=float(lat),
                    equatorial_width=predicted.equatorial_width,
                    angular_height=predicted.angular_height,
                )
                candidates.append((score, candidate))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_state = predicted
        best_score = -1.0
        for coarse_score, coarse_state in candidates[: self.config.relocalize_topk]:
            refined_state, refined_score = self._local_search(frame, coarse_state)
            if refined_score > best_score:
                best_state = refined_state
                best_score = max(refined_score, coarse_score)
        return best_state, best_score
