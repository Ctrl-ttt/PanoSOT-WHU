from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, List

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
    # --- 手工特征参数（baseline，始终可用）---
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

    # --- 深度特征参数（Phase 1 新增）---
    use_deep_features: bool = False
    backbone_name: str = "mobilenet_v3_small"
    deep_template_size: int = 112
    coarse_search_size: int = 224
    refine_search_size: int = 160
    device: str = "cpu"
    use_amp: bool = False
    # 搜索区域相对于目标的放大倍数（deep模式下用更大的search patch一次覆盖）
    deep_search_enlarge: float = 2.5


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
    """A light-weight 360 tracking baseline with spherical search and re-detection.

    支持两种模式：
    - 手工特征模式（默认）：使用灰度+梯度手工描述子
    - 深度特征模式（use_deep_features=True）：使用轻量 backbone + cross-correlation
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        deep_extractor: Any = None,
        similarity_head: Any = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.initialized = False
        self.state: SphereState | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.template: np.ndarray | None = None
        self.template_descriptor: np.ndarray | None = None
        self.lost_frames = 0
        self.frame_shape: tuple[int, int] | None = None

        # --- 深度特征相关 ---
        self.deep_extractor = deep_extractor
        self.similarity_head = similarity_head
        self.template_feat: Any = None  # torch.Tensor, 模板的深度特征
        self._deep_mode = False
        if self.config.use_deep_features and deep_extractor is not None and similarity_head is not None:
            self._deep_mode = True

    def initialize(self, frame: np.ndarray, init_bbox_xywh: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        self.state = erp_bbox_to_state(init_bbox_xywh, w, h)
        self.template = self._extract_template(frame, self.state)
        self.template_descriptor = patch_descriptor(self.template)
        if self._deep_mode:
            self.template_feat = self._extract_template_feat(frame, self.state)
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
            if self._deep_mode and self.template_feat is not None:
                new_feat = self._extract_template_feat(frame, self.state)
                self.template_feat = (
                    (1.0 - self.config.update_rate) * self.template_feat
                    + self.config.update_rate * new_feat
                )
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

    # ---------- 深度特征方法 ----------

    def _extract_template_feat(self, frame: np.ndarray, state: SphereState) -> Any:
        """提取模板的深度特征（归一化球面 patch → backbone 前向）。"""
        size = self.config.deep_template_size
        fov_x, fov_y = state_size_to_fov(state, enlarge=1.25)
        patch = tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)
        return self.deep_extractor.extract_template_feature(patch)

    def _extract_search_feat(
        self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float, refine: bool = False
    ) -> Any:
        """提取搜索区域的深度特征。"""
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        patch = tangent_patch(frame, lon, lat, fov_x, fov_y, out_size, out_size)
        return self.deep_extractor.extract_search_feature(patch, refine=refine)

    def _deep_score_and_offset(
        self, template_feat: Any, search_feat: Any
    ) -> tuple[float, tuple[float, float]]:
        """对 template 和 search 特征做 cross-correlation，返回峰值分数和偏移。

        Returns:
            score: 响应图峰值（0~1 之间归一化）
            (offset_y, offset_x): 峰值相对于响应图中心的偏移（归一化到 [-1, 1]）
        """
        response = self.similarity_head(template_feat, search_feat)
        response_np = response.squeeze().detach().cpu().numpy()
        max_idx = response_np.argmax()
        h, w = response_np.shape
        peak_y, peak_x = max_idx // w, max_idx % w
        score = float(response_np[peak_y, peak_x])

        # 归一化偏移：中心为 (0,0)，范围 [-1, 1]
        offset_y = (peak_y - (h - 1) / 2.0) / ((h - 1) / 2.0) if h > 1 else 0.0
        offset_x = (peak_x - (w - 1) / 2.0) / ((w - 1) / 2.0) if w > 1 else 0.0
        return score, (offset_y, offset_x)

    # ---------- 打分与搜索 ----------

    def _score_patch(self, patch: np.ndarray) -> float:
        descriptor = patch_descriptor(patch)
        score = float(np.mean(np.sum(self.template_descriptor * descriptor, axis=1)))
        return score

    def _score_patch_deep(self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float) -> float:
        """深度特征打分：提取搜索patch特征，与模板做 cross-correlation，返回峰值分数。"""
        search_feat = self._extract_search_feat(frame, lon, lat, fov_x, fov_y)
        score, _ = self._deep_score_and_offset(self.template_feat, search_feat)
        return score

    def _local_search(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        if self._deep_mode:
            return self._local_search_deep(frame, predicted)
        return self._local_search_handcrafted(frame, predicted)

    def _local_search_handcrafted(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        """原始手工特征局部搜索（保持不变）。"""
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
                        frame, lon, lat, fov_x, fov_y,
                        self.config.template_size, self.config.template_size,
                    )
                    score = self._score_patch(patch)
                    score -= 0.015 * (abs(dx) + abs(dy))
                    if score > best_score:
                        best_score = score
                        best_state = SphereState(
                            lon=float(lon), lat=float(lat),
                            equatorial_width=float(candidate_width),
                            angular_height=float(candidate_height),
                        )
        return best_state, best_score

    def _local_search_deep(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        """深度特征 coarse-to-fine 搜索。

        Coarse: 在预测位置提取大范围搜索patch，cross-correlation 一次得到候选位置。
        Refine: 在候选位置附近，小范围高分辨率精修。
        """
        best_state = predicted
        best_score = -1.0

        for scale in self.config.scale_factors:
            candidate_height = predicted.angular_height * scale
            candidate_width = predicted.equatorial_width * scale
            fov_x, fov_y = state_size_to_fov(
                SphereState(predicted.lon, predicted.lat, candidate_width, candidate_height),
                enlarge=self.config.deep_search_enlarge,
            )

            # --- Coarse stage ---
            search_feat = self._extract_search_feat(
                frame, predicted.lon, predicted.lat, fov_x, fov_y, refine=False,
            )
            score, (off_y, off_x) = self._deep_score_and_offset(self.template_feat, search_feat)

            # 将 offset 映射回球面坐标
            coarse_lon = wrap_lon(predicted.lon + off_x * (0.5 * fov_x))
            coarse_lat = clamp_lat(predicted.lat + off_y * (0.5 * fov_y))

            # --- Refine stage: 在 coarse 位置周围做高分辨率精修 ---
            refine_fov_x = fov_x * 0.5
            refine_fov_y = fov_y * 0.5
            refine_feat = self._extract_search_feat(
                frame, coarse_lon, coarse_lat, refine_fov_x, refine_fov_y, refine=True,
            )
            refine_score, (roff_y, roff_x) = self._deep_score_and_offset(self.template_feat, refine_feat)

            refined_lon = wrap_lon(coarse_lon + roff_x * (0.5 * refine_fov_x))
            refined_lat = clamp_lat(coarse_lat + roff_y * (0.5 * refine_fov_y))

            # 综合 coarse + refine 分数
            combined_score = 0.3 * score + 0.7 * refine_score
            # 轻微惩罚尺度变化
            combined_score -= 0.02 * abs(scale - 1.0)

            if combined_score > best_score:
                best_score = combined_score
                best_state = SphereState(
                    lon=float(refined_lon), lat=float(refined_lat),
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
                if self._deep_mode:
                    score = self._score_patch_deep(frame, float(lon), float(lat), fov_x, fov_y)
                else:
                    patch = tangent_patch(
                        frame, float(lon), float(lat), fov_x, fov_y,
                        self.config.template_size, self.config.template_size,
                    )
                    score = self._score_patch(patch)
                score -= 0.01 * abs(lon_distance(float(lon), predicted.lon))
                score -= 0.02 * abs(float(lat) - predicted.lat)
                candidate = SphereState(
                    lon=float(lon), lat=float(lat),
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
