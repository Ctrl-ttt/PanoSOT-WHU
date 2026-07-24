from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, List

import numpy as np
from PIL import Image

from .debug import TrackerDebugRecorder
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
    handcrafted_match_enlarge: float = 1.25
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
    deep_feature_layer: int | None = 12
    normalize_deep_features: bool = False
    deep_motion_momentum: float = 0.5
    deep_template_rotations_deg: tuple[float, ...] = (0.0, -45.0, 45.0, 90.0)
    device: str = "auto"
    use_amp: bool = False
    deep_search_enlarge: float = 2.5
    deep_template_enlarge: float = 2.0  # 模板提取时的上下文扩展倍率（小目标加大可获取更多背景）
    deep_scale_update_confidence: float = 0.58
    min_target_size_ratio: float = 0.25
    max_target_size_ratio: float = 4.0
    max_output_width_ratio: float = 0.75

    # --- 三模板记忆参数（Phase 2）---
    num_templates: int = 3
    template_max_age: int = 50
    template_update_ema: float = 0.08
    template_update_background: float = 0.02

    # --- 三模板加权融合参数 ---
    template_weight_init: float = 0.40
    template_weight_short: float = 0.35
    template_weight_long: float = 0.25

    # --- 遮挡/异常帧抑制参数 ---
    occlusion_threshold: float = 0.35
    occlusion_suppress_frames: int = 3
    max_score_drop: float = 0.20

    # --- 全局重定位优化参数 ---
    relocalize_min_interval: int = 10
    relocalize_confidence_threshold: float = 0.35

    # --- 遮挡/异常帧抑制参数（P1）---
    confirmation_frames: int = 2        # 连续高分帧数达到此值才更新模板
    update_quality_threshold: float = 0.65  # 高于此分才计入"确认"计数（比 high_confidence 更严格）

    # --- 深度响应图置信度阈值 ---
    deep_high_confidence: float = 0.52
    deep_occlusion_threshold: float = 0.42
    deep_relocalize_confidence_threshold: float = 0.42
    deep_update_quality_threshold: float = 0.55

    # --- 极区自适应参数（P3）---
    polar_lat_threshold_deg: float = 55.0    # 纬度超过此值视为极区
    polar_rotation_angles_deg: tuple[float, ...] = (0.0, -30.0, 30.0, -60.0, 60.0, 90.0, -90.0, -120.0, 120.0, 150.0)
    polar_template_enlarge: float = 5.0       # 极区模板扩大更多上下文

    # --- 增强重定位参数（P4）---
    relocalize_multi_scale: bool = True           # 重定位时多尺度搜索
    relocalize_extra_scales: tuple[float, ...] = (0.7, 1.3)   # 额外尺度因子
    relocalize_use_init_only: bool = True         # 重定位时只用init模板
    relocalize_lost_trigger: int = 5              # 连续丢N帧强制重定位
    relocalize_reset_on_success: bool = True       # 重定位成功高置信时重置模板
    relocalize_reset_score: float = 0.55          # 触发重置的分数阈值
    relocalize_start_frame: int = 15               # 前N帧禁止重定位（避免早期假阳性）

    # --- 尺度更新控制 ---
    deep_scale_update_confidence: float = 0.6  # 深度模式低于此分冻结尺度更新

    # --- 可视化调试 ---
    debug_dir: str | None = None
    debug_start_frame: int = 0
    debug_max_frames: int = 20
    debug_frame_stride: int = 1
    debug_save_response_maps: bool = True


@dataclass
class TrackerRuntimeStats:
    frames: int = 0
    local_searches: int = 0
    relocalizations: int = 0
    template_updates: int = 0
    response_maps: int = 0
    response_batches: int = 0
    last_score: float = 0.0
    last_peak: float = 0.0
    last_psr: float = 0.0
    last_apce: float = 0.0
    deep_forward_calls_start: int = 0


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
        self.lost_frames = 0
        self.frame_shape: tuple[int, int] | None = None
        self._init_equatorial_width: float | None = None
        self._init_angular_height: float | None = None
        self._init_bbox_width_px: float | None = None
        self._init_bbox_height_px: float | None = None

        # --- 多模板存储（列表，长度 ≤ num_templates）---
        self._templates: list[np.ndarray] = []
        self._descriptors: list[np.ndarray] = []
        self._template_feats: list[Any] = []
        self._template_ages: list[int] = []
        self._template_scores: list[float] = []
        self._frame_count: int = 0
        self._template_feat_banks: list[list[Any]] = []
        self.num_templates = self.config.num_templates

# --- 三模板类型标记 ---
        self._template_types: list[str] = []

        # --- 遮挡抑制状态 ---
        self._consecutive_good: int = 0

        # --- 向后兼容别名（指向列表第一个元素）---
        self.template: np.ndarray | None = None
        self.template_descriptor: np.ndarray | None = None
        self.template_feat: Any = None

        # --- 深度特征提取器 ---
        self.deep_extractor = deep_extractor
        self.similarity_head = similarity_head
        self._deep_mode = False
        if self.config.use_deep_features and deep_extractor is not None and similarity_head is not None:
            self._deep_mode = True

        # --- 遮挡/异常帧抑制状态 ---
        self._occlusion_frames = 0
        self._last_high_conf_score = 0.0

        # --- 全局重定位冷却状态 ---
        self._last_relocalize_frame = -self.config.relocalize_min_interval
        self.runtime_stats = TrackerRuntimeStats()
        self._debug_payload: dict[str, np.ndarray] | None = None
        self._debug_recorder: TrackerDebugRecorder | None = None
        if self.config.debug_dir:
            self._debug_recorder = TrackerDebugRecorder(
                debug_dir=self.config.debug_dir,
                start_frame=self.config.debug_start_frame,
                max_frames=self.config.debug_max_frames,
                frame_stride=self.config.debug_frame_stride,
                save_response_maps=self.config.debug_save_response_maps,
            )

    def reset_runtime_stats(self) -> None:
        forward_calls = getattr(self.deep_extractor, "forward_calls", 0)
        self.runtime_stats = TrackerRuntimeStats(deep_forward_calls_start=int(forward_calls))

    def get_runtime_stats(self) -> dict[str, float | int]:
        stats = self.runtime_stats
        current_forward_calls = int(getattr(self.deep_extractor, "forward_calls", 0))
        return {
            "frames": stats.frames,
            "local_searches": stats.local_searches,
            "relocalizations": stats.relocalizations,
            "template_updates": stats.template_updates,
            "response_maps": stats.response_maps,
            "response_batches": stats.response_batches,
            "last_score": stats.last_score,
            "last_peak": stats.last_peak,
            "last_psr": stats.last_psr,
            "last_apce": stats.last_apce,
            "deep_forward_calls": current_forward_calls - stats.deep_forward_calls_start,
        }

    def _high_confidence(self) -> float:
        return self.config.deep_high_confidence if self._deep_mode else self.config.high_confidence

    def _occlusion_threshold(self) -> float:
        return self.config.deep_occlusion_threshold if self._deep_mode else self.config.occlusion_threshold

    def _relocalize_confidence_threshold(self) -> float:
        if self._deep_mode:
            return self.config.deep_relocalize_confidence_threshold
        return self.config.relocalize_confidence_threshold

    def _update_quality_threshold(self) -> float:
        if self._deep_mode:
            return self.config.deep_update_quality_threshold
        return self.config.update_quality_threshold

    def initialize(self, frame: np.ndarray, init_bbox_xywh: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        self.state = erp_bbox_to_state(init_bbox_xywh, w, h)
        self._init_equatorial_width = self.state.equatorial_width
        self._init_angular_height = self.state.angular_height
        self._init_bbox_width_px = float(init_bbox_xywh[2])
        self._init_bbox_height_px = float(init_bbox_xywh[3])
        self._frame_count = 0
        self.lost_frames = 0
        self._consecutive_good = 0
        self.velocity[:] = 0.0

        self._templates.clear()
        self._descriptors.clear()
        self._template_feats.clear()
        self._template_ages.clear()
        self._template_scores.clear()
        self._template_feat_banks.clear()
        self._template_types.clear()

        self._occlusion_frames = 0
        self._last_high_conf_score = 0.0
        self._last_relocalize_frame = -self.config.relocalize_min_interval
        self.reset_runtime_stats()

        self._add_template(frame, self.state, template_type="init")

        if self.num_templates > 1:
            offsets = self._template_offsets()[1:]
            for i, (off_lon, off_lat) in enumerate(offsets):
                perturbed = SphereState(
                    lon=float(wrap_lon(self.state.lon + off_lon)),
                    lat=float(clamp_lat(self.state.lat + off_lat)),
                    equatorial_width=self.state.equatorial_width,
                    angular_height=self.state.angular_height,
                )
                t_type = "short" if i == 0 else "long"
                self._add_template(frame, perturbed, template_type=t_type)

        self._sync_aliases()
        self.initialized = True
        init_bbox = self._state_to_output_bbox(self.state, w, h)
        if self._debug_recorder is not None:
            self._debug_recorder.record_initialize(
                frame=frame,
                init_bbox_xywh=init_bbox,
                template_patch=self._templates[0] if self._templates else None,
                metadata={
                    "branch": "deep" if self._deep_mode else "handcrafted",
                    "lat": f"{self.state.lat:.6f}",
                    "lon": f"{self.state.lon:.6f}",
                },
            )
        return init_bbox

    def _template_offsets(self) -> list[tuple[float, float]]:
        """生成 num_templates 个模板的初始位置偏移（弧度）。"""
        if self.num_templates <= 1:
            return [(0.0, 0.0)]
        offsets = [(0.0, 0.0)]
        # 后续模板用目标尺寸的小比例偏移
        step_lon = self.state.equatorial_width * 0.08
        step_lat = self.state.angular_height * 0.08
        for i in range(1, self.num_templates):
            angle = 2.0 * math.pi * i / (self.num_templates - 1)
            offsets.append((step_lon * math.cos(angle), step_lat * math.sin(angle)))
        return offsets[: self.num_templates]

    def _clamp_target_size(self, equatorial_width: float, angular_height: float) -> tuple[float, float]:
        if self._init_equatorial_width is None or self._init_angular_height is None:
            return float(equatorial_width), float(angular_height)

        min_ratio = max(float(self.config.min_target_size_ratio), 1e-3)
        max_ratio = max(float(self.config.max_target_size_ratio), min_ratio)
        min_width = self._init_equatorial_width * min_ratio
        max_width = self._init_equatorial_width * max_ratio
        min_height = self._init_angular_height * min_ratio
        max_height = self._init_angular_height * max_ratio
        width = min(max(float(equatorial_width), min_width), max_width)
        height = min(max(float(angular_height), min_height), max_height)
        return width, height

    def _clamp_state_size(self, state: SphereState) -> SphereState:
        width, height = self._clamp_target_size(state.equatorial_width, state.angular_height)
        return SphereState(
            lon=state.lon,
            lat=state.lat,
            equatorial_width=width,
            angular_height=height,
        )

    def _state_to_output_bbox(self, state: SphereState, image_width: int, image_height: int) -> np.ndarray:
        bbox = state_to_erp_bbox(state, image_width, image_height)
        if self._init_bbox_width_px is None or self._init_bbox_height_px is None:
            return bbox

        max_width = min(
            float(image_width) * float(self.config.max_output_width_ratio),
            self._init_bbox_width_px * float(self.config.max_target_size_ratio),
        )
        max_height = min(
            float(image_height),
            self._init_bbox_height_px * float(self.config.max_target_size_ratio),
        )

        if bbox[2] > max_width:
            center_x = (float(bbox[0]) + 0.5 * float(bbox[2])) % float(image_width)
            bbox[2] = max_width
            bbox[0] = (center_x - 0.5 * max_width) % float(image_width)

        if bbox[3] > max_height:
            center_y = float(bbox[1]) + 0.5 * float(bbox[3])
            bbox[3] = max_height
            bbox[1] = np.clip(center_y - 0.5 * max_height, 0.0, float(image_height) - max_height)

        return bbox.astype(np.float32, copy=False)

    def _add_template(self, frame: np.ndarray, state: SphereState, template_type: str = "short") -> None:
        """提取并添加一个新模板（手工 + 深度特征）。"""
        img_patch = self._extract_template(frame, state)
        descriptor = patch_descriptor(img_patch)
        self._templates.append(img_patch)
        self._descriptors.append(descriptor)
        self._template_ages.append(0)
        self._template_scores.append(0.0)
        self._template_types.append(template_type)
        if self._deep_mode:
            feat_bank = self._extract_template_feat_bank(frame, state, trust_location=True)
            self._template_feats.append(feat_bank[0])
            self._template_feat_banks.append(feat_bank)

    def _sync_aliases(self) -> None:
        """将列表内容同步到向后兼容的别名。"""
        if self._templates:
            self.template = self._templates[0]
            self.template_descriptor = self._descriptors[0]
        if self._template_feats:
            self.template_feat = self._template_feats[0]

    def track(self, frame: np.ndarray) -> np.ndarray:
        if not self.initialized or self.state is None or self.frame_shape is None:
            raise RuntimeError("Tracker must be initialized before calling track().")

        self._frame_count += 1
        self.runtime_stats.frames += 1
        self._debug_payload = None
        h, w = self.frame_shape
        predicted = SphereState(
            lon=float(wrap_lon(self.state.lon + self.velocity[0])),
            lat=float(clamp_lat(self.state.lat + self.velocity[1])),
            equatorial_width=self.state.equatorial_width,
            angular_height=self.state.angular_height,
        )

        best_state, best_score = self._local_search(frame, predicted)

        score_drop = self._last_high_conf_score - best_score
        is_abnormal = score_drop > self.config.max_score_drop and self._last_high_conf_score > 0.0

        high_confidence = self._high_confidence()
        occlusion_threshold = self._occlusion_threshold()
        relocalize_threshold = self._relocalize_confidence_threshold()
        update_quality_threshold = self._update_quality_threshold()

        if best_score >= high_confidence:
            self._occlusion_frames = 0
            self._last_high_conf_score = best_score
        elif best_score < occlusion_threshold or is_abnormal:
            self._occlusion_frames += 1
        else:
            self._occlusion_frames = max(0, self._occlusion_frames - 1)

        should_relocalize = False
        if self._frame_count >= self.config.relocalize_start_frame:
            if best_score < relocalize_threshold:
                frames_since_relocalize = self._frame_count - self._last_relocalize_frame
                if frames_since_relocalize >= self.config.relocalize_min_interval:
                    should_relocalize = True

            # P4: 连续丢失帧数过多时强制触发重定位
            if not should_relocalize and self.lost_frames >= self.config.relocalize_lost_trigger:
                frames_since_relocalize = self._frame_count - self._last_relocalize_frame
                if frames_since_relocalize >= self.config.relocalize_min_interval:
                    should_relocalize = True

        relocalize_applied = False
        if should_relocalize:
            self.runtime_stats.relocalizations += 1
            relocalized, relocalized_score = self._global_relocalize(frame, predicted)
            if relocalized_score > best_score:
                best_state, best_score = relocalized, relocalized_score
                self._last_relocalize_frame = self._frame_count
                relocalize_applied = True

                # P4: 高置信重定位成功后重置模板
                if (self.config.relocalize_reset_on_success
                        and relocalized_score >= self.config.relocalize_reset_score):
                    self._reset_templates(frame, best_state)

        best_state = self._clamp_state_size(best_state)
        if self._deep_mode and best_score < self.config.deep_scale_update_confidence:
            best_state = SphereState(
                lon=best_state.lon,
                lat=best_state.lat,
                equatorial_width=self.state.equatorial_width,
                angular_height=self.state.angular_height,
            )
        lon_delta = lon_distance(best_state.lon, self.state.lon)
        lat_delta = best_state.lat - self.state.lat
        momentum = self.config.deep_motion_momentum if self._deep_mode else self.config.motion_momentum

        if self._occlusion_frames > self.config.occlusion_suppress_frames:
            momentum = min(momentum, 0.1)

        self.velocity[0] = momentum * self.velocity[0] + (1.0 - momentum) * lon_delta
        self.velocity[1] = momentum * self.velocity[1] + (1.0 - momentum) * lat_delta
        self.state = best_state

# --- 多模板更新（含遮挡/异常帧抑制）---
        if best_score >= update_quality_threshold:
            self._consecutive_good += 1
        else:
            self._consecutive_good = 0

        if best_score >= high_confidence:
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        if self._consecutive_good >= self.config.confirmation_frames:
            self.runtime_stats.template_updates += 1
            self._update_templates(frame, best_state, best_score)
        else:
            for i in range(len(self._template_ages)):
                self._template_ages[i] += 1

        self._sync_aliases()
        result_bbox = self._state_to_output_bbox(self.state, w, h)
        self._record_debug_frame(
            frame=frame,
            predicted=predicted,
            result_bbox=result_bbox,
            score=best_score,
            relocalize_applied=relocalize_applied,
        )
        return result_bbox

    def _find_best_template(self, frame: np.ndarray, state: SphereState) -> int:
        """对给定状态评估所有模板，返回最佳匹配的索引。"""
        best_idx = 0
        best_score = -1.0
        if self._deep_mode:
            fov_x, fov_y = state_size_to_fov(state, enlarge=self.config.deep_search_enlarge)
            search_feat = self._extract_search_feat(
                frame, state.lon, state.lat, fov_x, fov_y, refine=True,
            )
            for i, t_feat in enumerate(self._template_feats):
                template_bank = self._template_feat_banks[i] if i < len(self._template_feat_banks) else [t_feat]
                score, _ = self._score_template_bank(template_bank, search_feat)
                if score > best_score:
                    best_score = score
                    best_idx = i
        else:
            patch = self._extract_template(frame, state)
            for i, desc in enumerate(self._descriptors):
                score = float(np.mean(np.sum(desc * patch_descriptor(patch), axis=1)))
                if score > best_score:
                    best_score = score
                    best_idx = i
        return best_idx

    def _update_templates(self, frame: np.ndarray, state: SphereState, score: float) -> None:
        """三模板更新策略：保护init模板，EMA更新short/long模板。"""
        best_idx = self._find_best_template(frame, state)

        oldest_idx = int(np.argmax(self._template_ages))
        if self._template_ages[oldest_idx] > self.config.template_max_age:
            t_type = self._template_types[oldest_idx] if oldest_idx < len(self._template_types) else "short"
            if t_type != "init":
                self._templates.pop(oldest_idx)
                self._descriptors.pop(oldest_idx)
                self._template_ages.pop(oldest_idx)
                self._template_scores.pop(oldest_idx)
                self._template_types.pop(oldest_idx)
                if self._deep_mode:
                    self._template_feats.pop(oldest_idx)
                    self._template_feat_banks.pop(oldest_idx)

                new_type = "long" if t_type == "short" else "short"
                self._add_template(frame, state, template_type=new_type)
                best_idx = self._find_best_template(frame, state)

        new_patch = self._extract_template(frame, state)
        for i in range(len(self._templates)):
            t_type = self._template_types[i] if i < len(self._template_types) else "short"
            if t_type == "init":
                self._template_ages[i] = 0

            update_rate = (
                self.config.template_update_ema if i == best_idx
                else self.config.template_update_background
            )
            if t_type == "init":
                update_rate = self.config.template_update_background  # init 缓慢更新
            self._templates[i] = (1.0 - update_rate) * self._templates[i] + update_rate * new_patch
            self._descriptors[i] = patch_descriptor(self._templates[i])
            self._template_ages[i] = 0 if i == best_idx else self._template_ages[i] + 1
            self._template_scores[i] = score if i == best_idx else self._template_scores[i]

        if self._deep_mode:
            new_feat_bank = self._extract_template_feat_bank(frame, state)
            for i in range(len(self._template_feats)):
                t_type = self._template_types[i] if i < len(self._template_types) else "short"

                update_rate = (
                    self.config.template_update_ema if i == best_idx
                    else self.config.template_update_background
                )
                if t_type == "init":
                    update_rate = self.config.template_update_background
                old_feat_bank = (
                    self._template_feat_banks[i]
                    if i < len(self._template_feat_banks)
                    else [self._template_feats[i]]
                )
                updated_bank = []
                for old_feat, new_feat in zip(old_feat_bank, new_feat_bank):
                    updated_bank.append((1.0 - update_rate) * old_feat + update_rate * new_feat)
                self._template_feat_banks[i] = updated_bank
                self._template_feats[i] = updated_bank[0]

    def _reset_templates(self, frame: np.ndarray, state: SphereState) -> None:
        """重定位成功后重新初始化所有模板。"""
        self._templates.clear()
        self._descriptors.clear()
        self._template_feats.clear()
        self._template_ages.clear()
        self._template_scores.clear()
        self._template_feat_banks.clear()
        self._template_types.clear()
        self._occlusion_frames = 0
        self._last_high_conf_score = 0.0
        self._consecutive_good = 0
        self.lost_frames = 0

        self._add_template(frame, state, template_type="init")

        if self.num_templates > 1:
            offsets = self._template_offsets()[1:]
            for i, (off_lon, off_lat) in enumerate(offsets):
                perturbed = SphereState(
                    lon=float(wrap_lon(state.lon + off_lon)),
                    lat=float(clamp_lat(state.lat + off_lat)),
                    equatorial_width=state.equatorial_width,
                    angular_height=state.angular_height,
                )
                t_type = "short" if i == 0 else "long"
                self._add_template(frame, perturbed, template_type=t_type)

        self._sync_aliases()

    def track_sequence(
        self,
        frames: Iterable[np.ndarray],
        init_bbox_xywh: np.ndarray,
    ) -> List[np.ndarray]:
        it = iter(frames)
        try:
            first = next(it)
        except StopIteration:
            return []
        outputs = [self.initialize(first, init_bbox_xywh)]
        for frame in it:
            outputs.append(self.track(frame))
        return outputs

    def _extract_template(self, frame: np.ndarray, state: SphereState) -> np.ndarray:
        size = self.config.template_size
        fov_x, fov_y = self._handcrafted_match_fov(state)
        return tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)

    def _should_capture_debug(self) -> bool:
        return self._debug_recorder is not None and self._debug_recorder.wants_frame(self._frame_count)

    def _record_debug_frame(
        self,
        frame: np.ndarray,
        predicted: SphereState,
        result_bbox: np.ndarray,
        score: float,
        relocalize_applied: bool,
    ) -> None:
        if self._debug_recorder is None or self.frame_shape is None:
            return

        predicted_bbox = self._state_to_output_bbox(predicted, self.frame_shape[1], self.frame_shape[0])
        artifacts = dict(self._debug_payload or {})
        if self._templates:
            artifacts.setdefault("template_patch", self._templates[0])

        self._debug_recorder.record_step(
            frame_index=self._frame_count,
            frame=frame,
            predicted_bbox_xywh=predicted_bbox,
            result_bbox_xywh=result_bbox,
            score=score,
            metadata={
                "branch": "deep" if self._deep_mode else "handcrafted",
                "lost_frames": self.lost_frames,
                "occlusion_frames": self._occlusion_frames,
                "relocalized": relocalize_applied,
                "last_peak": f"{self.runtime_stats.last_peak:.6f}",
                "last_psr": f"{self.runtime_stats.last_psr:.6f}",
                "last_apce": f"{self.runtime_stats.last_apce:.6f}",
                "velocity_lat": f"{self.velocity[1]:.6f}",
                "velocity_lon": f"{self.velocity[0]:.6f}",
            },
            artifacts=artifacts,
        )

    def _handcrafted_match_fov(self, state: SphereState) -> tuple[float, float]:
        """Return the target-scale FoV used by the handcrafted branch."""
        return state_size_to_fov(state, enlarge=self.config.handcrafted_match_enlarge)

    # ---------- 深度特征方法 ----------

    def _extract_template_feat(self, frame: np.ndarray, state: SphereState) -> Any:
        """提取模板的深度特征（归一化球面 patch → backbone 前向）。"""
        size = self.config.deep_template_size
        fov_x, fov_y = state_size_to_fov(state, enlarge=self.config.deep_template_enlarge)
        patch = tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)
        return self.deep_extractor.extract_template_feature(patch)

    def _rotate_patch(self, patch: np.ndarray, angle_deg: float) -> np.ndarray:
        if abs(angle_deg) < 1e-6:
            return patch
        image = Image.fromarray(np.clip(patch * 255.0, 0.0, 255.0).astype(np.uint8))
        rotated = image.rotate(float(angle_deg), resample=Image.BILINEAR)
        return np.asarray(rotated, dtype=np.float32) / 255.0

    def _is_polar(self, lat: float) -> bool:
        """判断纬度是否处于极区（高畸变区域）。"""
        return abs(math.degrees(lat)) > self.config.polar_lat_threshold_deg

    def _get_rotation_angles(self, lat: float) -> tuple[float, ...]:
        """根据纬度返回合适的旋转角度集合：极区用更多角度覆盖外观变化。"""
        if self._is_polar(lat):
            return self.config.polar_rotation_angles_deg
        return self.config.deep_template_rotations_deg

    def _get_template_enlarge(self, lat: float) -> float:
        """根据纬度返回模板提取的上下文扩大倍率。"""
        if self._is_polar(lat):
            return self.config.polar_template_enlarge
        return self.config.deep_template_enlarge

    def _extract_template_feat_bank(self, frame: np.ndarray, state: SphereState,
                                     trust_location: bool = False) -> list[Any]:
        """提取模板的深度特征bank（含旋转增强）。

        Args:
            trust_location: True=位置可信（初始化/添加模板），用极区多旋转角；
                           False=跟踪更新，只用正常旋转角避免错误匹配。
        """
        size = self.config.deep_template_size
        enlarge = self._get_template_enlarge(state.lat) if trust_location else self.config.deep_template_enlarge
        angles = self._get_rotation_angles(state.lat) if trust_location else self.config.deep_template_rotations_deg
        fov_x, fov_y = state_size_to_fov(state, enlarge=enlarge)
        patch = tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)
        feat_bank: list[Any] = []
        for angle_deg in angles:
            rotated_patch = self._rotate_patch(patch, angle_deg)
            feat_bank.append(self.deep_extractor.extract_template_feature(rotated_patch))
        return feat_bank

    def _extract_search_feat(
        self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float, refine: bool = False
    ) -> Any:
        """提取搜索区域的深度特征。"""
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        patch = self._extract_search_patch(frame, lon, lat, fov_x, fov_y, refine=refine)
        return self.deep_extractor.extract_search_feature(patch, refine=refine)

    def _extract_search_patch(
        self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float, refine: bool = False
    ) -> np.ndarray:
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        return tangent_patch(frame, lon, lat, fov_x, fov_y, out_size, out_size)

    def _response_to_numpy(self, response: Any) -> np.ndarray:
        response_np = response.squeeze().detach().float().cpu().numpy()
        if response_np.ndim == 0:
            response_np = response_np.reshape(1, 1)
        elif response_np.ndim == 1:
            response_np = response_np.reshape(1, -1)
        return response_np

    def _deep_score_and_offset(
        self, template_feat: Any, search_feat: Any
    ) -> tuple[float, tuple[float, float]]:
        response = self.similarity_head(template_feat, search_feat)
        score, offset, _ = self._response_score_offset(response, template_feat, search_feat)
        return score, offset

    def _template_weight(self, template_idx: int) -> float:
        t_type = self._template_types[template_idx] if template_idx < len(self._template_types) else "short"
        if t_type == "init":
            return self.config.template_weight_init
        if t_type == "short":
            return self.config.template_weight_short
        return self.config.template_weight_long

    def _best_bank_response(self, template_bank: list[Any], search_feat: Any) -> Any:
        if not template_bank:
            return None
        torch = getattr(self.deep_extractor, "_torch", None)
        if torch is None or len(template_bank) == 1:
            response = self.similarity_head(template_bank[0], search_feat)
            self.runtime_stats.response_maps += 1
            self.runtime_stats.response_batches += 1
            return response

        bank_feat = torch.cat(template_bank, dim=0)
        response = self.similarity_head(bank_feat, search_feat)
        self.runtime_stats.response_maps += int(response.shape[0])
        self.runtime_stats.response_batches += 1
        flat = response.flatten(start_dim=1)
        best_idx = int(flat.max(dim=1).values.argmax().detach().cpu())
        return response[best_idx : best_idx + 1]

    def _fused_template_response(self, search_feat: Any) -> tuple[Any, Any]:
        fused_response = None
        total_weight = 0.0
        reference_template = None
        for i, t_feat in enumerate(self._template_feats):
            template_bank = (
                self._template_feat_banks[i]
                if i < len(self._template_feat_banks)
                else [t_feat]
            )
            response = self._best_bank_response(template_bank, search_feat)
            if response is None:
                continue
            weight = self._template_weight(i)
            fused_response = response * weight if fused_response is None else fused_response + response * weight
            total_weight += weight
            if reference_template is None:
                reference_template = t_feat

        if fused_response is None or reference_template is None:
            raise RuntimeError("No deep templates available for scoring.")
        if total_weight > 0.0:
            fused_response = fused_response / total_weight
        return fused_response, reference_template

    def _response_confidence(self, response_np: np.ndarray, peak_y: int, peak_x: int) -> tuple[float, float, float, float]:
        peak = float(response_np[peak_y, peak_x])
        side = response_np.copy()
        y0 = max(0, peak_y - 1)
        y1 = min(response_np.shape[0], peak_y + 2)
        x0 = max(0, peak_x - 1)
        x1 = min(response_np.shape[1], peak_x + 2)
        side[y0:y1, x0:x1] = np.nan
        valid = side[np.isfinite(side)]
        if valid.size == 0:
            valid = response_np.reshape(-1)
        side_mean = float(np.mean(valid))
        side_std = float(np.std(valid))
        psr = (peak - side_mean) / max(side_std, 1e-6)

        resp_min = float(np.min(response_np))
        apce_den = float(np.mean((response_np - resp_min) ** 2))
        apce = ((peak - resp_min) ** 2) / max(apce_den, 1e-6)

        confidence = 1.0 / (1.0 + math.exp(-(psr - 2.0) / 2.0))
        return float(confidence), peak, float(psr), float(apce)

    def _response_score_offset(
        self,
        response: Any,
        template_feat: Any,
        search_feat: Any,
    ) -> tuple[float, tuple[float, float], dict[str, float]]:
        response_np = response.squeeze().detach().float().cpu().numpy()
        if response_np.ndim == 0:
            response_np = response_np.reshape(1, 1)
        elif response_np.ndim == 1:
            response_np = response_np.reshape(1, -1)
        max_idx = response_np.argmax()
        h, w = response_np.shape
        peak_y, peak_x = max_idx // w, max_idx % w
        score, peak, psr, apce = self._response_confidence(response_np, int(peak_y), int(peak_x))

        template_h, template_w = int(template_feat.shape[-2]), int(template_feat.shape[-1])
        search_h, search_w = int(search_feat.shape[-2]), int(search_feat.shape[-1])
        max_offset_y = max(search_h - template_h, 0) / (2.0 * max(search_h, 1))
        max_offset_x = max(search_w - template_w, 0) / (2.0 * max(search_w, 1))
        offset_y = ((peak_y / (h - 1)) - 0.5) * 2.0 * max_offset_y if h > 1 else 0.0
        offset_x = ((peak_x / (w - 1)) - 0.5) * 2.0 * max_offset_x if w > 1 else 0.0
        meta = {"peak": peak, "psr": psr, "apce": apce}
        return score, (float(offset_y), float(offset_x)), meta

    def _score_template_bank(
        self,
        template_bank: list[Any],
        search_feat: Any,
    ) -> tuple[float, tuple[float, float]]:
        response = self._best_bank_response(template_bank, search_feat)
        if response is None:
            return -1.0, (0.0, 0.0)
        score, offset, _ = self._response_score_offset(response, template_bank[0], search_feat)
        return score, offset

    # ---------- 打分 ----------

    def _score_patch(self, patch: np.ndarray) -> float:
        """使用三模板加权融合打分。"""
        patch_desc = patch_descriptor(patch)
        scores = []
        weights = []
        for i, desc in enumerate(self._descriptors):
            t_type = self._template_types[i] if i < len(self._template_types) else "short"
            score = float(np.mean(np.sum(desc * patch_desc, axis=1)))
            scores.append(score)
            if t_type == "init":
                weights.append(self.config.template_weight_init)
            elif t_type == "short":
                weights.append(self.config.template_weight_short)
            else:
                weights.append(self.config.template_weight_long)
        if not scores:
            return -1.0
        total_weight = sum(weights) if sum(weights) > 0 else len(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return float(weighted_score)

    def _score_patch_deep(self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float) -> float:
        """深度特征三模板加权融合打分。"""
        search_feat = self._extract_search_feat(frame, lon, lat, fov_x, fov_y)
        response, reference_template = self._fused_template_response(search_feat)
        score, _, meta = self._response_score_offset(response, reference_template, search_feat)
        self.runtime_stats.last_score = score
        self.runtime_stats.last_peak = meta["peak"]
        self.runtime_stats.last_psr = meta["psr"]
        self.runtime_stats.last_apce = meta["apce"]
        return float(score)

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
        best_patch: np.ndarray | None = None
        capture_debug = self._should_capture_debug()

        for scale in self.config.scale_factors:
            candidate_width, candidate_height = self._clamp_target_size(
                predicted.equatorial_width * scale,
                predicted.angular_height * scale,
            )
            candidate = SphereState(
                lon=predicted.lon,
                lat=predicted.lat,
                equatorial_width=candidate_width,
                angular_height=candidate_height,
            )
            fov_x, fov_y = self._handcrafted_match_fov(candidate)

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
                        if capture_debug:
                            best_patch = patch.copy()
                        best_state = SphereState(
                            lon=float(lon), lat=float(lat),
                            equatorial_width=float(candidate_width),
                            angular_height=float(candidate_height),
                        )
        if capture_debug:
            self._debug_payload = {}
            if best_patch is not None:
                self._debug_payload["match_patch"] = best_patch
        return best_state, best_score

    def _local_search_deep(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        """深度特征 coarse-to-fine 搜索（三模板加权融合版本）。"""
        self.runtime_stats.local_searches += 1
        best_state = predicted
        best_score = -1.0
        best_debug_payload: dict[str, np.ndarray] | None = None
        capture_debug = self._should_capture_debug()

        for scale in self.config.scale_factors:
            candidate_width, candidate_height = self._clamp_target_size(
                predicted.equatorial_width * scale,
                predicted.angular_height * scale,
            )
            fov_x, fov_y = state_size_to_fov(
                SphereState(predicted.lon, predicted.lat, candidate_width, candidate_height),
                enlarge=self.config.deep_search_enlarge,
            )

            scale_best_score = -1.0
            scale_best_state = predicted
            scale_best_debug_payload: dict[str, np.ndarray] | None = None

            coarse_patch = None
            if capture_debug:
                coarse_patch = self._extract_search_patch(
                    frame, predicted.lon, predicted.lat, fov_x, fov_y, refine=False,
                )
                search_feat = self.deep_extractor.extract_search_feature(coarse_patch, refine=False)
            else:
                search_feat = self._extract_search_feat(
                    frame, predicted.lon, predicted.lat, fov_x, fov_y, refine=False,
                )
            response, reference_template = self._fused_template_response(search_feat)
            score, (off_y, off_x), _ = self._response_score_offset(
                response, reference_template, search_feat,
            )
            coarse_response = self._response_to_numpy(response) if capture_debug else None

            coarse_lon = wrap_lon(predicted.lon + off_x * fov_x)
            coarse_lat = clamp_lat(predicted.lat + off_y * fov_y)

            refine_fov_x = fov_x * 0.5
            refine_fov_y = fov_y * 0.5
            refine_patch = None
            if capture_debug:
                refine_patch = self._extract_search_patch(
                    frame, coarse_lon, coarse_lat, refine_fov_x, refine_fov_y, refine=True,
                )
                refine_feat = self.deep_extractor.extract_search_feature(refine_patch, refine=True)
            else:
                refine_feat = self._extract_search_feat(
                    frame, coarse_lon, coarse_lat, refine_fov_x, refine_fov_y, refine=True,
                )
            refine_response, refine_reference = self._fused_template_response(refine_feat)
            refine_score, (roff_y, roff_x), refine_meta = self._response_score_offset(
                refine_response, refine_reference, refine_feat,
            )
            refine_response_np = self._response_to_numpy(refine_response) if capture_debug else None

            refined_lon = wrap_lon(coarse_lon + roff_x * refine_fov_x)
            refined_lat = clamp_lat(coarse_lat + roff_y * refine_fov_y)

            combined_score = 0.3 * score + 0.7 * refine_score
            combined_score -= 0.02 * abs(scale - 1.0)

            if combined_score > scale_best_score:
                scale_best_score = combined_score
                scale_best_state = SphereState(
                    lon=float(refined_lon), lat=float(refined_lat),
                    equatorial_width=float(candidate_width),
                    angular_height=float(candidate_height),
                )
                if capture_debug:
                    scale_best_debug_payload = {}
                    if coarse_patch is not None:
                        scale_best_debug_payload["coarse_patch"] = coarse_patch
                    if coarse_response is not None:
                        scale_best_debug_payload["coarse_response"] = coarse_response
                    if refine_patch is not None:
                        scale_best_debug_payload["refine_patch"] = refine_patch
                    if refine_response_np is not None:
                        scale_best_debug_payload["refine_response"] = refine_response_np

            if scale_best_score > best_score:
                best_score = scale_best_score
                best_state = scale_best_state
                self.runtime_stats.last_score = float(best_score)
                self.runtime_stats.last_peak = refine_meta["peak"]
                self.runtime_stats.last_psr = refine_meta["psr"]
                self.runtime_stats.last_apce = refine_meta["apce"]
                if capture_debug:
                    best_debug_payload = scale_best_debug_payload

        if capture_debug:
            self._debug_payload = best_debug_payload or {}
        return best_state, best_score

    def _global_relocalize(
        self,
        frame: np.ndarray,
        predicted: SphereState,
    ) -> tuple[SphereState, float]:
        coarse_stride_lon = self.config.relocalize_stride_deg
        coarse_stride_lat = self.config.relocalize_lat_stride_deg
        fine_stride_lon = coarse_stride_lon * 0.5
        fine_stride_lat = coarse_stride_lat * 0.5

        lon_values_coarse = np.deg2rad(
            np.arange(-180.0, 180.0, coarse_stride_lon, dtype=np.float32)
        )
        lat_values_coarse = np.deg2rad(
            np.arange(-72.0, 72.1, coarse_stride_lat, dtype=np.float32)
        )

        # P4: 多尺度搜索
        scales = [1.0]
        if self.config.relocalize_multi_scale:
            scales.extend(self.config.relocalize_extra_scales)

        best_overall_state = predicted
        best_overall_score = -1.0

        for reloc_scale in scales:
            reloc_state = SphereState(
                lon=predicted.lon,
                lat=predicted.lat,
                equatorial_width=predicted.equatorial_width * reloc_scale,
                angular_height=predicted.angular_height * reloc_scale,
            )

            if self._deep_mode:
                fov_x, fov_y = state_size_to_fov(reloc_state, enlarge=self.config.deep_search_enlarge)
            else:
                fov_x, fov_y = self._handcrafted_match_fov(reloc_state)

            candidates: list[tuple[float, SphereState]] = []

            for lon in lon_values_coarse:
                for lat in lat_values_coarse:
                    lat_val = clamp_lat(float(lat))
                    if self._deep_mode:
                        score = self._reloc_score_deep(frame, float(lon), lat_val, fov_x, fov_y)
                    else:
                        patch = tangent_patch(
                            frame, float(lon), lat_val, fov_x, fov_y,
                            self.config.template_size, self.config.template_size,
                        )
                        score = self._score_patch(patch)
                    dist_lon = abs(lon_distance(float(lon), predicted.lon))
                    dist_lat = abs(lat_val - predicted.lat)
                    score -= 0.01 * dist_lon + 0.02 * dist_lat
                    candidates.append((score, SphereState(
                        lon=float(lon), lat=lat_val,
                        equatorial_width=reloc_state.equatorial_width,
                        angular_height=reloc_state.angular_height,
                    )))

            candidates.sort(key=lambda item: item[0], reverse=True)
            best_state = predicted
            best_score = -1.0

            for coarse_score, coarse_state in candidates[: self.config.relocalize_topk]:
                if coarse_score < self._occlusion_threshold():
                    continue

                lon_range = np.deg2rad(np.arange(
                    -fine_stride_lon, fine_stride_lon + 1e-6, fine_stride_lon, dtype=np.float32
                ))
                lat_range = np.deg2rad(np.arange(
                    -fine_stride_lat, fine_stride_lat + 1e-6, fine_stride_lat, dtype=np.float32
                ))

                fine_candidates: list[tuple[float, SphereState]] = []
                for dlon in lon_range:
                    for dlat in lat_range:
                        lon_val = wrap_lon(coarse_state.lon + dlon)
                        lat_val = clamp_lat(coarse_state.lat + dlat)
                        if self._deep_mode:
                            score = self._reloc_score_deep(frame, float(lon_val), float(lat_val), fov_x, fov_y)
                        else:
                            patch = tangent_patch(
                                frame, float(lon_val), float(lat_val), fov_x, fov_y,
                                self.config.template_size, self.config.template_size,
                            )
                            score = self._score_patch(patch)
                        fine_candidates.append((score, SphereState(
                            lon=float(lon_val), lat=float(lat_val),
                            equatorial_width=coarse_state.equatorial_width,
                            angular_height=coarse_state.angular_height,
                        )))

                if fine_candidates:
                    fine_candidates.sort(key=lambda item: item[0], reverse=True)
                    best_fine_state = fine_candidates[0][1]
                    best_fine_score = fine_candidates[0][0]
                else:
                    best_fine_state = coarse_state
                    best_fine_score = coarse_score

                refined_state, refined_score = self._local_search(frame, best_fine_state)
                final_score = max(refined_score, best_fine_score)

                if final_score > best_score:
                    best_state = refined_state
                    best_score = final_score

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_state = best_state

        return best_overall_state, best_overall_score

    def _reloc_score_deep(self, frame: np.ndarray, lon: float, lat: float,
                          fov_x: float, fov_y: float) -> float:
        """P4: 重定位专用的深度特征打分，支持 init-only 模式。"""
        search_feat = self._extract_search_feat(frame, lon, lat, fov_x, fov_y)

        if self.config.relocalize_use_init_only and self._template_feats:
            # 只用 init 模板（index 0），最可靠
            init_bank = (self._template_feat_banks[0]
                         if self._template_feat_banks else [self._template_feats[0]])
            score, _ = self._score_template_bank(init_bank, search_feat)
            return float(score)

        # Fallback: 标准多模板加权打分
        return self._score_patch_deep(frame, lon, lat, fov_x, fov_y)
