from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, List

import numpy as np
from PIL import Image

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
    deep_motion_momentum: float = 0.5
    deep_template_rotations_deg: tuple[float, ...] = (0.0, -45.0, 45.0, 90.0)
    device: str = "cpu"
    use_amp: bool = False
    deep_search_enlarge: float = 2.5

    # --- 三模板记忆参数（Phase 2）---
    num_templates: int = 3
    template_max_age: int = 50  # 超过此帧数未匹配的模板被替换
    template_update_ema: float = 0.08  # 匹配到的模板EMA更新率
    template_update_background: float = 0.02  # 未匹配模板的微弱更新率

    # --- 遮挡/异常帧抑制参数（P1）---
    confirmation_frames: int = 2        # 连续高分帧数达到此值才更新模板
    update_quality_threshold: float = 0.65  # 高于此分才计入"确认"计数（比 high_confidence 更严格）


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

        # --- 多模板存储（列表，长度 ≤ num_templates）---
        self._templates: list[np.ndarray] = []           # 图像模板
        self._descriptors: list[np.ndarray] = []          # 手工特征描述子
        self._template_feats: list[Any] = []               # 深度特征
        self._template_ages: list[int] = []                # 各模板的存活帧数
        self._template_scores: list[float] = []            # 最近一次匹配分数
        self._frame_count: int = 0                         # 总帧数计数器
        self._template_feat_banks: list[list[Any]] = []
        self.num_templates = self.config.num_templates

        # --- 遮挡抑制状态 ---
        self._consecutive_good: int = 0  # 连续高质量帧计数器

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

    def initialize(self, frame: np.ndarray, init_bbox_xywh: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        self.state = erp_bbox_to_state(init_bbox_xywh, w, h)
        self._frame_count = 0
        self.lost_frames = 0
        self._consecutive_good = 0
        self.velocity[:] = 0.0

        # 清空多模板存储
        self._templates.clear()
        self._descriptors.clear()
        self._template_feats.clear()
        self._template_ages.clear()
        self._template_scores.clear()
        self._template_feat_banks.clear()

        # 生成 num_templates 个初始模板（轻微位置扰动）
        offsets = self._template_offsets()
        for off_lon, off_lat in offsets:
            perturbed = SphereState(
                lon=float(wrap_lon(self.state.lon + off_lon)),
                lat=float(clamp_lat(self.state.lat + off_lat)),
                equatorial_width=self.state.equatorial_width,
                angular_height=self.state.angular_height,
            )
            self._add_template(frame, perturbed)

        # 向后兼容别名
        self._sync_aliases()
        self.initialized = True
        return state_to_erp_bbox(self.state, w, h)

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

    def _add_template(self, frame: np.ndarray, state: SphereState) -> None:
        """提取并添加一个新模板（手工 + 深度特征）。"""
        img_patch = self._extract_template(frame, state)
        descriptor = patch_descriptor(img_patch)
        self._templates.append(img_patch)
        self._descriptors.append(descriptor)
        self._template_ages.append(0)
        self._template_scores.append(0.0)
        if self._deep_mode:
            feat_bank = self._extract_template_feat_bank(frame, state)
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
        momentum = self.config.deep_motion_momentum if self._deep_mode else self.config.motion_momentum
        self.velocity[0] = momentum * self.velocity[0] + (1.0 - momentum) * lon_delta
        self.velocity[1] = momentum * self.velocity[1] + (1.0 - momentum) * lat_delta
        self.state = best_state

        # --- 多模板更新（含遮挡/异常帧抑制）---
        # 高质量帧：增加"确认"计数；否则清零
        if best_score >= self.config.update_quality_threshold:
            self._consecutive_good += 1
        else:
            self._consecutive_good = 0

        # 高置信帧：解除丢失状态
        if best_score >= self.config.high_confidence:
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        # 只有连续确认达标才更新模板（防止在错误帧上学习）
        if self._consecutive_good >= self.config.confirmation_frames:
            self._update_templates(frame, best_state, best_score)
        else:
            # 低质量帧：模板仅年龄增长，不更新内容
            for i in range(len(self._template_ages)):
                self._template_ages[i] += 1

        self._sync_aliases()
        return state_to_erp_bbox(self.state, w, h)

    def _find_best_template(self, frame: np.ndarray, state: SphereState) -> int:
        """对给定状态评估所有模板，返回最佳匹配的索引。"""
        best_idx = 0
        best_score = -1.0
        if self._deep_mode:
            fov_x, fov_y = state_size_to_fov(state, enlarge=self.config.deep_search_enlarge)
            for i, t_feat in enumerate(self._template_feats):
                template_bank = self._template_feat_banks[i] if i < len(self._template_feat_banks) else [t_feat]
                search_feat = self._extract_search_feat(
                    frame, state.lon, state.lat, fov_x, fov_y, refine=True,
                )
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
        """多模板更新策略：EMA 更新最佳匹配模板，替换过期模板。"""
        best_idx = self._find_best_template(frame, state)

        # 定位最老模板（用于替换）
        oldest_idx = int(np.argmax(self._template_ages))

        # 如果最老模板太旧，替换为新模板
        if self._template_ages[oldest_idx] > self.config.template_max_age:
            self._templates.pop(oldest_idx)
            self._descriptors.pop(oldest_idx)
            self._template_ages.pop(oldest_idx)
            self._template_scores.pop(oldest_idx)
            if self._deep_mode:
                self._template_feats.pop(oldest_idx)
                self._template_feat_banks.pop(oldest_idx)
            # 重新添加新模板
            self._add_template(frame, state)
            # 重新定位最佳索引（列表已变）
            best_idx = self._find_best_template(frame, state)

        # EMA 更新：最佳匹配模板更新率最高，其他模板微弱更新
        for i in range(len(self._templates)):
            update_rate = (
                self.config.template_update_ema if i == best_idx
                else self.config.template_update_background
            )
            new_patch = self._extract_template(frame, state)
            self._templates[i] = (1.0 - update_rate) * self._templates[i] + update_rate * new_patch
            self._descriptors[i] = patch_descriptor(self._templates[i])
            self._template_ages[i] = 0 if i == best_idx else self._template_ages[i] + 1
            self._template_scores[i] = score if i == best_idx else self._template_scores[i]

        if self._deep_mode:
            for i in range(len(self._template_feats)):
                update_rate = (
                    self.config.template_update_ema if i == best_idx
                    else self.config.template_update_background
                )
                new_feat_bank = self._extract_template_feat_bank(frame, state)
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

    def _handcrafted_match_fov(self, state: SphereState) -> tuple[float, float]:
        """Return the target-scale FoV used by the handcrafted branch."""
        return state_size_to_fov(state, enlarge=self.config.handcrafted_match_enlarge)

    # ---------- 深度特征方法 ----------

    def _extract_template_feat(self, frame: np.ndarray, state: SphereState) -> Any:
        """提取模板的深度特征（归一化球面 patch → backbone 前向）。"""
        size = self.config.deep_template_size
        fov_x, fov_y = state_size_to_fov(state, enlarge=1.25)
        patch = tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)
        return self.deep_extractor.extract_template_feature(patch)

    def _rotate_patch(self, patch: np.ndarray, angle_deg: float) -> np.ndarray:
        if abs(angle_deg) < 1e-6:
            return patch
        image = Image.fromarray(np.clip(patch * 255.0, 0.0, 255.0).astype(np.uint8))
        rotated = image.rotate(float(angle_deg), resample=Image.BILINEAR)
        return np.asarray(rotated, dtype=np.float32) / 255.0

    def _extract_template_feat_bank(self, frame: np.ndarray, state: SphereState) -> list[Any]:
        size = self.config.deep_template_size
        fov_x, fov_y = state_size_to_fov(state, enlarge=1.25)
        patch = tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)
        feat_bank: list[Any] = []
        for angle_deg in self.config.deep_template_rotations_deg:
            rotated_patch = self._rotate_patch(patch, angle_deg)
            feat_bank.append(self.deep_extractor.extract_template_feature(rotated_patch))
        return feat_bank

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
        template_h, template_w = int(template_feat.shape[-2]), int(template_feat.shape[-1])
        search_h, search_w = int(search_feat.shape[-2]), int(search_feat.shape[-1])
        max_offset_y = max(search_h - template_h, 0) / (2.0 * max(search_h, 1))
        max_offset_x = max(search_w - template_w, 0) / (2.0 * max(search_w, 1))
        offset_y = ((peak_y / (h - 1)) - 0.5) * 2.0 * max_offset_y if h > 1 else 0.0
        offset_x = ((peak_x / (w - 1)) - 0.5) * 2.0 * max_offset_x if w > 1 else 0.0
        return score, (offset_y, offset_x)

    def _score_template_bank(
        self,
        template_bank: list[Any],
        search_feat: Any,
    ) -> tuple[float, tuple[float, float]]:
        best_score = -1.0
        best_offset = (0.0, 0.0)
        for template_feat in template_bank:
            score, offset = self._deep_score_and_offset(template_feat, search_feat)
            if score > best_score:
                best_score = score
                best_offset = offset
        return best_score, best_offset

    # ---------- 打分 ----------

    def _score_patch(self, patch: np.ndarray) -> float:
        """使用所有模板描述子匹配，返回最高分数。"""
        best = -1.0
        for desc in self._descriptors:
            score = float(np.mean(np.sum(desc * patch_descriptor(patch), axis=1)))
            if score > best:
                best = score
        return best

    def _score_patch_deep(self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float) -> float:
        """深度特征打分：对所有模板做 cross-correlation，返回最高分数。"""
        search_feat = self._extract_search_feat(frame, lon, lat, fov_x, fov_y)
        best_score = -1.0
        for i, t_feat in enumerate(self._template_feats):
            template_bank = (
                self._template_feat_banks[i]
                if i < len(self._template_feat_banks)
                else [t_feat]
            )
            score, _ = self._score_template_bank(template_bank, search_feat)
            if score > best_score:
                best_score = score
        return best_score

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
                        best_state = SphereState(
                            lon=float(lon), lat=float(lat),
                            equatorial_width=float(candidate_width),
                            angular_height=float(candidate_height),
                        )
        return best_state, best_score

    def _local_search_deep(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        """深度特征 coarse-to-fine 搜索（多模板版本）。

        Coarse: 在预测位置提取大范围搜索patch，对所有模板做 cross-correlation，取最高分。
        Refine: 在最佳候选位置附近小范围高分辨率精修。
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

            # 每个 scale 对所有模板尝试匹配
            scale_best_score = -1.0
            scale_best_state = predicted
            scale_best_template_idx = 0

            for t_idx, t_feat in enumerate(self._template_feats):
                template_bank = (
                    self._template_feat_banks[t_idx]
                    if t_idx < len(self._template_feat_banks)
                    else [t_feat]
                )
                # --- Coarse stage ---
                search_feat = self._extract_search_feat(
                    frame, predicted.lon, predicted.lat, fov_x, fov_y, refine=False,
                )
                score, (off_y, off_x) = self._score_template_bank(template_bank, search_feat)

                # 将 offset 映射回球面坐标
                coarse_lon = wrap_lon(predicted.lon + off_x * fov_x)
                coarse_lat = clamp_lat(predicted.lat + off_y * fov_y)

                # --- Refine stage ---
                refine_fov_x = fov_x * 0.5
                refine_fov_y = fov_y * 0.5
                refine_feat = self._extract_search_feat(
                    frame, coarse_lon, coarse_lat, refine_fov_x, refine_fov_y, refine=True,
                )
                refine_score, (roff_y, roff_x) = self._score_template_bank(template_bank, refine_feat)

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
                    scale_best_template_idx = t_idx

            if scale_best_score > best_score:
                best_score = scale_best_score
                best_state = scale_best_state
                # 更新最佳匹配模板的分数
                self._template_scores[scale_best_template_idx] = scale_best_score

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

        if self._deep_mode:
            fov_x, fov_y = state_size_to_fov(predicted, enlarge=self.config.deep_search_enlarge)
        else:
            fov_x, fov_y = self._handcrafted_match_fov(predicted)
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
