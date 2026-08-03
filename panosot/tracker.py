from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
from typing import Any, Iterable, List

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # OpenCV is optional; template matching remains available.
    cv2 = None

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
    deep_motion_momentum: float = 0.3
    deep_template_rotations_deg: tuple[float, ...] = (0.0, -45.0, 45.0, 90.0)
    device: str = "auto"
    use_amp: bool = False
    deep_search_enlarge: float = 2.5
    deep_template_enlarge: float = 4.0
    deep_scale_update_confidence: float = 0.58
    min_target_size_ratio: float = 0.25
    max_target_size_ratio: float = 8.0
    max_output_width_ratio: float = 0.75

    # --- 深度模式尺度因子（扩展范围以适应小目标）---
    deep_scale_factors: tuple[float, ...] = (0.85, 0.95, 1.0, 1.05, 1.15)
    deep_refine_topk: int = 3
    # Deep relocalization is expensive; keep the nominal scale and the
    # expansion scale that covers the long-sequence size jump, then refine
    # only the strongest coarse location.
    deep_relocalize_scales: tuple[float, ...] = (1.0, 1.3)
    deep_relocalize_topk: int = 1

    # --- 三模板记忆参数（Phase 2）---
    num_templates: int = 3
    template_max_age: int = 50
    template_update_ema: float = 0.08
    template_update_background: float = 0.0
    deep_template_update_ema: float = 0.02
    deep_template_update_background: float = 0.0
    deep_confirmation_frames: int = 5

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
    confirmation_frames: int = 2
    update_quality_threshold: float = 0.65

    # --- 深度响应图置信度阈值 ---
    deep_high_confidence: float = 0.52
    deep_occlusion_threshold: float = 0.42
    deep_relocalize_confidence_threshold: float = 0.42
    deep_update_quality_threshold: float = 0.55

    # --- 极区自适应参数（P3）---
    polar_lat_threshold_deg: float = 55.0
    polar_rotation_angles_deg: tuple[float, ...] = (0.0, -30.0, 30.0, -60.0, 60.0, 90.0, -90.0, -120.0, 120.0, 150.0)
    polar_template_enlarge: float = 5.0

    # --- 增强重定位参数（P4）---
    relocalize_multi_scale: bool = True
    relocalize_extra_scales: tuple[float, ...] = (0.7, 1.3)
    relocalize_use_init_only: bool = True
    relocalize_lost_trigger: int = 5
    relocalize_reset_on_success: bool = True
    relocalize_reset_score: float = 0.55
    relocalize_start_frame: int = 15

    # --- 尺度更新控制 ---
    deep_scale_update_confidence: float = 0.6

    # --- 可视化调试 ---
    debug_dir: str | None = None
    debug_start_frame: int = 0
    debug_max_frames: int = 20
    debug_frame_stride: int = 1
    debug_save_response_maps: bool = True

    # --- 重定位守门 ---
    relocalize_min_start_frame_override: int = 25
    relocalize_min_lost_frames_override: int = 8
    relocalize_score_margin: float = 0.15
    relocalize_jump_gate_frames: int = 30
    relocalize_jump_gate_lost_frames: int = 10
    relocalize_max_lon_jump_deg: float = 40.0
    relocalize_max_lat_jump_deg: float = 18.0

    # --- 低置信局部更新守门 ---
    deep_local_jump_gate_confidence: float = 0.54
    deep_state_trust_low: float = 0.46
    deep_state_trust_high: float = 0.58
    deep_state_trust_min: float = 0.12
    deep_state_trust_psr_center: float = 2.0
    deep_state_trust_psr_scale: float = 0.75
    deep_velocity_low_psr_threshold: float = 1.85
    deep_velocity_low_trust_threshold: float = 0.65
    deep_velocity_decay: float = 0.20
    deep_fallback_psr_threshold: float = 2.0
    # A low deep PSR only requests a fallback; it must not force-accept a
    # weak handcrafted candidate and overwrite the tracking state.
    deep_fallback_min_hand_score: float = 0.18
    deep_fallback_score_margin: float = 0.03
    deep_probe_interval: int = 20
    deep_probe_backoff_after: int = 3
    deep_probe_backoff_multiplier: float = 2.0
    deep_probe_max_interval: int = 20
    deep_relocalize_psr_threshold: float = 1.70
    deep_relocalize_accept_psr_threshold: float = 1.75
    deep_relocalize_backoff_after: int = 2
    deep_relocalize_backoff_multiplier: float = 2.0
    deep_relocalize_max_interval: int = 80
    deep_relocalize_min_start_frame: int = 6
    deep_relocalize_min_lost_frames: int = 2
    deep_ncc_scale_collapse_ratio: float = 0.55
    deep_probe_ncc_disagreement_lon_deg: float = 12.0
    deep_probe_ncc_disagreement_lat_deg: float = 8.0
    deep_probe_ncc_min_deep_score: float = 0.55
    deep_probe_ncc_override_margin: float = 0.05
    deep_probe_recovery_max_lon_deg: float = 14.0
    deep_probe_recovery_max_lat_deg: float = 10.0
    # A global deep peak must remain close to the latest independent probe;
    # this blocks high-scoring background peaks after NCC loses scale.
    deep_relocalize_probe_max_lon_gap_deg: float = 30.0
    deep_relocalize_probe_max_lat_gap_deg: float = 20.0
    deep_ncc_jump_score_threshold: float = 0.70
    deep_ncc_max_jump_ratio: float = 0.65
    deep_ncc_growth_psr_threshold: float = 1.85
    deep_ncc_growth_min_init_ratio: float = 1.20
    deep_ncc_growth_max_scale_step: float = 1.35
    deep_ncc_growth_scale_pairs: tuple[tuple[float, float], ...] = (
        (1.35, 1.00), (1.60, 1.00), (1.90, 1.00), (2.20, 1.00),
    )
    deep_local_min_lon_jump_deg: float = 2.5
    deep_local_min_lat_jump_deg: float = 2.0
    deep_local_max_lon_jump_deg: float = 7.0
    deep_local_max_lat_jump_deg: float = 5.0
    deep_local_lon_jump_size_ratio: float = 0.90
    deep_local_lat_jump_size_ratio: float = 0.25
    deep_semantic_proposal_enabled: bool = True
    deep_semantic_history_size: int = 8
    deep_semantic_history_interval: int = 40
    deep_semantic_history_ncc_score: float = 0.55
    deep_semantic_trigger_ncc_score: float = 0.40
    deep_semantic_trigger_frames: int = 8
    deep_semantic_min_interval: int = 20
    deep_semantic_min_score: float = 0.58
    deep_semantic_topk: int = 3
    deep_semantic_nms_deg: float = 18.0
    deep_semantic_lon_stride_deg: float = 24.0
    deep_semantic_lat_stride_deg: float = 18.0
    deep_semantic_lat_limit_deg: float = 36.0
    deep_semantic_scale_pairs: tuple[tuple[float, float], ...] = (
        (0.8, 0.8), (1.0, 0.8), (1.0, 1.0),
        (1.25, 0.7), (1.5, 0.6), (1.5, 0.8),
        (2.0, 0.6), (2.0, 0.8), (2.5, 0.8),
    )
    deep_semantic_verify_score_margin: float = 0.08
    deep_semantic_verify_psr: float = 1.75
    deep_semantic_refine_offsets_deg: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    deep_semantic_motion_penalty: float = 0.10
    deep_semantic_batch_size: int = 32

    # --- 手工特征低置信保护 ---
    handcrafted_hold_position_score: float = 0.12
    handcrafted_center_score_margin: float = 0.02
    handcrafted_scale_update_confidence: float = 0.16
    handcrafted_high_confidence: float = 0.24
    handcrafted_occlusion_threshold: float = 0.08
    handcrafted_relocalize_confidence_threshold: float = 0.10
    handcrafted_update_quality_threshold: float = 0.20
    handcrafted_state_trust_low: float = 0.08
    handcrafted_state_trust_high: float = 0.24
    handcrafted_state_trust_min: float = 0.0
    handcrafted_flow_enabled: bool = True
    # OpenCV matchTemplate runs inside the NCC worker pool. Limit its own
    # thread fan-out to avoid multiplying CPU threads.
    handcrafted_cv_threads: int = 8
    handcrafted_flow_max_points: int = 100
    handcrafted_flow_min_points: int = 12
    handcrafted_flow_min_inlier_ratio: float = 0.65
    handcrafted_flow_max_fb_error: float = 0.5
    handcrafted_flow_max_spread: float = 2.5
    handcrafted_flow_padding: float = 0.15
    handcrafted_flow_template_score: float = 0.08
    handcrafted_color_enabled: bool = True
    handcrafted_color_hue_tolerance: float = 12.0
    handcrafted_color_min_saturation: int = 90
    handcrafted_color_min_value: int = 35
    handcrafted_color_min_area: int = 15
    handcrafted_color_search_width: int = 640
    handcrafted_color_search_height: int = 480
    handcrafted_color_max_scale_step: float = 1.7
    handcrafted_ncc_enabled: bool = True
    handcrafted_ncc_search_factor: float = 3.0
    handcrafted_ncc_max_jump_ratio: float = 0.5
    handcrafted_ncc_parallel_workers: int = 4
    handcrafted_ncc_update_rate: float = 0.10
    handcrafted_ncc_update_score: float = 0.55
    handcrafted_ncc_reliable_score: float = 0.35
    handcrafted_ncc_min_template_std: float = 3.0
    handcrafted_ncc_min_search_std: float = 3.0
    handcrafted_ncc_distance_penalty: float = 0.30
    handcrafted_ncc_scale_penalty: float = 0.05
    handcrafted_ncc_flow_min_points: int = 8
    handcrafted_ncc_flow_min_inlier_ratio: float = 0.20
    handcrafted_ncc_flow_max_fb_error: float = 1.5
    handcrafted_ncc_flow_max_spread: float = 3.0
    handcrafted_ncc_velocity_momentum: float = 0.50
    handcrafted_ncc_flow_motion_trigger: float = 0.06
    handcrafted_ncc_flow_score_trigger: float = 0.72
    handcrafted_ncc_flow_disagreement_ratio: float = 0.35
    handcrafted_ncc_flow_axis_disagreement_ratio: float = 0.20
    handcrafted_ncc_flow_disagreement_score: float = 0.55
    handcrafted_ncc_flow_grace_frames: int = 3
    handcrafted_ncc_flow_guard_min_size: float = 64.0
    handcrafted_ncc_flow_scale_tolerance: float = 0.10
    handcrafted_ncc_flow_scale_min_inlier_ratio: float = 0.50
    handcrafted_ncc_scale_anchor_score: float = 0.55
    handcrafted_ncc_scale_anchor_min_ratio: float = 0.65
    handcrafted_ncc_low_score_commit_threshold: float = 0.55
    handcrafted_ncc_relocalize_streak: int = 8
    handcrafted_ncc_max_scale_step: float = 1.12
    small_target_ncc_max_jump_ratio: float = 2.0

    # --- 小目标保护参数 ---
    small_target_threshold: float = 0.02
    small_target_template_enlarge: float = 6.0
    small_target_search_enlarge: float = 3.0
    small_target_min_template_pixels: float = 18.0
    small_target_max_scale_step: float = 1.08
    small_target_local_grid_radius: int = 4
    small_target_local_step_factor: float = 0.65


@dataclass
class TrackerRuntimeStats:
    frames: int = 0
    local_searches: int = 0
    relocalizations: int = 0
    relocalization_accepts: int = 0
    relocalization_psr_rejects: int = 0
    last_relocalization_psr: float = 0.0
    fallback_attempts: int = 0
    fallback_accepts: int = 0
    fallback_low_score_triggers: int = 0
    fallback_low_psr_triggers: int = 0
    fallback_confident_wrong_triggers: int = 0
    fallback_flow_results: int = 0
    fallback_ncc_results: int = 0
    fallback_color_results: int = 0
    fallback_local_search_results: int = 0
    ncc_flow_disagreement_rejects: int = 0
    ncc_flow_scale_adjustments: int = 0
    ncc_scale_anchor_adjustments: int = 0
    ncc_exact_matches: int = 0
    fallback_deep_score_sum: float = 0.0
    fallback_hand_score_sum: float = 0.0
    deep_probe_skips: int = 0
    deep_psr_samples: int = 0
    deep_psr_sum: float = 0.0
    deep_psr_min: float = float("inf")
    deep_psr_max: float = float("-inf")
    deep_psr_below_125: int = 0
    deep_psr_below_150: int = 0
    deep_psr_below_175: int = 0
    deep_psr_below_200: int = 0
    semantic_proposal_attempts: int = 0
    semantic_proposal_candidates: int = 0
    semantic_refine_attempts: int = 0
    semantic_refine_accepts: int = 0
    semantic_history_updates: int = 0
    last_semantic_score: float = 0.0
    last_semantic_refine_score: float = 0.0
    last_semantic_current_score: float = 0.0
    last_semantic_refine_psr: float = 0.0
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

    _cv_thread_lock = threading.Lock()
    _cv_thread_users = 0
    _cv_thread_previous: int | None = None

    def __init__(
        self,
        config: TrackerConfig | None = None,
        deep_extractor: Any = None,
        similarity_head: Any = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self._cv_threads_acquired = False
        self._acquire_cv_threads()
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
        self._last_relocalize_attempt_frame = -self.config.relocalize_min_interval
        self._last_deep_probe_frame = -max(self.config.deep_probe_interval, 1)
        self._consecutive_low_deep_probes = 0
        self._consecutive_relocalization_rejects = 0
        self._semantic_history: list[Any] = []
        self._semantic_size_anchor: SphereState | None = None
        self._last_semantic_history_frame = -max(self.config.deep_semantic_history_interval, 1)
        self._last_semantic_proposal_frame = -max(self.config.deep_semantic_min_interval, 1)
        self._consecutive_low_ncc = 0
        self._last_deep_probe_state: SphereState | None = None
        self._last_deep_probe_score: float = 0.0
        self.runtime_stats = TrackerRuntimeStats()
        self._debug_payload: dict[str, np.ndarray] | None = None
        self._last_state_trust = 1.0
        self._last_local_jump_gated = False
        self._last_center_score = 0.0
        self._last_best_minus_center = 0.0
        self._last_center_preferred = False
        self._last_deep_probe_state = None
        self._last_deep_probe_score = 0.0
        self._previous_frame_gray: np.ndarray | None = None
        self._last_flow_reliable = False
        self._last_flow_inlier_ratio = 0.0
        self._last_flow_fb_error = 0.0
        self._last_flow_spread = 0.0
        self._color_hue: float | None = None
        self._color_bbox: np.ndarray | None = None
        self._last_color_reliable = False
        self._ncc_initial_template: np.ndarray | None = None
        self._ncc_short_template: np.ndarray | None = None
        self._ncc_bbox: np.ndarray | None = None
        self._ncc_velocity = np.zeros(2, dtype=np.float64)
        self._ncc_flow_scale: np.ndarray | None = None
        self._ncc_scale_anchor: np.ndarray | None = None
        self._ncc_flow_was_reliable = False
        self._ncc_frames_since_flow = self.config.handcrafted_ncc_flow_grace_frames + 1
        self._ncc_last_score = 1.0
        self._last_ncc_reliable = False
        self._ncc_executor: ThreadPoolExecutor | None = None
        self._ncc_initial_resize_cache: dict[tuple[int, int], np.ndarray] = {}
        self._ncc_short_resize_cache: dict[tuple[int, int], np.ndarray] = {}
        self._ncc_flow_mask: np.ndarray | None = None
        # --- 手工模式独立状态（P1：不受深度漂移污染）---
        self._hand_state: SphereState | None = None
        self._hand_velocity = np.zeros(2, dtype=np.float32)
        self._debug_recorder: TrackerDebugRecorder | None = None
        if self.config.debug_dir:
            self._debug_recorder = TrackerDebugRecorder(
                debug_dir=self.config.debug_dir,
                start_frame=self.config.debug_start_frame,
                max_frames=self.config.debug_max_frames,
                frame_stride=self.config.debug_frame_stride,
                save_response_maps=self.config.debug_save_response_maps,
            )

    def close(self) -> None:
        executor = getattr(self, "_ncc_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._ncc_executor = None
        self._release_cv_threads()

    def __del__(self) -> None:
        self.close()

    def _acquire_cv_threads(self) -> None:
        if cv2 is None:
            return
        requested = int(self.config.handcrafted_cv_threads)
        if requested <= 0:
            return
        with self._cv_thread_lock:
            if self._cv_thread_users == 0:
                self._cv_thread_previous = int(cv2.getNumThreads())
                cv2.setNumThreads(requested)
            self._cv_thread_users += 1
            self._cv_threads_acquired = True

    def _release_cv_threads(self) -> None:
        if not getattr(self, "_cv_threads_acquired", False) or cv2 is None:
            return
        with self._cv_thread_lock:
            self._cv_thread_users = max(self._cv_thread_users - 1, 0)
            if self._cv_thread_users == 0 and self._cv_thread_previous is not None:
                cv2.setNumThreads(self._cv_thread_previous)
                self._cv_thread_previous = None
            self._cv_threads_acquired = False

    def reset_runtime_stats(self) -> None:
        forward_calls = getattr(self.deep_extractor, "forward_calls", 0)
        self.runtime_stats = TrackerRuntimeStats(deep_forward_calls_start=int(forward_calls))

    def get_runtime_stats(self) -> dict[str, float | int]:
        stats = self.runtime_stats
        current_forward_calls = int(getattr(self.deep_extractor, "forward_calls", 0))
        has_psr = stats.deep_psr_samples > 0
        return {
            "frames": stats.frames,
            "local_searches": stats.local_searches,
            "relocalizations": stats.relocalizations,
            "relocalization_accepts": stats.relocalization_accepts,
            "relocalization_psr_rejects": stats.relocalization_psr_rejects,
            "last_relocalization_psr": stats.last_relocalization_psr,
            "relocalization_interval": self._current_relocalization_interval(),
            "consecutive_relocalization_rejects": self._consecutive_relocalization_rejects,
            "fallback_attempts": stats.fallback_attempts,
            "fallback_accepts": stats.fallback_accepts,
            "fallback_low_score_triggers": stats.fallback_low_score_triggers,
            "fallback_low_psr_triggers": stats.fallback_low_psr_triggers,
            "fallback_confident_wrong_triggers": stats.fallback_confident_wrong_triggers,
            "fallback_flow_results": stats.fallback_flow_results,
            "fallback_ncc_results": stats.fallback_ncc_results,
            "fallback_color_results": stats.fallback_color_results,
            "fallback_local_search_results": stats.fallback_local_search_results,
            "ncc_flow_disagreement_rejects": stats.ncc_flow_disagreement_rejects,
            "ncc_flow_scale_adjustments": stats.ncc_flow_scale_adjustments,
            "ncc_scale_anchor_adjustments": stats.ncc_scale_anchor_adjustments,
            "ncc_exact_matches": stats.ncc_exact_matches,
            "fallback_deep_score_sum": stats.fallback_deep_score_sum,
            "fallback_hand_score_sum": stats.fallback_hand_score_sum,
            "deep_probe_skips": stats.deep_probe_skips,
            "deep_probe_interval": self._current_deep_probe_interval(),
            "consecutive_low_deep_probes": self._consecutive_low_deep_probes,
            "deep_psr_samples": stats.deep_psr_samples,
            "deep_psr_mean": stats.deep_psr_sum / stats.deep_psr_samples if has_psr else 0.0,
            "deep_psr_min": stats.deep_psr_min if has_psr else 0.0,
            "deep_psr_max": stats.deep_psr_max if has_psr else 0.0,
            "deep_psr_below_125": stats.deep_psr_below_125,
            "deep_psr_below_150": stats.deep_psr_below_150,
            "deep_psr_below_175": stats.deep_psr_below_175,
            "deep_psr_below_200": stats.deep_psr_below_200,
            "semantic_proposal_attempts": stats.semantic_proposal_attempts,
            "semantic_proposal_candidates": stats.semantic_proposal_candidates,
            "semantic_refine_attempts": stats.semantic_refine_attempts,
            "semantic_refine_accepts": stats.semantic_refine_accepts,
            "semantic_history_updates": stats.semantic_history_updates,
            "semantic_history_size": len(self._semantic_history),
            "last_semantic_score": stats.last_semantic_score,
            "last_semantic_refine_score": stats.last_semantic_refine_score,
            "last_semantic_current_score": stats.last_semantic_current_score,
            "last_semantic_refine_psr": stats.last_semantic_refine_psr,
            "template_updates": stats.template_updates,
            "response_maps": stats.response_maps,
            "response_batches": stats.response_batches,
            "last_score": stats.last_score,
            "last_peak": stats.last_peak,
            "last_psr": stats.last_psr,
            "last_apce": stats.last_apce,
            "deep_forward_calls": current_forward_calls - stats.deep_forward_calls_start,
        }

    def _high_confidence(self, handcrafted_result: bool = False) -> float:
        if self._deep_mode and not handcrafted_result:
            return self.config.deep_high_confidence
        return self.config.handcrafted_high_confidence

    def _occlusion_threshold(self, handcrafted_result: bool = False) -> float:
        if self._deep_mode and not handcrafted_result:
            return self.config.deep_occlusion_threshold
        return self.config.handcrafted_occlusion_threshold

    def _relocalize_confidence_threshold(self, handcrafted_result: bool = False) -> float:
        if self._deep_mode and not handcrafted_result:
            return self.config.deep_relocalize_confidence_threshold
        return self.config.handcrafted_relocalize_confidence_threshold

    def _relocalize_cooldown_elapsed(self) -> bool:
        return (
            self._frame_count - self._last_relocalize_attempt_frame
            >= self._current_relocalization_interval()
        )

    def _current_relocalization_interval(self) -> int:
        base_interval = max(int(self.config.relocalize_min_interval), 1)
        if not self._deep_mode:
            return base_interval
        backoff_after = max(int(self.config.deep_relocalize_backoff_after), 1)
        backoff_steps = self._consecutive_relocalization_rejects // backoff_after
        multiplier = max(float(self.config.deep_relocalize_backoff_multiplier), 1.0)
        interval = int(round(base_interval * multiplier ** backoff_steps))
        return min(interval, max(int(self.config.deep_relocalize_max_interval), base_interval))

    def _deep_probe_due(self) -> bool:
        interval = self._current_deep_probe_interval()
        return self._frame_count - self._last_deep_probe_frame >= interval

    def _current_deep_probe_interval(self) -> int:
        base_interval = max(int(self.config.deep_probe_interval), 1)
        backoff_after = max(int(self.config.deep_probe_backoff_after), 1)
        backoff_steps = self._consecutive_low_deep_probes // backoff_after
        multiplier = max(float(self.config.deep_probe_backoff_multiplier), 1.0)
        interval = int(round(base_interval * multiplier ** backoff_steps))
        return min(interval, max(int(self.config.deep_probe_max_interval), base_interval))

    def _record_deep_probe_psr(self, psr: float) -> None:
        if psr < self.config.deep_fallback_psr_threshold:
            self._consecutive_low_deep_probes += 1
        else:
            self._consecutive_low_deep_probes = 0

    def _accept_deep_fallback(
        self,
        hand_score: float,
        deep_score: float | None,
        source: str,
    ) -> bool:
        """Accept fallback only when it is independently plausible.

        Low deep PSR is evidence that deep tracking is uncertain, not evidence
        that any handcrafted result is correct.  NCC/flow/color have their own
        reliability checks; local search still needs a meaningful score.
        """
        if not np.isfinite(hand_score) or hand_score < self.config.deep_fallback_min_hand_score:
            return False
        if source in {"flow", "color"}:
            return True
        if deep_score is None:
            return True
        return hand_score >= deep_score + self.config.deep_fallback_score_margin

    def _handcrafted_confidence(self, score: float, source: str) -> float:
        """Keep NCC confidence tied to its raw score; fallback motion is heuristic."""
        if source == "ncc":
            return float(score)
        return float(self.config.handcrafted_high_confidence)

    def _use_deep_relocalization(self) -> bool:
        return bool(self._deep_mode)

    def _accept_relocalization_candidate(
        self,
        candidate_score: float,
        current_score: float,
        candidate_psr: float,
        large_jump_early: bool,
        candidate_is_deep: bool,
    ) -> bool:
        if large_jump_early:
            return False
        if candidate_score <= current_score + self.config.relocalize_score_margin:
            return False
        if candidate_is_deep and candidate_psr < self.config.deep_relocalize_accept_psr_threshold:
            return False
        return True

    def _allow_early_deep_relocalization_jump(
        self,
        candidate_score: float,
        current_score: float,
        candidate_psr: float,
    ) -> bool:
        """Allow an early large jump only with both score and peak evidence."""
        if not self._deep_mode:
            return False
        if self._frame_count <= self.config.relocalize_jump_gate_frames - 10:
            return False
        return (
            candidate_score >= self.config.relocalize_reset_score
            and candidate_score > current_score + max(
                self.config.relocalize_score_margin,
                0.20,
            )
            and candidate_psr >= self.config.deep_relocalize_accept_psr_threshold
        )

    def _ncc_scale_collapsed(self) -> bool:
        """Detect a long-lived NCC box shrinking away from initialization."""
        if not self._deep_mode or self._ncc_bbox is None:
            return False
        if self._init_bbox_width_px is None or self._init_bbox_height_px is None:
            return False
        ratio = max(float(self.config.deep_ncc_scale_collapse_ratio), 0.1)
        return bool(
            self._ncc_bbox[2] < self._init_bbox_width_px * ratio
            or self._ncc_bbox[3] < self._init_bbox_height_px * ratio
        )

    def _deep_probe_disagrees_with_ncc(
        self,
        hand_state: SphereState,
        source: str,
        hand_score: float | None = None,
    ) -> bool:
        if source != "ncc" or self._last_deep_probe_state is None:
            return False
        if self._last_deep_probe_score < self.config.deep_probe_ncc_min_deep_score:
            return False
        lon_gap = abs(math.degrees(lon_distance(
            hand_state.lon,
            self._last_deep_probe_state.lon,
        )))
        lat_gap = abs(math.degrees(
            hand_state.lat - self._last_deep_probe_state.lat,
        ))
        return bool(
            lon_gap >= self.config.deep_probe_ncc_disagreement_lon_deg
            or lat_gap >= self.config.deep_probe_ncc_disagreement_lat_deg
        ) and (
            hand_score is None
            or self._last_deep_probe_score
            >= hand_score + self.config.deep_probe_ncc_override_margin
        )

    def _recent_deep_probe_supports_jump(
        self,
        candidate: SphereState | None = None,
    ) -> bool:
        """Require recent independent evidence before accepting a large jump."""
        if self._last_deep_probe_state is None:
            return False
        max_age = max(int(self.config.deep_probe_interval) * 2, 1)
        if not (
            self._frame_count - self._last_deep_probe_frame <= max_age
            and self._last_deep_probe_score >= self.config.deep_probe_ncc_min_deep_score
        ):
            return False
        if candidate is None:
            return True
        lon_gap = abs(math.degrees(lon_distance(
            candidate.lon,
            self._last_deep_probe_state.lon,
        )))
        lat_gap = abs(math.degrees(
            candidate.lat - self._last_deep_probe_state.lat,
        ))
        return bool(
            lon_gap <= self.config.deep_relocalize_probe_max_lon_gap_deg
            and lat_gap <= self.config.deep_relocalize_probe_max_lat_gap_deg
        )

    def _relocalization_jump_is_gated(
        self,
        lon_jump_deg: float,
        lat_jump_deg: float,
    ) -> bool:
        jump_is_large = (
            lon_jump_deg > self.config.relocalize_max_lon_jump_deg
            or lat_jump_deg > self.config.relocalize_max_lat_jump_deg
        )
        if not jump_is_large:
            return False
        enough_loss = max(
            self.lost_frames,
            self._consecutive_low_ncc,
        ) >= self.config.relocalize_jump_gate_lost_frames
        return self._frame_count <= self.config.relocalize_jump_gate_frames or not enough_loss

    def _accept_handcrafted_fallback(self, hand_state: SphereState) -> None:
        if self._hand_state is None:
            return
        hand_lon_delta = lon_distance(hand_state.lon, self._hand_state.lon)
        hand_lat_delta = hand_state.lat - self._hand_state.lat
        momentum = self.config.motion_momentum
        self._hand_velocity[0] = momentum * self._hand_velocity[0] + (1.0 - momentum) * hand_lon_delta
        self._hand_velocity[1] = momentum * self._hand_velocity[1] + (1.0 - momentum) * hand_lat_delta
        self._hand_state = SphereState(
            lon=hand_state.lon,
            lat=hand_state.lat,
            equatorial_width=hand_state.equatorial_width,
            angular_height=hand_state.angular_height,
        )

    def _semantic_descriptor(self, feature: Any) -> Any:
        torch = self.deep_extractor._torch
        descriptor = feature.float().mean(dim=(2, 3))
        return torch.nn.functional.normalize(descriptor, p=2, dim=1, eps=1e-6)

    def _semantic_proposal_due(self) -> bool:
        return (
            self._deep_mode
            and self.config.deep_semantic_proposal_enabled
            and bool(self._semantic_history)
            and self._semantic_size_anchor is not None
            and self._consecutive_low_ncc >= max(int(self.config.deep_semantic_trigger_frames), 1)
            and self._frame_count - self._last_semantic_proposal_frame
            >= max(int(self.config.deep_semantic_min_interval), 1)
        )

    def _record_semantic_history(self, frame: np.ndarray, state: SphereState) -> None:
        if not self._deep_mode or not self.config.deep_semantic_proposal_enabled:
            return
        if (
            self._semantic_size_anchor is None
            or state.equatorial_width * state.angular_height
            >= self._semantic_size_anchor.equatorial_width * self._semantic_size_anchor.angular_height
        ):
            self._semantic_size_anchor = SphereState(
                state.lon, state.lat, state.equatorial_width, state.angular_height,
            )
        interval = max(int(self.config.deep_semantic_history_interval), 1)
        if self._semantic_history and self._frame_count - self._last_semantic_history_frame < interval:
            return
        feature = self._extract_template_feat(frame, state)
        self._semantic_history.append(self._semantic_descriptor(feature).detach())
        max_history = max(int(self.config.deep_semantic_history_size), 1)
        self._semantic_history = self._semantic_history[-max_history:]
        self._last_semantic_history_frame = self._frame_count
        self.runtime_stats.semantic_history_updates += 1

    def _semantic_score_features(self, features: Any) -> Any:
        history = self.deep_extractor._torch.cat(self._semantic_history, dim=0)
        return (self._semantic_descriptor(features) @ history.T).mean(dim=1)

    def _select_semantic_topk(
        self,
        states: list[SphereState],
        scores: Any,
    ) -> list[tuple[SphereState, float]]:
        ranked = sorted(
            zip(states, (float(score) for score in scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        selected: list[tuple[SphereState, float]] = []
        min_distance = math.radians(max(float(self.config.deep_semantic_nms_deg), 0.0))
        for state, score in ranked:
            if score < self.config.deep_semantic_min_score:
                break
            if any(
                math.hypot(
                    lon_distance(state.lon, existing.lon) * math.cos(existing.lat),
                    state.lat - existing.lat,
                ) < min_distance
                for existing, _ in selected
            ):
                continue
            selected.append((state, score))
            if len(selected) >= max(int(self.config.deep_semantic_topk), 1):
                break
        return selected

    def _semantic_proposals(self, frame: np.ndarray) -> list[tuple[SphereState, float]]:
        if self._semantic_size_anchor is None or not self._semantic_history:
            return []
        lon_values = np.deg2rad(np.arange(
            -180.0, 180.0,
            max(float(self.config.deep_semantic_lon_stride_deg), 1.0),
            dtype=np.float32,
        ))
        lat_limit = min(max(float(self.config.deep_semantic_lat_limit_deg), 0.0), 80.0)
        lat_values = np.deg2rad(np.arange(
            -lat_limit, lat_limit + 1e-6,
            max(float(self.config.deep_semantic_lat_stride_deg), 1.0),
            dtype=np.float32,
        ))
        states: list[SphereState] = []
        patches: list[np.ndarray] = []
        for width_scale, height_scale in self.config.deep_semantic_scale_pairs:
            width, height = self._clamp_target_size(
                self._semantic_size_anchor.equatorial_width * width_scale,
                self._semantic_size_anchor.angular_height * height_scale,
            )
            size_state = SphereState(0.0, 0.0, width, height)
            fov_x, fov_y = state_size_to_fov(
                size_state, enlarge=self.config.deep_search_enlarge,
            )
            for lon in lon_values:
                for lat in lat_values:
                    state = SphereState(float(lon), float(lat), width, height)
                    states.append(state)
                    patches.append(self._extract_search_patch(
                        frame, state.lon, state.lat, fov_x, fov_y, refine=False,
                    ))
        if not patches:
            return []
        features = self.deep_extractor.extract_search_features_batch(
            patches,
            refine=False,
            chunk_size=max(int(self.config.deep_semantic_batch_size), 1),
            assume_normalized=True,
        )
        return self._select_semantic_topk(states, self._semantic_score_features(features))

    def _refine_semantic_proposals(
        self,
        frame: np.ndarray,
        proposals: list[SphereState],
        current: SphereState,
    ) -> list[tuple[SphereState, float, float, float, float]]:
        if not proposals:
            return []
        refine_states: list[SphereState] = []
        refine_patches: list[np.ndarray] = []
        proposal_ranges: list[tuple[int, int]] = []
        for proposal in proposals:
            start = len(refine_states)
            fov_x, fov_y = state_size_to_fov(
                proposal, enlarge=self.config.deep_search_enlarge,
            )
            for lon_offset_deg in self.config.deep_semantic_refine_offsets_deg:
                for lat_offset_deg in self.config.deep_semantic_refine_offsets_deg:
                    state = SphereState(
                        lon=float(wrap_lon(proposal.lon + math.radians(lon_offset_deg))),
                        lat=float(clamp_lat(proposal.lat + math.radians(lat_offset_deg))),
                        equatorial_width=proposal.equatorial_width,
                        angular_height=proposal.angular_height,
                    )
                    refine_states.append(state)
                    refine_patches.append(self._extract_search_patch(
                        frame, state.lon, state.lat, fov_x, fov_y, refine=False,
                    ))
            proposal_ranges.append((start, len(refine_states)))
        features = self.deep_extractor.extract_search_features_batch(
            refine_patches,
            refine=False,
            chunk_size=max(int(self.config.deep_semantic_batch_size), 1),
            assume_normalized=True,
        )
        semantic_scores = self._semantic_score_features(features)
        selected_indices = [
            start + int(semantic_scores[start:end].argmax())
            for start, end in proposal_ranges
        ]
        selected_features = features[selected_indices]
        correlation_scores = self._reloc_scores_deep_features(selected_features)
        results = []
        motion_penalty = max(float(self.config.deep_semantic_motion_penalty), 0.0)
        for index, (score, psr) in zip(selected_indices, correlation_scores):
            refined = refine_states[index]
            distance = math.hypot(
                lon_distance(refined.lon, current.lon) * math.cos(current.lat),
                refined.lat - current.lat,
            )
            ranked_score = float(score) - motion_penalty * distance
            results.append((
                refined,
                float(semantic_scores[index]),
                float(score),
                float(psr),
                ranked_score,
            ))
        return results

    def _refine_semantic_proposal(
        self,
        frame: np.ndarray,
        proposal: SphereState,
        current: SphereState,
    ) -> tuple[SphereState, float, float, float, float]:
        return self._refine_semantic_proposals(frame, [proposal], current)[0]

    def _verify_semantic_proposals(
        self,
        frame: np.ndarray,
        current: SphereState,
        current_score: float,
        proposals: list[tuple[SphereState, float]],
    ) -> tuple[SphereState, float, float] | None:
        if not proposals:
            return None
        saved_quality = (
            self.runtime_stats.last_score,
            self.runtime_stats.last_peak,
            self.runtime_stats.last_psr,
            self.runtime_stats.last_apce,
        )
        current_refined, current_refine_score = self._local_search_deep(frame, current)
        current_refine_score = max(float(current_refine_score), float(current_score))
        current_ranked_score = current_refine_score
        best_state = current_refined
        best_score = current_refine_score
        best_ranked_score = current_ranked_score
        best_psr = float(self.runtime_stats.last_psr)
        best_semantic_score = 0.0
        best_quality = (
            self.runtime_stats.last_score,
            self.runtime_stats.last_peak,
            self.runtime_stats.last_psr,
            self.runtime_stats.last_apce,
        )
        refined_proposals = self._refine_semantic_proposals(
            frame, [proposal for proposal, _ in proposals], current,
        )
        self.runtime_stats.semantic_refine_attempts += len(refined_proposals)
        for refined_result in refined_proposals:
            refined, refined_semantic_score, score, psr, ranked_score = (
                refined_result
            )
            if ranked_score > best_ranked_score and psr >= self.config.deep_semantic_verify_psr:
                best_state = SphereState(
                    refined.lon,
                    refined.lat,
                    current.equatorial_width,
                    current.angular_height,
                )
                best_score = float(score)
                best_ranked_score = ranked_score
                best_psr = psr
                best_semantic_score = refined_semantic_score
                best_quality = (
                    float(score),
                    saved_quality[1],
                    float(psr),
                    saved_quality[3],
                )

        self.runtime_stats.last_semantic_current_score = current_refine_score
        self.runtime_stats.last_semantic_refine_score = best_score
        self.runtime_stats.last_semantic_refine_psr = best_psr
        accepted = (
            best_semantic_score > 0.0
            and best_ranked_score
            >= current_ranked_score + self.config.deep_semantic_verify_score_margin
        )
        if not accepted:
            (
                self.runtime_stats.last_score,
                self.runtime_stats.last_peak,
                self.runtime_stats.last_psr,
                self.runtime_stats.last_apce,
            ) = saved_quality
            return None
        (
            self.runtime_stats.last_score,
            self.runtime_stats.last_peak,
            self.runtime_stats.last_psr,
            self.runtime_stats.last_apce,
        ) = best_quality
        self.runtime_stats.last_semantic_score = best_semantic_score
        return best_state, best_score, best_psr

    def _sync_semantic_recovery(self, frame: np.ndarray, state: SphereState) -> None:
        self.velocity[:] = 0.0
        self._hand_velocity[:] = 0.0
        self._hand_state = state
        if self.frame_shape is None:
            return
        bbox = self._state_to_output_bbox(state, self.frame_shape[1], self.frame_shape[0])
        self._ncc_bbox = bbox.astype(np.float32)
        self._color_bbox = bbox.astype(np.float32)

    def _update_quality_threshold(self, handcrafted_result: bool = False) -> float:
        if self._deep_mode and not handcrafted_result:
            return self.config.deep_update_quality_threshold
        return self.config.handcrafted_update_quality_threshold

    def _template_confirmation_frames(self) -> int:
        if self._deep_mode:
            return self.config.deep_confirmation_frames
        return self.config.confirmation_frames

    def _template_update_rate(self, template_type: str, is_best: bool) -> float:
        if template_type == "init":
            return 0.0
        if self._deep_mode:
            rate = (
                self.config.deep_template_update_ema
                if is_best
                else self.config.deep_template_update_background
            )
            return rate

        rate = self.config.template_update_ema if is_best else self.config.template_update_background
        return rate

    def _freeze_scale_update_confidence(self, handcrafted_result: bool = False) -> float:
        if self._deep_mode and not handcrafted_result:
            return self.config.deep_scale_update_confidence
        return self.config.handcrafted_scale_update_confidence

    def _freeze_state_scale(self, state: SphereState) -> SphereState:
        if self.state is None:
            return state
        return SphereState(
            lon=state.lon,
            lat=state.lat,
            equatorial_width=self.state.equatorial_width,
            angular_height=self.state.angular_height,
        )

    def _state_trust(self, score: float, handcrafted_result: bool = False) -> float:
        if self._deep_mode and not handcrafted_result:
            low_cfg = self.config.deep_state_trust_low
            high_cfg = self.config.deep_state_trust_high
            min_trust = self.config.deep_state_trust_min
        else:
            low_cfg = self.config.handcrafted_state_trust_low
            high_cfg = self.config.handcrafted_state_trust_high
            min_trust = self.config.handcrafted_state_trust_min

        low = min(low_cfg, high_cfg - 1e-4)
        high = max(high_cfg, low + 1e-4)
        if score <= low:
            return min_trust
        if score >= high:
            return 1.0

        alpha = (score - low) / (high - low)
        trust = min_trust + alpha * (1.0 - min_trust)

        if not self._deep_mode or handcrafted_result:
            return float(trust)

        psr = max(float(self.runtime_stats.last_psr), 0.0)
        psr_scale = max(float(self.config.deep_state_trust_psr_scale), 1e-4)
        psr_factor = 1.0 / (1.0 + math.exp(-(psr - self.config.deep_state_trust_psr_center) / psr_scale))
        psr_factor = 0.25 + 0.75 * psr_factor
        return trust * psr_factor

    def _blend_state_position(
        self,
        anchor: SphereState,
        candidate: SphereState,
        trust: float,
    ) -> SphereState:
        trust = float(np.clip(trust, 0.0, 1.0))
        lon = wrap_lon(anchor.lon + trust * lon_distance(candidate.lon, anchor.lon))
        lat = clamp_lat(anchor.lat + trust * (candidate.lat - anchor.lat))
        return SphereState(
            lon=float(lon),
            lat=float(lat),
            equatorial_width=candidate.equatorial_width,
            angular_height=candidate.angular_height,
        )

    def _guard_low_confidence_state(
        self,
        predicted: SphereState,
        candidate: SphereState,
        score: float,
        handcrafted_result: bool = False,
    ) -> SphereState:
        self._last_state_trust = 1.0
        self._last_local_jump_gated = False
        trust = self._state_trust(score, handcrafted_result)
        self._last_state_trust = trust
        if not self._deep_mode or handcrafted_result:
            if trust >= 0.999:
                return candidate
            if self.lost_frames >= 2 and not self._last_flow_reliable and not self._last_color_reliable:
                trust = min(trust, 0.20)
                self._last_state_trust = trust
            return self._blend_state_position(predicted, candidate, trust)

        if score >= self.config.deep_local_jump_gate_confidence and trust >= 0.999:
            return candidate

        angular_width = candidate.equatorial_width / max(math.cos(candidate.lat), 1e-3)
        max_lon_jump = max(
            math.radians(self.config.deep_local_min_lon_jump_deg),
            angular_width * self.config.deep_local_lon_jump_size_ratio,
        )
        max_lat_jump = max(
            math.radians(self.config.deep_local_min_lat_jump_deg),
            candidate.angular_height * self.config.deep_local_lat_jump_size_ratio,
        )
        max_lon_jump = min(max_lon_jump, math.radians(self.config.deep_local_max_lon_jump_deg))
        max_lat_jump = min(max_lat_jump, math.radians(self.config.deep_local_max_lat_jump_deg))

        lon_jump = lon_distance(candidate.lon, predicted.lon)
        lat_jump = candidate.lat - predicted.lat
        clipped_lon_jump = float(np.clip(lon_jump, -max_lon_jump, max_lon_jump))
        clipped_lat_jump = float(np.clip(lat_jump, -max_lat_jump, max_lat_jump))
        gated = (
            abs(clipped_lon_jump - lon_jump) > 1e-6
            or abs(clipped_lat_jump - lat_jump) > 1e-6
        )
        if gated:
            candidate = SphereState(
                lon=float(wrap_lon(predicted.lon + clipped_lon_jump)),
                lat=float(clamp_lat(predicted.lat + clipped_lat_jump)),
                equatorial_width=candidate.equatorial_width,
                angular_height=candidate.angular_height,
            )

        self._last_local_jump_gated = gated
        return self._blend_state_position(predicted, candidate, trust)

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
        self._hand_velocity[:] = 0.0
        self._hand_state = erp_bbox_to_state(init_bbox_xywh, w, h)
        frame_gray = self._handcrafted_gray(frame, assume_normalized=True)
        self._previous_frame_gray = frame_gray if self.config.handcrafted_flow_enabled else None
        self._last_flow_reliable = False
        self._initialize_color_model(frame, init_bbox_xywh)
        self._initialize_ncc_model(frame, init_bbox_xywh, gray=frame_gray)

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
        self._last_relocalize_attempt_frame = -self.config.relocalize_min_interval
        self._last_deep_probe_frame = -max(self.config.deep_probe_interval, 1)
        self._consecutive_low_deep_probes = 0
        self._consecutive_relocalization_rejects = 0
        self._semantic_history.clear()
        self._semantic_size_anchor = None
        self._last_semantic_history_frame = -max(self.config.deep_semantic_history_interval, 1)
        self._last_semantic_proposal_frame = -max(self.config.deep_semantic_min_interval, 1)
        self._consecutive_low_ncc = 0
        self._last_deep_probe_state = None
        self._last_deep_probe_score = 0.0
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
        self._record_semantic_history(frame, self.state)
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
        offset_ratio = 0.08  # 恢复 P3 的深度模式模板偏移，提供搜索多样性
        step_lon = self.state.equatorial_width * offset_ratio
        step_lat = self.state.angular_height * offset_ratio
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

        # Very small targets are especially sensitive to one-frame scale noise.
        # Limit the step around the previous state while keeping the original
        # initialization bounds as the long-term safety envelope.
        if self.state is not None and self._is_small_target(self.state):
            max_step = max(float(self.config.small_target_max_scale_step), 1.01)
            prev_width = max(float(self.state.equatorial_width), 1e-6)
            prev_height = max(float(self.state.angular_height), 1e-6)
            width = float(np.clip(width, prev_width / max_step, prev_width * max_step))
            height = float(np.clip(height, prev_height / max_step, prev_height * max_step))
        return width, height

    def _is_small_target(self, state: SphereState) -> bool:
        if self.frame_shape is None:
            return False
        h, w = self.frame_shape
        area = state.equatorial_width * state.angular_height * w * h / (2.0 * math.pi)
        return area < self.config.small_target_threshold * w * h

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
        self._last_state_trust = 1.0
        self._last_local_jump_gated = False
        self._last_center_score = 0.0
        self._last_best_minus_center = 0.0
        self._last_center_preferred = False
        h, w = self.frame_shape
        predicted = SphereState(
            lon=float(wrap_lon(self.state.lon + self.velocity[0])),
            lat=float(clamp_lat(self.state.lat + self.velocity[1])),
            equatorial_width=self.state.equatorial_width,
            angular_height=self.state.angular_height,
        )

        flow_reliable = False
        color_reliable = False
        ncc_reliable = False
        handcrafted_result = False
        fallback_source = ""
        semantic_recovery = False
        frame_gray: np.ndarray | None = None
        if not self._deep_mode:
            if self._color_hue is None:
                frame_gray = self._handcrafted_gray(frame, assume_normalized=True)
                ncc_state, ncc_score, ncc_reliable = self._predict_with_ncc(frame, gray=frame_gray)
                if ncc_reliable:
                    best_state = ncc_state
                    best_score = self._handcrafted_confidence(ncc_score, "ncc")
                else:
                    best_state, best_score = self._local_search(frame, predicted)
            else:
                frame_gray = self._handcrafted_gray(frame, assume_normalized=True)
                flow_state, flow_reliable = self._predict_with_optical_flow(
                    frame, self.state, current_gray=frame_gray,
                )
                if flow_reliable:
                    flow_score = self._score_state_handcrafted(frame, flow_state)
                    best_state = flow_state
                    best_score = max(flow_score, self.config.handcrafted_high_confidence)
                else:
                    color_state, color_reliable = self._predict_with_color(frame, predicted)
                    if color_reliable:
                        best_state = color_state
                        best_score = self.config.handcrafted_high_confidence
                    else:
                        best_state, best_score = self._local_search(frame, predicted)
        else:
            last_probe_low = (
                self.runtime_stats.deep_psr_samples > 0
                and self.runtime_stats.last_psr < self.config.deep_fallback_psr_threshold
            )
            skip_deep_probe = last_probe_low and not self._deep_probe_due()
            if skip_deep_probe:
                self.runtime_stats.deep_probe_skips += 1
                self.runtime_stats.fallback_attempts += 1
                self.runtime_stats.fallback_low_psr_triggers += 1
                frame_gray = self._handcrafted_gray(frame, assume_normalized=True)
                hand_state, hand_score, fallback_source = self._try_handcrafted_fallback(
                    frame, predicted, gray=frame_gray,
                )
                flow_reliable = fallback_source == "flow"
                self.runtime_stats.fallback_hand_score_sum += float(hand_score)
                if self._accept_deep_fallback(hand_score, None, fallback_source):
                    self.runtime_stats.fallback_accepts += 1
                    best_state, best_score = hand_state, hand_score
                    handcrafted_result = True
                else:
                    # Do not commit an untrusted fallback, but keep the frame
                    # in the normal loss/relocalization state machine.
                    best_state = predicted
                    best_score = 0.0
            else:
                self._last_deep_probe_frame = self._frame_count
                best_state, best_score = self._local_search(frame, predicted)
                self._last_deep_probe_state = best_state
                self._last_deep_probe_score = float(best_score)
                deep_psr = float(self.runtime_stats.last_psr)
                self._record_deep_probe_psr(deep_psr)
                self.runtime_stats.deep_psr_samples += 1
                self.runtime_stats.deep_psr_sum += deep_psr
                self.runtime_stats.deep_psr_min = min(self.runtime_stats.deep_psr_min, deep_psr)
                self.runtime_stats.deep_psr_max = max(self.runtime_stats.deep_psr_max, deep_psr)
                self.runtime_stats.deep_psr_below_125 += int(deep_psr < 1.25)
                self.runtime_stats.deep_psr_below_150 += int(deep_psr < 1.50)
                self.runtime_stats.deep_psr_below_175 += int(deep_psr < 1.75)
                self.runtime_stats.deep_psr_below_200 += int(deep_psr < 2.00)
                deep_score_before_fallback = best_score
                psr_low = self.runtime_stats.last_psr < self.config.deep_fallback_psr_threshold
                hand_model_warm = self._frame_count > 15
                confident_wrong = best_score > self.config.deep_high_confidence and psr_low and hand_model_warm
                low_score_fallback = best_score < self.config.deep_occlusion_threshold
                need_fallback = low_score_fallback or psr_low
                if need_fallback:
                    self.runtime_stats.fallback_attempts += 1
                    self.runtime_stats.fallback_low_score_triggers += int(low_score_fallback)
                    self.runtime_stats.fallback_low_psr_triggers += int(psr_low)
                    self.runtime_stats.fallback_confident_wrong_triggers += int(confident_wrong)
                    self.runtime_stats.fallback_deep_score_sum += float(deep_score_before_fallback)
                    frame_gray = self._handcrafted_gray(frame, assume_normalized=True)
                    hand_state, hand_score, fallback_source = self._try_handcrafted_fallback(
                        frame, predicted, gray=frame_gray,
                    )
                    flow_reliable = fallback_source == "flow"
                    self.runtime_stats.fallback_hand_score_sum += float(hand_score)
                    probe_disagrees = self._deep_probe_disagrees_with_ncc(
                        hand_state,
                        fallback_source,
                        hand_score=hand_score,
                    )
                    if probe_disagrees:
                        best_state, best_score = (
                            self._last_deep_probe_state,
                            self._last_deep_probe_score,
                        )
                        handcrafted_result = False
                    elif self._accept_deep_fallback(
                        hand_score,
                        deep_score_before_fallback,
                        fallback_source,
                    ):
                        self.runtime_stats.fallback_accepts += 1
                        best_state, best_score = hand_state, hand_score
                        handcrafted_result = True
                    else:
                        # If the handcrafted fallback is weak, prefer a
                        # recent, independently-scored deep probe that is
                        # still near the current prediction.  Holding the
                        # prediction here used to let a bad NCC/colour frame
                        # poison the motion state for many subsequent frames.
                        probe = self._last_deep_probe_state
                        probe_age = self._frame_count - self._last_deep_probe_frame
                        if (
                            probe is not None
                            and probe_age <= max(int(self.config.deep_probe_interval) * 2, 1)
                            and self._last_deep_probe_score >= self.config.deep_probe_ncc_min_deep_score
                            and abs(math.degrees(lon_distance(probe.lon, predicted.lon)))
                            <= self.config.deep_probe_recovery_max_lon_deg
                            and abs(math.degrees(probe.lat - predicted.lat))
                            <= self.config.deep_probe_recovery_max_lat_deg
                        ):
                            best_state = probe
                            best_score = self._last_deep_probe_score
                        else:
                            # Keep the prior motion prediction and let the
                            # loss/relocalization state machine handle it.
                            best_state = predicted
                            best_score = deep_score_before_fallback
        if self._deep_mode and handcrafted_result and fallback_source == "ncc":
            if self._ncc_last_score < self.config.handcrafted_ncc_low_score_commit_threshold:
                self._consecutive_low_ncc += 1
            else:
                self._consecutive_low_ncc = 0
            if self._ncc_last_score >= self.config.deep_semantic_history_ncc_score:
                self._record_semantic_history(frame, best_state)
        elif self._deep_mode and not handcrafted_result:
            self._consecutive_low_ncc = 0

        if self._semantic_proposal_due():
            self.runtime_stats.semantic_proposal_attempts += 1
            self._last_semantic_proposal_frame = self._frame_count
            proposals = self._semantic_proposals(frame)
            self.runtime_stats.semantic_proposal_candidates += len(proposals)
            semantic_result = self._verify_semantic_proposals(
                frame, best_state, best_score, proposals,
            )
            if semantic_result is not None:
                best_state, best_score, semantic_psr = semantic_result
                self.runtime_stats.semantic_refine_accepts += 1
                self.runtime_stats.last_psr = semantic_psr
                handcrafted_result = False
                semantic_recovery = True
                self._consecutive_low_ncc = 0
        ncc_low_confidence = (
            self._deep_mode
            and handcrafted_result
            and fallback_source == "ncc"
            and best_score < self.config.handcrafted_ncc_low_score_commit_threshold
        )
        ncc_relocalize_due = (
            ncc_low_confidence
            and self._consecutive_low_ncc
            >= max(int(self.config.handcrafted_ncc_relocalize_streak), 1)
        )
        candidate_trust = self._state_trust(best_score, handcrafted_result)
        deep_low_quality = (
            self._deep_mode
            and not handcrafted_result
            and not semantic_recovery
            and (
                candidate_trust < self.config.deep_velocity_low_trust_threshold
                or self.runtime_stats.last_psr < self.config.deep_velocity_low_psr_threshold
            )
        )
        handcrafted_low_quality = (
            not self._deep_mode
            and candidate_trust < 0.35
        )

        score_drop = self._last_high_conf_score - best_score
        is_abnormal = score_drop > self.config.max_score_drop and self._last_high_conf_score > 0.0

        high_confidence = self._high_confidence(handcrafted_result)
        occlusion_threshold = self._occlusion_threshold(handcrafted_result)
        relocalize_threshold = self._relocalize_confidence_threshold(handcrafted_result)
        update_quality_threshold = self._update_quality_threshold(handcrafted_result)

        if deep_low_quality:
            self._occlusion_frames += 1
        elif best_score >= high_confidence:
            self._occlusion_frames = 0
            self._last_high_conf_score = best_score
        elif best_score < occlusion_threshold or is_abnormal:
            self._occlusion_frames += 1
        else:
            self._occlusion_frames = max(0, self._occlusion_frames - 1)

        should_relocalize = False
        relocalize_start_frame = max(
            self.config.relocalize_start_frame,
            self.config.relocalize_min_start_frame_override,
        )
        relocalize_lost_trigger = max(
            self.config.relocalize_lost_trigger,
            self.config.relocalize_min_lost_frames_override,
        )
        if self._deep_mode:
            if not handcrafted_result:
                relocalize_start_frame = min(relocalize_start_frame, self.config.deep_relocalize_min_start_frame)
                relocalize_lost_trigger = min(relocalize_lost_trigger, self.config.deep_relocalize_min_lost_frames)
        if self._frame_count >= relocalize_start_frame:
            if best_score < relocalize_threshold or ncc_relocalize_due:
                if self._relocalize_cooldown_elapsed():
                    should_relocalize = True

            # P4: 连续丢失帧数过多时强制触发重定位
            if not should_relocalize and self.lost_frames >= relocalize_lost_trigger:
                if self._relocalize_cooldown_elapsed():
                    should_relocalize = True

            if (
                self._deep_mode
                and not handcrafted_result
                and not should_relocalize
                and self.runtime_stats.last_psr < self.config.deep_relocalize_psr_threshold
                and max(self.lost_frames, self._occlusion_frames)
                >= self.config.deep_relocalize_min_lost_frames
                and self._relocalize_cooldown_elapsed()
            ):
                should_relocalize = True

            if (
                not should_relocalize
                and fallback_source == "ncc"
                and self._ncc_scale_collapsed()
                and self._relocalize_cooldown_elapsed()
            ):
                should_relocalize = True



        relocalize_applied = semantic_recovery
        if should_relocalize:
            self.runtime_stats.relocalizations += 1
            self._last_relocalize_attempt_frame = self._frame_count
            # Deep mode must keep relocalization in the same feature space even
            # when the current frame arrived through the NCC fallback branch.
            deep_relocalization = self._use_deep_relocalization()
            relocalized, relocalized_score, relocalized_psr = self._global_relocalize(
                frame,
                predicted,
                use_deep=deep_relocalization,
            )
            self.runtime_stats.last_relocalization_psr = (
                float(relocalized_psr) if deep_relocalization else 0.0
            )
            lon_jump_deg = abs(math.degrees(lon_distance(relocalized.lon, predicted.lon)))
            lat_jump_deg = abs(math.degrees(relocalized.lat - predicted.lat))
            large_jump_early = self._relocalization_jump_is_gated(
                lon_jump_deg, lat_jump_deg,
            )
            if (
                large_jump_early
                and self._allow_early_deep_relocalization_jump(
                relocalized_score,
                best_score,
                relocalized_psr,
                )
                and (
                    not ncc_relocalize_due
                    or self._recent_deep_probe_supports_jump(relocalized)
                )
            ):
                large_jump_early = False
            accept_relocalization = self._accept_relocalization_candidate(
                relocalized_score,
                best_score,
                relocalized_psr,
                large_jump_early,
                deep_relocalization,
            )
            # A global deep peak can be very sharp on a distractor.  For a
            # large jump, require agreement with a recent local deep probe
            # regardless of which loss trigger requested re-localization.
            # Previously this guard only ran for NCC-triggered recovery, so a
            # low-score/low-PSR trigger could still jump to background.
            if (
                accept_relocalization
                and deep_relocalization
                and (lon_jump_deg > self.config.relocalize_max_lon_jump_deg
                     or lat_jump_deg > self.config.relocalize_max_lat_jump_deg)
                and not self._recent_deep_probe_supports_jump(relocalized)
            ):
                accept_relocalization = False
            # A weak NCC streak is not an independent confirmation for a
            # global deep peak. Without a recent strong probe, keep the local
            # track rather than replacing it with a plausible distractor.
            if (
                accept_relocalization
                and ncc_relocalize_due
                and not self._recent_deep_probe_supports_jump(relocalized)
            ):
                accept_relocalization = False
            if (
                deep_relocalization
                and relocalized_score > best_score + self.config.relocalize_score_margin
                and relocalized_psr < self.config.deep_relocalize_accept_psr_threshold
            ):
                self.runtime_stats.relocalization_psr_rejects += 1
            if accept_relocalization:
                self.runtime_stats.relocalization_accepts += 1
                self._consecutive_relocalization_rejects = 0
                best_state, best_score = relocalized, relocalized_score
                handcrafted_result = not deep_relocalization
                self._last_relocalize_frame = self._frame_count
                relocalize_applied = True
                ncc_low_confidence = False

                # P4: 高置信重定位成功后重置模板
                if (
                    self.config.relocalize_reset_on_success
                    and relocalized_score >= self.config.relocalize_reset_score
                    and (not self._deep_mode or deep_relocalization)
                ):
                    self._reset_templates(frame, best_state)
            else:
                self._consecutive_relocalization_rejects += 1

        if relocalize_applied:
            high_confidence = self._high_confidence(handcrafted_result)
            update_quality_threshold = self._update_quality_threshold(handcrafted_result)

        best_state = self._clamp_state_size(best_state)
        if not semantic_recovery and best_score < self._freeze_scale_update_confidence(handcrafted_result):
            best_state = self._freeze_state_scale(best_state)
        if semantic_recovery:
            self._last_state_trust = 1.0
            self._last_local_jump_gated = False
        else:
            best_state = self._guard_low_confidence_state(
                predicted, best_state, best_score, handcrafted_result,
            )
        # Position gating may blend a weak candidate, but it must not carry
        # an implausible scale into the next frame. Re-apply the scale freeze
        # after all position guards so low-confidence deep probes cannot
        # enlarge the tracking state.
        if not semantic_recovery and best_score < self._freeze_scale_update_confidence(handcrafted_result):
            best_state = self._freeze_state_scale(best_state)
        lon_delta = lon_distance(best_state.lon, self.state.lon)
        lat_delta = best_state.lat - self.state.lat
        momentum = (
            self.config.motion_momentum
            if handcrafted_result else self.config.deep_motion_momentum
        ) if self._deep_mode else self.config.motion_momentum

        if self._occlusion_frames > self.config.occlusion_suppress_frames:
            momentum = min(momentum, 0.1)

        if semantic_recovery:
            self.velocity[:] = 0.0
        elif deep_low_quality:
            decay = float(np.clip(self.config.deep_velocity_decay, 0.0, 1.0))
            self.velocity[0] = decay * self.velocity[0]
            self.velocity[1] = decay * self.velocity[1]
        elif handcrafted_low_quality:
            self.velocity *= 0.25
        else:
            self.velocity[0] = momentum * self.velocity[0] + (1.0 - momentum) * lon_delta
            self.velocity[1] = momentum * self.velocity[1] + (1.0 - momentum) * lat_delta
        self.state = best_state
        if semantic_recovery:
            self._sync_semantic_recovery(frame, best_state)

        # P1 fix: 深度模式后台维护手工独立状态，避免完全失活
        if self._deep_mode and self._hand_state is not None and not handcrafted_result:
            # 深度可信时（高分数 + 高PSR），手工速度跟随深度
            deep_reliable = (
                best_score >= self.config.deep_high_confidence
                and self.runtime_stats.last_psr >= self.config.deep_velocity_low_psr_threshold
            )
            hand_uninitialized = self._frame_count <= 3 and np.all(self._hand_velocity == 0)
            if deep_reliable or hand_uninitialized:
                self._hand_velocity[0] = self.velocity[0]
                self._hand_velocity[1] = self.velocity[1]
            # 手工状态按自身速度外推
            self._hand_state = SphereState(
                lon=float(wrap_lon(self._hand_state.lon + self._hand_velocity[0])),
                lat=float(clamp_lat(self._hand_state.lat + self._hand_velocity[1])),
                equatorial_width=self._hand_state.equatorial_width,
                angular_height=self._hand_state.angular_height,
            )

# --- 多模板更新（含遮挡/异常帧抑制）---
        if handcrafted_result:
            self._consecutive_good = 0
        elif best_score >= update_quality_threshold:
            self._consecutive_good += 1
        else:
            self._consecutive_good = 0

        if best_score >= high_confidence and not ncc_low_confidence:
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        if (
            self._consecutive_good >= self._template_confirmation_frames()
            and not handcrafted_result
        ):
            self.runtime_stats.template_updates += 1
            self._update_templates(frame, best_state, best_score)
        else:
            for i in range(len(self._template_ages)):
                self._template_ages[i] += 1

        self._sync_aliases()
        result_bbox = self._state_to_output_bbox(self.state, w, h)
        if flow_reliable:
            self._color_bbox = result_bbox.copy()
        self._record_debug_frame(
            frame=frame,
            predicted=predicted,
            result_bbox=result_bbox,
            score=best_score,
            relocalize_applied=relocalize_applied,
        )
        if self.config.handcrafted_flow_enabled:
            self._previous_frame_gray = (
                frame_gray if frame_gray is not None else self._flow_gray(frame)
            )
        else:
            self._previous_frame_gray = None
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

            update_rate = self._template_update_rate(t_type, i == best_idx)
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

                update_rate = self._template_update_rate(t_type, i == best_idx)
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

    def _flow_gray(self, frame: np.ndarray) -> np.ndarray | None:
        return self._handcrafted_gray(frame)

    def _handcrafted_gray(
        self,
        frame: np.ndarray,
        *,
        assume_normalized: bool = False,
    ) -> np.ndarray | None:
        if (
            cv2 is None
            or not (
                self.config.handcrafted_flow_enabled
                or self.config.handcrafted_ncc_enabled
            )
        ):
            return None
        if assume_normalized:
            rgb = (frame * np.float32(255.0)).astype(np.uint8)
        else:
            rgb = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _initialize_color_model(self, frame: np.ndarray, bbox_xywh: np.ndarray) -> None:
        self._color_hue = None
        self._color_bbox = np.asarray(bbox_xywh, dtype=np.float32).copy()
        self._last_color_reliable = False
        if cv2 is None or not self.config.handcrafted_color_enabled:
            return

        rgb = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        x, y, width, height = [int(round(v)) for v in bbox_xywh]
        x0 = max(0, min(x, hsv.shape[1] - 1))
        y0 = max(0, min(y, hsv.shape[0] - 1))
        x1 = max(x0 + 1, min(x + width, hsv.shape[1]))
        y1 = max(y0 + 1, min(y + height, hsv.shape[0]))
        roi = hsv[y0:y1, x0:x1]
        saturated = roi[..., 1] >= self.config.handcrafted_color_min_saturation
        if saturated.mean() < 0.30:
            return

        hues = roi[..., 0][saturated].astype(np.float32)
        if hues.size < self.config.handcrafted_color_min_area:
            return
        angles = hues / 180.0 * 2.0 * math.pi
        concentration = math.hypot(float(np.cos(angles).mean()), float(np.sin(angles).mean()))
        if concentration < 0.55:
            return
        mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
        self._color_hue = (mean_angle % (2.0 * math.pi)) / (2.0 * math.pi) * 180.0

    def _crop_erp_box(self, image: np.ndarray, bbox_xywh: np.ndarray) -> np.ndarray | None:
        x, y, width, height = [int(round(v)) for v in bbox_xywh]
        image_height, image_width = image.shape[:2]
        if width < 2 or height < 2 or image_width < 1 or image_height < 1:
            return None
        width = min(width, image_width)
        y0 = max(0, min(y, image_height - 1))
        y1 = max(y0 + 1, min(y + height, image_height))
        xs = np.mod(np.arange(x, x + width), image_width).astype(np.intp, copy=False)
        return image[y0:y1][:, xs]

    def _initialize_ncc_model(
        self,
        frame: np.ndarray,
        bbox_xywh: np.ndarray,
        gray: np.ndarray | None = None,
    ) -> None:
        self._ncc_initial_template = None
        self._ncc_short_template = None
        self._ncc_initial_resize_cache.clear()
        self._ncc_short_resize_cache.clear()
        self._ncc_flow_mask = np.zeros(gray.shape, dtype=np.uint8) if gray is not None else None
        self._ncc_bbox = np.asarray(bbox_xywh, dtype=np.float32).copy()
        self._ncc_velocity[:] = 0.0
        self._ncc_flow_scale = None
        self._ncc_scale_anchor = self._ncc_bbox[2:4].astype(np.float64)
        self._ncc_flow_was_reliable = False
        self._ncc_frames_since_flow = self.config.handcrafted_ncc_flow_grace_frames + 1
        self._ncc_last_score = 1.0
        self._last_ncc_reliable = False
        if cv2 is None or not self.config.handcrafted_ncc_enabled:
            return

        if gray is None:
            gray = self._handcrafted_gray(frame)
        if gray is None:
            return
        template = self._crop_erp_box(gray, self._ncc_bbox)
        if template is None:
            return
        self._ncc_initial_template = template.copy()
        self._ncc_short_template = template.copy()

    def _predict_with_ncc(
        self,
        frame: np.ndarray,
        gray: np.ndarray | None = None,
    ) -> tuple[SphereState, float, bool]:
        self._last_ncc_reliable = False
        if (
            cv2 is None
            or self._ncc_initial_template is None
            or self._ncc_short_template is None
            or self._ncc_bbox is None
            or self.frame_shape is None
        ):
            if self.state is None:
                raise RuntimeError("Tracker state is unavailable.")
            return self.state, -1.0, False

        if gray is None:
            gray = self._handcrafted_gray(frame)
        if gray is None:
            if self.state is None:
                raise RuntimeError("Tracker state is unavailable.")
            return self.state, -1.0, False
        image_height, image_width = self.frame_shape
        box = self._ncc_bbox.astype(np.float64)
        center_x = float(box[0] + 0.5 * box[2])
        center_y = float(box[1] + 0.5 * box[3])
        relative_motion = max(
            abs(float(self._ncc_velocity[0])) / max(float(box[2]), 1.0),
            abs(float(self._ncc_velocity[1])) / max(float(box[3]), 1.0),
        )
        should_estimate_flow = (
            relative_motion >= self.config.handcrafted_ncc_flow_motion_trigger
            or self._ncc_last_score < self.config.handcrafted_ncc_flow_score_trigger
        )
        flow_delta = self._estimate_ncc_flow(gray, box) if should_estimate_flow else None
        if flow_delta is not None:
            predicted_delta = flow_delta
            self._ncc_velocity = flow_delta.copy()
            self._ncc_flow_was_reliable = True
            self._ncc_frames_since_flow = 0
        elif self._ncc_flow_was_reliable:
            predicted_delta = self._ncc_velocity.copy()
            self._ncc_flow_was_reliable = False
            self._ncc_frames_since_flow += 1
        else:
            predicted_delta = self._ncc_velocity.copy()
            self._ncc_frames_since_flow += 1
        predicted_center_x = center_x + float(predicted_delta[0])
        predicted_center_y = center_y + float(predicted_delta[1])
        search_width = min(
            max(box[2] * self.config.handcrafted_ncc_search_factor, 220.0),
            float(image_width),
        )
        search_width = float(min(max(int(round(search_width)), 1), image_width))
        search_height = max(box[3] * self.config.handcrafted_ncc_search_factor, 180.0)
        x0 = int(math.floor(predicted_center_x - 0.5 * search_width))
        y0 = max(0, int(predicted_center_y - 0.5 * search_height))
        x1 = x0 + int(search_width)
        y1 = min(image_height, int(predicted_center_y + 0.5 * search_height))
        xs = np.mod(np.arange(x0, x1), image_width).astype(np.intp, copy=False)
        search = gray[y0:y1][:, xs]
        if search.shape[0] < 16 or search.shape[1] < 16:
            if self.state is None:
                raise RuntimeError("Tracker state is unavailable.")
            return self.state, -1.0, False

        max_template_std = max(
            float(np.std(self._ncc_short_template)),
            float(np.std(self._ncc_initial_template)),
        )
        search_std = float(np.std(search))
        if (
            flow_delta is None
            and (
                max_template_std < self.config.handcrafted_ncc_min_template_std
                or search_std < self.config.handcrafted_ncc_min_search_std
            )
        ):
            self._ncc_last_score = -1.0
            if self.state is None:
                raise RuntimeError("Tracker state is unavailable.")
            return self.state, -1.0, False

        scale_pairs = [
            (0.85, 0.85),
            (0.93, 0.93),
            (1.0, 1.0),
            (1.08, 1.08),
            (1.16, 1.16),
            (1.15, 0.90),
            (1.25, 0.90),
            (1.35, 0.85),
            (1.0, 0.85),
            (0.90, 1.10),
        ]
        growth_mode = (
            self._deep_mode
            and self._init_bbox_width_px is not None
            and self._init_bbox_height_px is not None
            and self.runtime_stats.last_psr < self.config.deep_ncc_growth_psr_threshold
            and (
                box[2] >= self._init_bbox_width_px * self.config.deep_ncc_growth_min_init_ratio
                or box[3] >= self._init_bbox_height_px * self.config.deep_ncc_growth_min_init_ratio
            )
        )
        if growth_mode:
            scale_pairs.extend(self.config.deep_ncc_growth_scale_pairs)
        best_score = -1.0
        best_box: np.ndarray | None = None
        best_center_x: float | None = None
        best_center_y: float | None = None
        candidates: list[
            tuple[float, float, int, int, np.ndarray, int, int, int, int]
        ] = []
        for width_scale, height_scale in scale_pairs:
            width = max(12, int(round(box[2] * width_scale)))
            height = max(12, int(round(box[3] * height_scale)))
            if width >= search.shape[1] or height >= search.shape[0]:
                continue

            center_location_x = predicted_center_x - x0 - 0.5 * width
            center_location_y = predicted_center_y - y0 - 0.5 * height
            box_area_ratio = (box[2] * box[3]) / max(float(image_width * image_height), 1.0)
            ncc_jump_ratio = (
                self.config.small_target_ncc_max_jump_ratio
                if box_area_ratio < self.config.small_target_threshold
                else self.config.handcrafted_ncc_max_jump_ratio
            )
            max_dx = box[2] * ncc_jump_ratio
            max_dy = box[3] * ncc_jump_ratio
            response_height = search.shape[0] - height + 1
            response_width = search.shape[1] - width + 1
            allowed_x0 = max(0, int(math.floor(center_location_x - max_dx)))
            allowed_y0 = max(0, int(math.floor(center_location_y - max_dy)))
            allowed_x1 = min(response_width, int(math.ceil(center_location_x + max_dx + 1)))
            allowed_y1 = min(response_height, int(math.ceil(center_location_y + max_dy + 1)))
            if allowed_x1 <= allowed_x0 or allowed_y1 <= allowed_y0:
                continue

            for source_name, source in (
                ("short", self._ncc_short_template),
                ("initial", self._ncc_initial_template),
            ):
                candidate_template = self._resize_ncc_template(
                    source_name, source, width, height,
                )
                candidates.append(
                    (
                        width_scale, height_scale, width, height, candidate_template,
                        allowed_x0, allowed_y0, allowed_x1, allowed_y1,
                    ),
                )

        def match_candidate(
            item: tuple[float, float, int, int, np.ndarray, int, int, int, int],
        ) -> np.ndarray:
            (
                _, _, width, height, candidate_template,
                allowed_x0, allowed_y0, allowed_x1, allowed_y1,
            ) = item
            search_window = search[
                allowed_y0 : allowed_y1 + height - 1,
                allowed_x0 : allowed_x1 + width - 1,
            ]
            return cv2.matchTemplate(
                search_window, candidate_template, cv2.TM_CCOEFF_NORMED,
            )

        parallel_workers = max(int(self.config.handcrafted_ncc_parallel_workers), 1)
        if parallel_workers > 1:
            if self._ncc_executor is None:
                self._ncc_executor = ThreadPoolExecutor(max_workers=parallel_workers)
            responses = list(self._ncc_executor.map(match_candidate, candidates))
        else:
            responses = [match_candidate(item) for item in candidates]
        self.runtime_stats.ncc_exact_matches += len(responses)

        for candidate_index, candidate in enumerate(candidates):
            (
                width_scale, height_scale, width, height, candidate_template,
                allowed_x0, allowed_y0, _, _,
            ) = candidate
            response = responses[candidate_index]
            _, raw_score, _, local_location = cv2.minMaxLoc(response)
            location_x = allowed_x0 + local_location[0]
            location_y = allowed_y0 + local_location[1]
            candidate_start_x = x0 + location_x
            candidate_center_x = candidate_start_x + 0.5 * width
            candidate_center_y = y0 + location_y + 0.5 * height
            normalized_distance = math.hypot(
                (candidate_center_x - predicted_center_x) / search_width,
                (candidate_center_y - predicted_center_y) / search_height,
            )
            adjusted_score = (
                float(raw_score)
                - self.config.handcrafted_ncc_distance_penalty * normalized_distance
                - self.config.handcrafted_ncc_scale_penalty
                * (abs(math.log(width_scale)) + abs(math.log(height_scale)))
            )
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_box = np.array(
                    [candidate_start_x % image_width, y0 + location_y, width, height],
                    dtype=np.float64,
                )
                best_center_x = candidate_center_x
                best_center_y = candidate_center_y

        if best_box is None:
            if self.state is None:
                raise RuntimeError("Tracker state is unavailable.")
            return self.state, best_score, False

        self._ncc_last_score = float(best_score)
        if best_score < self.config.handcrafted_ncc_reliable_score:
            if self.state is None:
                raise RuntimeError("Tracker state is unavailable.")
            return self.state, best_score, False

        if best_center_x is None or best_center_y is None:
            best_center_x = float(best_box[0] + 0.5 * best_box[2])
            best_center_y = float(best_box[1] + 0.5 * best_box[3])
        measured_velocity = np.array(
            [
                best_center_x - center_x,
                best_center_y - center_y,
            ],
            dtype=np.float64,
        )
        recent_flow = (
            self._ncc_frames_since_flow <= max(int(self.config.handcrafted_ncc_flow_grace_frames), 0)
        )
        reference_delta = flow_delta if flow_delta is not None else (predicted_delta if recent_flow else None)
        if not self._ncc_candidate_consistent(
            measured_velocity,
            reference_delta,
            box,
            best_score,
            strict_axes=flow_delta is not None,
        ):
            self.runtime_stats.ncc_flow_disagreement_rejects += 1
            fallback_delta = (
                reference_delta
                if reference_delta is not None
                else predicted_delta
            )
            best_box = box.copy()
            best_box[0] += float(fallback_delta[0])
            best_box[1] += float(fallback_delta[1])
            best_box[0] = float(best_box[0] % image_width)
            best_box[1] = float(np.clip(best_box[1], 0.0, image_height - best_box[3]))
            measured_velocity = fallback_delta.copy()
        guarded_box = self._guard_ncc_candidate_scale(
            best_box, box, self._ncc_flow_scale if flow_delta is not None else None, best_score,
        )
        if not np.array_equal(guarded_box[2:4], best_box[2:4]):
            self.runtime_stats.ncc_flow_scale_adjustments += 1
        best_box = guarded_box
        anchored_box = self._guard_ncc_scale_anchor(best_box, box, best_score)
        if not np.array_equal(anchored_box[2:4], best_box[2:4]):
            self.runtime_stats.ncc_scale_anchor_adjustments += 1
        best_box = anchored_box
        previous_size = np.maximum(box[2:4], 1.0)
        max_scale_step = max(float(self.config.handcrafted_ncc_max_scale_step), 1.01)
        if growth_mode:
            max_scale_step = max(
                max_scale_step,
                float(self.config.deep_ncc_growth_max_scale_step),
            )
        temporal_scale = best_box[2:4] / previous_size
        guarded_temporal_scale = np.clip(
            temporal_scale, 1.0 / max_scale_step, max_scale_step,
        )
        if not np.array_equal(temporal_scale, guarded_temporal_scale):
            guarded = best_box.copy()
            center = best_box[:2] + 0.5 * best_box[2:4]
            guarded[2:4] = previous_size * guarded_temporal_scale
            guarded[:2] = center - 0.5 * guarded[2:4]
            best_box = guarded
            self.runtime_stats.ncc_flow_scale_adjustments += 1
        if best_score < self.config.handcrafted_ncc_low_score_commit_threshold:
            # A weak NCC peak is an observation, not a new tracking anchor.
            fallback_delta = reference_delta if reference_delta is not None else predicted_delta
            if fallback_delta is None:
                fallback_delta = np.zeros(2, dtype=np.float64)
            best_box[0] = float((box[0] + fallback_delta[0]) % image_width)
            best_box[1] = float(np.clip(box[1] + fallback_delta[1], 0.0, image_height - best_box[3]))
            measured_velocity = np.asarray(fallback_delta, dtype=np.float64)
        best_box[0] = float(best_box[0] % image_width)
        best_box[1] = float(np.clip(best_box[1], 0.0, image_height - best_box[3]))
        momentum = float(np.clip(self.config.handcrafted_ncc_velocity_momentum, 0.0, 1.0))
        self._ncc_velocity = momentum * self._ncc_velocity + (1.0 - momentum) * measured_velocity
        self._ncc_bbox = best_box.astype(np.float32)
        if best_score >= self.config.handcrafted_ncc_scale_anchor_score:
            self._ncc_scale_anchor = best_box[2:4].copy()
        if best_score >= self.config.handcrafted_ncc_update_score:
            patch = self._crop_erp_box(gray, self._ncc_bbox)
            if patch is not None:
                patch = cv2.resize(
                    patch,
                    (self._ncc_short_template.shape[1], self._ncc_short_template.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                rate = float(np.clip(self.config.handcrafted_ncc_update_rate, 0.0, 1.0))
                self._ncc_short_template = cv2.addWeighted(
                    self._ncc_short_template, 1.0 - rate, patch, rate, 0.0,
                )
                self._ncc_short_resize_cache.clear()

        self._last_ncc_reliable = True
        state = erp_bbox_to_state(self._ncc_bbox, image_width, image_height)
        return state, best_score, True

    def _resize_ncc_template(
        self,
        source_name: str,
        source: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        key = (int(width), int(height))
        cache = (
            self._ncc_short_resize_cache
            if source_name == "short"
            else self._ncc_initial_resize_cache
        )
        cached = cache.get(key)
        if cached is None:
            cached = cv2.resize(
                source, key, interpolation=cv2.INTER_LINEAR,
            )
            cache[key] = cached
        return cached

    def _ncc_candidate_consistent(
        self,
        measured_velocity: np.ndarray,
        reference_delta: np.ndarray | None,
        box: np.ndarray,
        score: float,
        *,
        strict_axes: bool = False,
    ) -> bool:
        deep_jump_guard = (
            self._deep_mode
            and score < self.config.deep_ncc_jump_score_threshold
        )
        if not deep_jump_guard and (
            reference_delta is None
            or score >= self.config.handcrafted_ncc_flow_disagreement_score
        ):
            return True
        scale = max(float(box[2]), float(box[3]), 1.0)
        if scale < self.config.handcrafted_ncc_flow_guard_min_size:
            return True
        if (
            self._deep_mode
            and score < self.config.deep_ncc_jump_score_threshold
        ):
            max_jump_ratio = max(float(self.config.deep_ncc_max_jump_ratio), 0.1)
            if (
                abs(float(measured_velocity[0])) > max_jump_ratio * max(float(box[2]), 1.0)
                or abs(float(measured_velocity[1])) > max_jump_ratio * max(float(box[3]), 1.0)
            ):
                return False
        if reference_delta is None:
            return True
        if float(np.linalg.norm(reference_delta)) < self.config.handcrafted_ncc_flow_motion_trigger * scale:
            return True
        disagreement = float(np.linalg.norm(measured_velocity - reference_delta))
        opposite_direction = float(np.dot(measured_velocity, reference_delta)) < 0.0
        axis_disagreement = (
            abs(float(measured_velocity[0] - reference_delta[0]))
            > self.config.handcrafted_ncc_flow_axis_disagreement_ratio * max(float(box[2]), 1.0)
            or abs(float(measured_velocity[1] - reference_delta[1]))
            > self.config.handcrafted_ncc_flow_axis_disagreement_ratio * max(float(box[3]), 1.0)
        )
        return not (
            (strict_axes and axis_disagreement)
            or (
                opposite_direction
                and disagreement > self.config.handcrafted_ncc_flow_disagreement_ratio * scale
            )
        )

    def _guard_ncc_candidate_scale(
        self,
        candidate_box: np.ndarray,
        previous_box: np.ndarray,
        flow_scale: np.ndarray | None,
        score: float,
    ) -> np.ndarray:
        if (
            flow_scale is None
            or score >= self.config.handcrafted_ncc_flow_disagreement_score
            or max(float(previous_box[2]), float(previous_box[3]))
            < self.config.handcrafted_ncc_flow_guard_min_size
        ):
            return candidate_box

        tolerance = max(float(self.config.handcrafted_ncc_flow_scale_tolerance), 0.0)
        lower = np.maximum(flow_scale / (1.0 + tolerance), 0.5)
        upper = np.minimum(flow_scale * (1.0 + tolerance), 2.0)
        candidate_scale = candidate_box[2:4] / np.maximum(previous_box[2:4], 1.0)
        guarded_scale = np.clip(candidate_scale, lower, upper)
        if np.array_equal(candidate_scale, guarded_scale):
            return candidate_box

        guarded = candidate_box.copy()
        center = candidate_box[:2] + 0.5 * candidate_box[2:4]
        guarded[2:4] = previous_box[2:4] * guarded_scale
        guarded[:2] = center - 0.5 * guarded[2:4]
        return guarded

    def _guard_ncc_scale_anchor(
        self,
        candidate_box: np.ndarray,
        previous_box: np.ndarray,
        score: float,
    ) -> np.ndarray:
        if (
            self._ncc_scale_anchor is None
            or max(float(previous_box[2]), float(previous_box[3]))
            < self.config.handcrafted_ncc_flow_guard_min_size
        ):
            return candidate_box

        min_ratio = float(np.clip(self.config.handcrafted_ncc_scale_anchor_min_ratio, 0.0, 1.0))
        minimum_size = self._ncc_scale_anchor * min_ratio
        if self._deep_mode and self._init_bbox_width_px is not None and self._init_bbox_height_px is not None:
            collapse_ratio = max(float(self.config.deep_ncc_scale_collapse_ratio), 0.1)
            minimum_size = np.maximum(
                minimum_size,
                np.array(
                    [
                        self._init_bbox_width_px * collapse_ratio,
                        self._init_bbox_height_px * collapse_ratio,
                    ],
                    dtype=np.float64,
                ),
            )
        guarded_size = np.maximum(candidate_box[2:4], minimum_size)
        if np.array_equal(candidate_box[2:4], guarded_size):
            return candidate_box

        guarded = candidate_box.copy()
        center = candidate_box[:2] + 0.5 * candidate_box[2:4]
        guarded[2:4] = guarded_size
        guarded[:2] = center - 0.5 * guarded_size
        return guarded

    def _estimate_ncc_flow(
        self,
        current_gray: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray | None:
        self._ncc_flow_scale = None
        previous_gray = self._previous_frame_gray
        if cv2 is None or previous_gray is None:
            return None

        x0 = max(0, int(math.floor(box[0])))
        y0 = max(0, int(math.floor(box[1])))
        x1 = min(previous_gray.shape[1], int(math.ceil(box[0] + box[2])))
        y1 = min(previous_gray.shape[0], int(math.ceil(box[1] + box[3])))
        box_width = int(round(box[2]))
        if box_width < 4 or y1 - y0 < 4:
            return None
        if self._ncc_flow_mask is None or self._ncc_flow_mask.shape != previous_gray.shape:
            self._ncc_flow_mask = np.zeros(previous_gray.shape, dtype=np.uint8)
        mask = self._ncc_flow_mask
        mask.fill(0)
        xs = np.mod(
            np.arange(int(math.floor(box[0])), int(math.floor(box[0])) + box_width),
            previous_gray.shape[1],
        ).astype(np.intp, copy=False)
        rows = np.arange(y0, y1, dtype=np.intp)[:, None]
        mask[rows, xs[None, :]] = 255

        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.config.handcrafted_flow_max_points,
            qualityLevel=0.005,
            minDistance=3,
            mask=mask,
            blockSize=3,
        )

        if points is None or len(points) < self.config.handcrafted_ncc_flow_min_points:
            return None

        lk_args = {
            "winSize": (31, 31),
            "maxLevel": 4,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, **lk_args,
        )
        if next_points is None or status is None:
            return None
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, next_points, None, **lk_args,
        )
        if back_points is None or back_status is None:
            return None

        forward_backward = np.linalg.norm(back_points - points, axis=2).reshape(-1)
        valid = (
            (status.reshape(-1) > 0)
            & (back_status.reshape(-1) > 0)
            & (forward_backward < self.config.handcrafted_ncc_flow_max_fb_error)
        )
        previous_points = points.reshape(-1, 2)[valid]
        tracked_points = next_points.reshape(-1, 2)[valid]
        deltas = tracked_points - previous_points
        if len(deltas) < self.config.handcrafted_ncc_flow_min_points:
            return None

        median_delta = np.median(deltas, axis=0)
        spread = float(np.median(np.linalg.norm(deltas - median_delta, axis=1)))
        inlier_ratio = float(len(deltas) / len(points))
        if (
            inlier_ratio < self.config.handcrafted_ncc_flow_min_inlier_ratio
            or spread > self.config.handcrafted_ncc_flow_max_spread
        ):
            return None

        affine, affine_inliers = cv2.estimateAffine2D(
            previous_points,
            tracked_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is not None and affine_inliers is not None:
            affine_inlier_ratio = float(np.mean(affine_inliers.reshape(-1) > 0))
            if affine_inlier_ratio >= self.config.handcrafted_ncc_flow_scale_min_inlier_ratio:
                scale_x = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
                scale_y = math.hypot(float(affine[0, 1]), float(affine[1, 1]))
                if np.isfinite(scale_x) and np.isfinite(scale_y):
                    self._ncc_flow_scale = np.clip(
                        np.array([scale_x, scale_y], dtype=np.float64), 0.80, 1.25,
                    )
        return median_delta.astype(np.float64)

    def _predict_with_color(
        self,
        frame: np.ndarray,
        anchor: SphereState,
    ) -> tuple[SphereState, bool]:
        self._last_color_reliable = False
        if cv2 is None or self._color_hue is None or self._color_bbox is None or self.frame_shape is None:
            return anchor, False

        rgb = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        image_height, image_width = self.frame_shape
        previous_box = self._color_bbox.astype(np.float64)
        center_x = float(previous_box[0] + 0.5 * previous_box[2])
        center_y = float(previous_box[1] + 0.5 * previous_box[3])
        search_width = min(
            float(self.config.handcrafted_color_search_width),
            max(220.0, previous_box[2] * 3.0),
        )
        search_height = min(
            float(self.config.handcrafted_color_search_height),
            max(220.0, previous_box[3] * 3.0),
        )
        x0 = max(0, int(center_x - 0.5 * search_width))
        x1 = min(image_width, int(center_x + 0.5 * search_width))
        y0 = max(0, int(center_y - 0.5 * search_height))
        y1 = min(image_height, int(center_y + 0.5 * search_height))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return anchor, False

        region = hsv[y0:y1, x0:x1]
        hue_delta = np.abs(region[..., 0].astype(np.float32) - self._color_hue)
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
        mask = (
            (hue_delta < self.config.handcrafted_color_hue_tolerance)
            & (region[..., 1] >= self.config.handcrafted_color_min_saturation)
            & (region[..., 2] >= self.config.handcrafted_color_min_value)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        previous_area = max(float(previous_box[2] * previous_box[3]), 1.0)
        best_box: np.ndarray | None = None
        best_score = -float("inf")
        for index in range(1, count):
            comp_x, comp_y, comp_width, comp_height, area = [int(v) for v in stats[index]]
            if area < self.config.handcrafted_color_min_area:
                continue
            if area > max(40000.0, previous_area * 5.0):
                continue
            candidate_area_ratio = float(area) / previous_area

            candidate_center = np.array(
                [x0 + comp_x + 0.5 * comp_width, y0 + comp_y + 0.5 * comp_height],
                dtype=np.float64,
            )
            normalized_distance = np.linalg.norm(
                (candidate_center - np.array([center_x, center_y]))
                / np.array([search_width, search_height])
            )
            fill_ratio = float(area) / max(float(comp_width * comp_height), 1.0)
            scale_change = abs(math.log(max(float(area), 1.0) / previous_area))
            score = (
                fill_ratio
                - 1.5 * normalized_distance
                - 0.25 * scale_change
                + 0.1 * math.log(float(area) + 1.0)
            )
            if score > best_score:
                best_score = score
                best_box = np.array(
                    [x0 + comp_x, y0 + comp_y, comp_width, comp_height],
                    dtype=np.float64,
                )

        if best_box is None:
            return anchor, False

        max_scale = max(float(self.config.handcrafted_color_max_scale_step), 1.01)
        width = float(np.clip(best_box[2], previous_box[2] / max_scale, previous_box[2] * max_scale))
        height = float(np.clip(best_box[3], previous_box[3] / max_scale, previous_box[3] * max_scale))
        best_box[0] += 0.5 * (best_box[2] - width)
        best_box[1] += 0.5 * (best_box[3] - height)
        best_box[2] = width
        best_box[3] = height
        best_box[0] = float(np.clip(best_box[0], 0.0, image_width - width))
        best_box[1] = float(np.clip(best_box[1], 0.0, image_height - height))
        self._color_bbox = best_box.astype(np.float32)
        self._last_color_reliable = True
        return erp_bbox_to_state(self._color_bbox, image_width, image_height), True

    def _score_state_handcrafted(self, frame: np.ndarray, state: SphereState) -> float:
        fov_x, fov_y = self._handcrafted_match_fov(state)
        patch = tangent_patch(
            frame,
            state.lon,
            state.lat,
            fov_x,
            fov_y,
            self.config.template_size,
            self.config.template_size,
        )
        return self._score_patch(patch)

    def _predict_with_optical_flow(
        self,
        frame: np.ndarray,
        state: SphereState,
        current_gray: np.ndarray | None = None,
    ) -> tuple[SphereState, bool]:
        self._last_flow_reliable = False
        self._last_flow_inlier_ratio = 0.0
        self._last_flow_fb_error = 0.0
        self._last_flow_spread = 0.0
        if current_gray is None:
            current_gray = self._flow_gray(frame)
        previous_gray = self._previous_frame_gray
        if (
            cv2 is None
            or not self.config.handcrafted_flow_enabled
            or current_gray is None
            or previous_gray is None
            or self.frame_shape is None
        ):
            return state, False

        image_height, image_width = self.frame_shape
        bbox = self._state_to_output_bbox(state, image_width, image_height)
        x, y, width, height = [float(v) for v in bbox]
        if x + width > image_width:
            return state, False

        padding = self.config.handcrafted_flow_padding
        x0 = max(0, int(math.floor(x - padding * width)))
        y0 = max(0, int(math.floor(y - padding * height)))
        x1 = min(image_width, int(math.ceil(x + (1.0 + padding) * width)))
        y1 = min(image_height, int(math.ceil(y + (1.0 + padding) * height)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return state, False

        mask = np.zeros_like(previous_gray, dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.config.handcrafted_flow_max_points,
            qualityLevel=0.005,
            minDistance=3,
            mask=mask,
            blockSize=3,
        )
        if points is None or len(points) < self.config.handcrafted_flow_min_points:
            return state, False

        lk_args = {
            "winSize": (31, 31),
            "maxLevel": 4,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, **lk_args,
        )
        if next_points is None or status is None:
            return state, False
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, next_points, None, **lk_args,
        )
        if back_points is None or back_status is None:
            return state, False

        forward_backward = np.linalg.norm(back_points - points, axis=2).reshape(-1)
        valid = (
            (status.reshape(-1) > 0)
            & (back_status.reshape(-1) > 0)
            & (forward_backward < 1.5)
        )
        previous_points = points.reshape(-1, 2)[valid]
        tracked_points = next_points.reshape(-1, 2)[valid]
        if len(previous_points) < self.config.handcrafted_flow_min_points:
            return state, False

        deltas = tracked_points - previous_points
        median_delta = np.median(deltas, axis=0)
        spread = float(np.median(np.linalg.norm(deltas - median_delta, axis=1)))
        fb_error = float(np.median(forward_backward[valid]))
        inlier_ratio = float(len(previous_points) / len(points))
        reliable = (
            inlier_ratio >= self.config.handcrafted_flow_min_inlier_ratio
            and fb_error <= self.config.handcrafted_flow_max_fb_error
            and spread <= self.config.handcrafted_flow_max_spread
        )
        self._last_flow_inlier_ratio = inlier_ratio
        self._last_flow_fb_error = fb_error
        self._last_flow_spread = spread
        self._last_flow_reliable = reliable
        if not reliable:
            return state, False

        delta_lon = float(median_delta[0]) / image_width * 2.0 * math.pi
        delta_lat = -float(median_delta[1]) / image_height * math.pi
        return SphereState(
            lon=float(wrap_lon(state.lon + delta_lon)),
            lat=float(clamp_lat(state.lat + delta_lat)),
            equatorial_width=state.equatorial_width,
            angular_height=state.angular_height,
        ), True

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
                "state_trust": f"{self._last_state_trust:.6f}",
                "local_jump_gated": self._last_local_jump_gated,
                "center_score": f"{self._last_center_score:.6f}",
                "best_minus_center": f"{self._last_best_minus_center:.6f}",
                "center_preferred": self._last_center_preferred,
                "flow_reliable": self._last_flow_reliable,
                "flow_inlier_ratio": f"{self._last_flow_inlier_ratio:.6f}",
                "flow_fb_error": f"{self._last_flow_fb_error:.6f}",
                "flow_spread": f"{self._last_flow_spread:.6f}",
                "color_reliable": self._last_color_reliable,
                "ncc_reliable": self._last_ncc_reliable,
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
        return self.deep_extractor.extract_template_feature(patch, assume_normalized=True)

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

        is_small = self._is_small_target(state)

        if is_small:
            enlarge = self.config.small_target_template_enlarge
        else:
            enlarge = self._get_template_enlarge(state.lat) if trust_location else self.config.deep_template_enlarge

        angles = self._get_rotation_angles(state.lat) if trust_location else self.config.deep_template_rotations_deg
        fov_x, fov_y = state_size_to_fov(state, enlarge=enlarge)
        patch = tangent_patch(frame, state.lon, state.lat, fov_x, fov_y, size, size)
        return [
            self.deep_extractor.extract_template_feature(
                self._rotate_patch(patch, angle_deg), assume_normalized=True,
            )
            for angle_deg in angles
        ]

    def _extract_search_feat(
        self, frame: np.ndarray, lon: float, lat: float, fov_x: float, fov_y: float, refine: bool = False
    ) -> Any:
        """提取搜索区域的深度特征。"""
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        patch = self._extract_search_patch(frame, lon, lat, fov_x, fov_y, refine=refine)
        return self.deep_extractor.extract_search_feature(
            patch, refine=refine, assume_normalized=True,
        )

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

    def _fused_template_responses_batch(self, search_features: Any) -> tuple[Any, Any]:
        fused_response = None
        total_weight = 0.0
        reference_template = None
        for index, template_feat in enumerate(self._template_feats):
            template_bank = (
                self._template_feat_banks[index]
                if index < len(self._template_feat_banks)
                else [template_feat]
            )
            bank_features = self.deep_extractor._torch.cat(template_bank, dim=0)
            bank_responses = self.similarity_head.forward_bank_to_searches(
                bank_features, search_features,
            )
            self.runtime_stats.response_maps += int(bank_responses.shape[0] * bank_responses.shape[1])
            self.runtime_stats.response_batches += 1
            bank_peaks = bank_responses.flatten(start_dim=3).max(dim=3).values.squeeze(-1)
            best_bank = bank_peaks.argmax(dim=1)
            search_indices = self.deep_extractor._torch.arange(
                int(search_features.shape[0]), device=search_features.device,
            )
            response = bank_responses[search_indices, best_bank]
            weight = self._template_weight(index)
            fused_response = response * weight if fused_response is None else fused_response + response * weight
            total_weight += weight
            if reference_template is None:
                reference_template = template_feat

        if fused_response is None or reference_template is None:
            raise RuntimeError("No deep templates available for scoring.")
        if total_weight > 0.0:
            fused_response = fused_response / total_weight
        return fused_response, reference_template

    def _response_scores_batch(self, responses: Any) -> list[float]:
        scores, _, _ = self._response_scores_offsets_batch(responses, None, None)
        return scores

    def _response_scores_offsets_batch(
        self,
        responses: Any,
        template_feat: Any | None,
        search_features: Any | None,
    ) -> tuple[list[float], list[tuple[float, float]], list[dict[str, float]]]:
        torch = self.deep_extractor._torch
        response_maps = responses.detach().float().squeeze(1)
        flat = response_maps.flatten(start_dim=1)
        peaks, peak_indices = flat.max(dim=1)
        height, width = int(response_maps.shape[-2]), int(response_maps.shape[-1])
        peak_y = peak_indices // width
        peak_x = peak_indices % width
        grid_y = torch.arange(height, device=responses.device)[None, :, None]
        grid_x = torch.arange(width, device=responses.device)[None, None, :]
        exclusion = (
            (torch.abs(grid_y - peak_y[:, None, None]) <= 1)
            & (torch.abs(grid_x - peak_x[:, None, None]) <= 1)
        )
        side = response_maps.masked_fill(exclusion, float("nan")).flatten(start_dim=1)
        side_count = torch.isfinite(side).sum(dim=1)
        side_mean = torch.nanmean(side, dim=1)
        side_mean = torch.where(side_count > 0, side_mean, flat.mean(dim=1))
        centered = side - side_mean[:, None]
        side_var = torch.nanmean(centered * centered, dim=1)
        fallback_var = ((flat - side_mean[:, None]) ** 2).mean(dim=1)
        side_var = torch.where(side_count > 0, side_var, fallback_var)
        psr = (peaks - side_mean) / torch.sqrt(side_var.clamp_min(1e-12))
        confidence = torch.sigmoid((psr - 2.0) / 2.0)
        response_min = flat.min(dim=1).values
        apce_den = ((flat - response_min[:, None]) ** 2).mean(dim=1)
        apce = ((peaks - response_min) ** 2) / apce_den.clamp_min(1e-6)

        if template_feat is not None and search_features is not None:
            template_h, template_w = int(template_feat.shape[-2]), int(template_feat.shape[-1])
            search_h, search_w = int(search_features.shape[-2]), int(search_features.shape[-1])
            max_offset_y = max(search_h - template_h, 0) / (2.0 * max(search_h, 1))
            max_offset_x = max(search_w - template_w, 0) / (2.0 * max(search_w, 1))
            offset_y = (
                (0.5 - peak_y.float() / (height - 1)) * 2.0 * max_offset_y
                if height > 1 else torch.zeros_like(peaks)
            )
            offset_x = (
                (peak_x.float() / (width - 1) - 0.5) * 2.0 * max_offset_x
                if width > 1 else torch.zeros_like(peaks)
            )
        else:
            offset_y = torch.zeros_like(peaks)
            offset_x = torch.zeros_like(peaks)

        packed = torch.stack((confidence, offset_y, offset_x, peaks, psr, apce), dim=1).cpu().tolist()
        scores = [float(row[0]) for row in packed]
        offsets = [(float(row[1]), float(row[2])) for row in packed]
        metadata = [
            {"peak": float(row[3]), "psr": float(row[4]), "apce": float(row[5])}
            for row in packed
        ]
        return scores, offsets, metadata

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
        # In ERP image space, a response peak above center means the target moved upward,
        # which corresponds to increasing latitude rather than decreasing it.
        offset_y = (0.5 - (peak_y / (h - 1))) * 2.0 * max_offset_y if h > 1 else 0.0
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
        self.runtime_stats.local_searches += 1
        is_small = self._is_small_target(predicted)
        local_grid_radius = (
            self.config.small_target_local_grid_radius
            if is_small else self.config.local_grid_radius
        )
        local_step_factor = (
            self.config.small_target_local_step_factor
            if is_small else self.config.local_step_factor
        )
        step_lon = max(
            predicted.equatorial_width * local_step_factor / max(math.cos(predicted.lat), 1e-3),
            math.radians(2.0),
        )
        step_lat = max(predicted.angular_height * local_step_factor, math.radians(2.0))
        center_fov_x, center_fov_y = self._handcrafted_match_fov(predicted)
        center_patch = tangent_patch(
            frame,
            predicted.lon,
            predicted.lat,
            center_fov_x,
            center_fov_y,
            self.config.template_size,
            self.config.template_size,
        )
        center_score = self._score_patch(center_patch)
        self._last_center_score = float(center_score)
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

            for dx in range(-local_grid_radius, local_grid_radius + 1):
                for dy in range(-local_grid_radius, local_grid_radius + 1):
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
        self._last_best_minus_center = float(best_score - center_score)
        if (
            best_score < self.config.handcrafted_hold_position_score
            and self._last_best_minus_center <= self.config.handcrafted_center_score_margin
        ):
            best_state = predicted
            best_score = float(center_score)
            self._last_center_preferred = True
            if capture_debug:
                best_patch = center_patch.copy()
        if capture_debug:
            self._debug_payload = {}
            if best_patch is not None:
                self._debug_payload["match_patch"] = best_patch
        self.runtime_stats.last_score = float(best_score)
        return best_state, best_score

    def _try_handcrafted_fallback(
        self,
        frame: np.ndarray,
        predicted: SphereState,
        gray: np.ndarray | None = None,
    ) -> tuple[SphereState, float, str]:
        """深度模式低置信时，尝试手工混合追踪（光流/NCC/颜色）作为兜底。

        复用手工分支的路由逻辑：
        - 无颜色 → NCC 匹配
        - 有颜色 → 光流 → 颜色连通域 → 局部搜索

        P1：使用独立的 _hand_state / _hand_velocity 作为锚点，避免被深度漂移污染。
        """
        if self._hand_state is None:
            hand_predicted = predicted
        else:
            hand_predicted = SphereState(
                lon=float(wrap_lon(self._hand_state.lon + self._hand_velocity[0])),
                lat=float(clamp_lat(self._hand_state.lat + self._hand_velocity[1])),
                equatorial_width=self._hand_state.equatorial_width,
                angular_height=self._hand_state.angular_height,
            )
        if self._color_hue is None:
            ncc_state, ncc_score, ncc_reliable = self._predict_with_ncc(frame, gray=gray)
            if ncc_reliable:
                self.runtime_stats.fallback_ncc_results += 1
                return ncc_state, self._handcrafted_confidence(ncc_score, "ncc"), "ncc"
            self.runtime_stats.fallback_local_search_results += 1
            state, score = self._local_search_handcrafted(frame, hand_predicted)
            return state, score, "local_search"
        else:
            flow_state, flow_reliable = self._predict_with_optical_flow(
                frame, self._hand_state or self.state, current_gray=gray,
            )
            if flow_reliable:
                flow_score = self._score_state_handcrafted(frame, flow_state)
                self.runtime_stats.fallback_flow_results += 1
                return flow_state, self._handcrafted_confidence(flow_score, "flow"), "flow"
            color_state, color_reliable = self._predict_with_color(frame, hand_predicted)
            if color_reliable:
                self.runtime_stats.fallback_color_results += 1
                return color_state, self.config.handcrafted_high_confidence, "color"
            self.runtime_stats.fallback_local_search_results += 1
            state, score = self._local_search_handcrafted(frame, hand_predicted)
            return state, score, "local_search"

    def _local_search_deep(self, frame: np.ndarray, predicted: SphereState) -> tuple[SphereState, float]:
        """深度特征 coarse-to-fine 搜索（三模板加权融合版本）。"""
        self.runtime_stats.local_searches += 1
        best_state = predicted
        best_score = -1.0
        best_debug_payload: dict[str, np.ndarray] | None = None
        capture_debug = self._should_capture_debug()

        is_small = self._is_small_target(predicted)

        search_enlarge = (
            self.config.small_target_search_enlarge
            if is_small
            else self.config.deep_search_enlarge
        )

        scale_factors = self.config.deep_scale_factors
        scale_data: list[tuple[float, float, float, float, float]] = []
        coarse_patches: list[np.ndarray] = []
        for scale in scale_factors:
            candidate_width, candidate_height = self._clamp_target_size(
                predicted.equatorial_width * scale,
                predicted.angular_height * scale,
            )
            fov_x, fov_y = state_size_to_fov(
                SphereState(predicted.lon, predicted.lat, candidate_width, candidate_height),
                enlarge=search_enlarge,
            )
            scale_data.append((scale, fov_x, fov_y, candidate_width, candidate_height))
            coarse_patches.append(self._extract_search_patch(
                frame, predicted.lon, predicted.lat, fov_x, fov_y, refine=False,
            ))

        coarse_features = self.deep_extractor.extract_search_features_batch(
            coarse_patches,
            refine=False,
            chunk_size=len(coarse_patches),
            assume_normalized=True,
        )
        coarse_responses, coarse_reference = self._fused_template_responses_batch(coarse_features)
        coarse_scores, coarse_offsets, coarse_metadata = self._response_scores_offsets_batch(
            coarse_responses, coarse_reference, coarse_features,
        )
        coarse_results: list[tuple[float, float, float, dict[str, float]]] = []
        for index, (_, fov_x, fov_y, _, _) in enumerate(scale_data):
            score = coarse_scores[index]
            off_y, off_x = coarse_offsets[index]
            coarse_lon = wrap_lon(predicted.lon + off_x * fov_x)
            coarse_lat = clamp_lat(predicted.lat + off_y * fov_y)
            coarse_results.append((float(score), float(coarse_lon), float(coarse_lat), coarse_metadata[index]))

        refine_indices = self._select_deep_refine_indices(
            [result[0] for result in coarse_results],
            self.config.deep_refine_topk,
        )
        refine_index_set = set(refine_indices)
        refine_patches: list[np.ndarray] = []
        for index in refine_indices:
            _, coarse_lon, coarse_lat, _ = coarse_results[index]
            _, fov_x, fov_y, _, _ = scale_data[index]
            refine_patches.append(self._extract_search_patch(
                frame, coarse_lon, coarse_lat, fov_x * 0.5, fov_y * 0.5, refine=True,
            ))

        refine_features = self.deep_extractor.extract_search_features_batch(
            refine_patches,
            refine=True,
            chunk_size=len(refine_patches),
            assume_normalized=True,
        )
        refine_responses, refine_reference = self._fused_template_responses_batch(refine_features)
        refine_scores, refine_offsets, refine_metadata = self._response_scores_offsets_batch(
            refine_responses, refine_reference, refine_features,
        )

        refine_positions = {index: position for position, index in enumerate(refine_indices)}
        for index, (scale, fov_x, fov_y, candidate_width, candidate_height) in enumerate(scale_data):
            score, coarse_lon, coarse_lat, _ = coarse_results[index]

            scale_best_score = -1.0
            scale_best_state = predicted
            scale_best_debug_payload: dict[str, np.ndarray] | None = None

            if index not in refine_index_set:
                scale_best_score = score - 0.02 * abs(scale - 1.0)
                scale_best_state = SphereState(
                    lon=float(coarse_lon), lat=float(coarse_lat),
                    equatorial_width=float(candidate_width),
                    angular_height=float(candidate_height),
                )

            if index in refine_index_set:
                refine_position = refine_positions[index]
                refine_fov_x = fov_x * 0.5
                refine_fov_y = fov_y * 0.5
                refine_score = refine_scores[refine_position]
                roff_y, roff_x = refine_offsets[refine_position]
                refine_meta = refine_metadata[refine_position]

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
                        scale_best_debug_payload = {
                            "coarse_patch": coarse_patches[index],
                            "refine_patch": refine_patches[refine_position],
                            "refine_response": self._response_to_numpy(
                                refine_responses[refine_position : refine_position + 1],
                            ),
                        }

            if scale_best_score > best_score:
                best_score = scale_best_score
                best_state = scale_best_state
                self.runtime_stats.last_score = float(best_score)
                if index in refine_index_set:
                    self.runtime_stats.last_peak = refine_meta["peak"]
                    self.runtime_stats.last_psr = refine_meta["psr"]
                    self.runtime_stats.last_apce = refine_meta["apce"]
                if capture_debug:
                    best_debug_payload = scale_best_debug_payload

        if capture_debug:
            self._debug_payload = best_debug_payload or {}
        return best_state, best_score

    @staticmethod
    def _select_deep_refine_indices(
        coarse_scores: list[float] | tuple[float, ...],
        requested_topk: int,
    ) -> list[int]:
        """Return the coarse candidates that should receive fine refinement."""
        count = len(coarse_scores)
        requested = int(requested_topk)
        if requested <= 0 or requested >= count:
            return list(range(count))
        return sorted(range(count), key=coarse_scores.__getitem__, reverse=True)[:requested]

    def _reloc_scores_deep_batch(self, patches: list[np.ndarray]) -> list[tuple[float, float]]:
        if not patches:
            return []
        search_features = self.deep_extractor.extract_search_features_batch(
            patches,
            refine=False,
            chunk_size=16,
            assume_normalized=True,
        )
        return self._reloc_scores_deep_features(search_features)

    def _reloc_scores_deep_features(self, search_features: Any) -> list[tuple[float, float]]:
        if int(search_features.shape[0]) == 0:
            return []
        if self.config.relocalize_use_init_only and self._template_feats:
            init_bank = (
                self._template_feat_banks[0]
                if self._template_feat_banks else [self._template_feats[0]]
            )
            bank_features = self.deep_extractor._torch.cat(init_bank, dim=0)
            bank_responses = self.similarity_head.forward_bank_to_searches(
                bank_features, search_features,
            )
            self.runtime_stats.response_maps += int(bank_responses.shape[0] * bank_responses.shape[1])
            self.runtime_stats.response_batches += 1
            bank_peaks = bank_responses.flatten(start_dim=3).max(dim=3).values.squeeze(-1)
            best_bank = bank_peaks.argmax(dim=1)
            search_indices = self.deep_extractor._torch.arange(
                int(search_features.shape[0]), device=search_features.device,
            )
            responses = bank_responses[search_indices, best_bank]
        else:
            responses, _ = self._fused_template_responses_batch(search_features)
        scores, _, metadata = self._response_scores_offsets_batch(responses, None, None)
        return [(score, meta["psr"]) for score, meta in zip(scores, metadata)]

    def _global_relocalize(
        self,
        frame: np.ndarray,
        predicted: SphereState,
        *,
        use_deep: bool | None = None,
    ) -> tuple[SphereState, float, float]:
        deep_search = self._deep_mode if use_deep is None else bool(use_deep and self._deep_mode)
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
        if deep_search:
            scales = list(self.config.deep_relocalize_scales)
        else:
            scales = [1.0]
            if self.config.relocalize_multi_scale:
                scales.extend(self.config.relocalize_extra_scales)

        best_overall_state = predicted
        best_overall_score = -1.0
        best_overall_psr = 0.0

        for reloc_scale in scales:
            reloc_state = SphereState(
                lon=predicted.lon,
                lat=predicted.lat,
                equatorial_width=predicted.equatorial_width * reloc_scale,
                angular_height=predicted.angular_height * reloc_scale,
            )

            if deep_search:
                fov_x, fov_y = state_size_to_fov(reloc_state, enlarge=self.config.deep_search_enlarge)
            else:
                fov_x, fov_y = self._handcrafted_match_fov(reloc_state)

            candidates: list[tuple[float, SphereState, float]] = []
            coarse_states: list[SphereState] = []
            coarse_patches: list[np.ndarray] = []

            for lon in lon_values_coarse:
                for lat in lat_values_coarse:
                    lat_val = clamp_lat(float(lat))
                    state = SphereState(
                        lon=float(lon), lat=lat_val,
                        equatorial_width=reloc_state.equatorial_width,
                        angular_height=reloc_state.angular_height,
                    )
                    if deep_search:
                        coarse_states.append(state)
                        continue
                    else:
                        patch = tangent_patch(
                            frame, float(lon), lat_val, fov_x, fov_y,
                            self.config.template_size, self.config.template_size,
                        )
                        score = self._score_patch(patch)
                    dist_lon = abs(lon_distance(float(lon), predicted.lon))
                    dist_lat = abs(lat_val - predicted.lat)
                    score -= 0.01 * dist_lon + 0.02 * dist_lat
                    candidates.append((score, state, float("inf")))

            if deep_search:
                coarse_patches = [
                    self._extract_search_patch(
                        frame, state.lon, state.lat, fov_x, fov_y, refine=False,
                    )
                    for state in coarse_states
                ]
                coarse_scores = self._reloc_scores_deep_batch(coarse_patches)
                for (score, psr), state in zip(coarse_scores, coarse_states):
                    dist_lon = abs(lon_distance(state.lon, predicted.lon))
                    dist_lat = abs(state.lat - predicted.lat)
                    candidates.append((score - 0.01 * dist_lon - 0.02 * dist_lat, state, psr))

            candidates.sort(key=lambda item: item[0], reverse=True)
            best_state = predicted
            best_score = -1.0
            best_psr = 0.0

            topk = (
                self.config.deep_relocalize_topk
                if deep_search else self.config.relocalize_topk
            )
            for coarse_score, coarse_state, coarse_psr in candidates[: max(int(topk), 1)]:
                if coarse_score < self._occlusion_threshold(handcrafted_result=not deep_search):
                    continue

                lon_range = np.deg2rad(np.arange(
                    -fine_stride_lon, fine_stride_lon + 1e-6, fine_stride_lon, dtype=np.float32
                ))
                lat_range = np.deg2rad(np.arange(
                    -fine_stride_lat, fine_stride_lat + 1e-6, fine_stride_lat, dtype=np.float32
                ))

                fine_candidates: list[tuple[float, SphereState, float]] = []
                fine_states: list[SphereState] = []
                fine_patches: list[np.ndarray] = []
                for dlon in lon_range:
                    for dlat in lat_range:
                        lon_val = wrap_lon(coarse_state.lon + dlon)
                        lat_val = clamp_lat(coarse_state.lat + dlat)
                        state = SphereState(
                            lon=float(lon_val), lat=float(lat_val),
                            equatorial_width=coarse_state.equatorial_width,
                            angular_height=coarse_state.angular_height,
                        )
                        if deep_search:
                            fine_states.append(state)
                            continue
                        else:
                            patch = tangent_patch(
                                frame, float(lon_val), float(lat_val), fov_x, fov_y,
                                self.config.template_size, self.config.template_size,
                            )
                            score = self._score_patch(patch)
                        fine_candidates.append((score, state, float("inf")))

                if deep_search:
                    fine_patches = [
                        self._extract_search_patch(
                            frame, state.lon, state.lat, fov_x, fov_y, refine=False,
                        )
                        for state in fine_states
                    ]
                    fine_scores = self._reloc_scores_deep_batch(fine_patches)
                    fine_candidates.extend(
                        (score, state, psr)
                        for (score, psr), state in zip(fine_scores, fine_states)
                    )

                if fine_candidates:
                    fine_candidates.sort(key=lambda item: item[0], reverse=True)
                    best_fine_state = fine_candidates[0][1]
                    best_fine_score = fine_candidates[0][0]
                    best_fine_psr = fine_candidates[0][2]
                else:
                    best_fine_state = coarse_state
                    best_fine_score = coarse_score
                    best_fine_psr = coarse_psr

                if deep_search:
                    refined_state, refined_score = self._local_search_deep(frame, best_fine_state)
                else:
                    refined_state, refined_score = self._local_search_handcrafted(frame, best_fine_state)
                final_score = max(refined_score, best_fine_score)
                final_psr = (
                    float(self.runtime_stats.last_psr)
                    if deep_search and refined_score >= best_fine_score
                    else best_fine_psr
                )

                if final_score > best_score:
                    best_state = refined_state
                    best_score = final_score
                    best_psr = final_psr

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_state = best_state
                best_overall_psr = best_psr

        return best_overall_state, best_overall_score, best_overall_psr

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
