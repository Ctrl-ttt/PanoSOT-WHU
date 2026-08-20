"""Tangent-plane frontend + OSTrack backend tracker.

This is the Phase 1+2 integration from the selection report:

1. every frame is processed through a local gnomonic (tangent-plane) window
   whose FOV adapts to latitude (polar width compensation),
2. the rectified template/search patches are fed to the single-stream OSTrack
   ViT, whose center head predicts score / offset / size maps,
3. the predicted patch-space box is back-projected onto the sphere with the
   exact gnomonic inverse (seam-safe), and a constant-velocity motion model
   recenters the search window between frames.

The public interface (initialize / track / track_sequence) mirrors
panosot.tracker.PanoSOTTracker so the backend can be swapped by the factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

from .geometry import (
    SphereState,
    clamp_lat,
    erp_bbox_to_state,
    state_to_erp_bbox,
    wrap_lon,
)
from .ostrack import (
    OSTrackConfig,
    OSTrackNet,
    OSTRACK_MEAN,
    OSTRACK_STD,
    build_ostrack,
    download_ostrack_384_weights,
    hann2d,
    load_ostrack_checkpoint,
    ostrack_384,
)
from .tangent import (
    RAD_TO_DEG,
    adaptive_fov,
    extract_patch,
    extract_patches_batch,
    patch_box_to_state,
)

__all__ = [
    "OstrackTrackerConfig",
    "PanoOSTrackTracker",
    "build_ostrack_tracker",
    "resolve_ostrack_weights",
]


@dataclass
class OstrackTrackerConfig:
    """Tracking-loop configuration for the tangent + OSTrack backend."""

    device: str = "auto"
    # Tangent window factors (OSTrack defaults: template 2x, search 4x).
    template_enlarge: float = 2.0
    search_enlarge: float = 4.0
    min_fov_deg: float = 8.0
    max_fov_deg: float = 150.0
    max_lat_fov_deg: float = 120.0
    # Motion model.
    motion_momentum: float = 0.7
    # Hann window: 1.0 multiplies the score map by the window directly
    # (official OSTrack behavior); lower values mix in the raw score.
    window_influence: float = 1.0
    # Confidence gates.
    score_threshold: float = 0.25
    velocity_decay: float = 0.5
    # Size smoothing and bounds (relative to the initialization size).
    size_ema: float = 0.35
    min_state_ratio: float = 0.2
    max_state_ratio: float = 5.0
    # Template refresh (LaSOT-style). 0 disables updates.
    template_update_interval: int = 0
    template_update_threshold: float = 0.5
    # Coarse global re-localization after consecutive low-confidence frames.
    relocalize_enabled: bool = True
    relocalize_trigger_lost_frames: int = 5
    relocalize_accept_score: float = 0.30
    relocalize_stride_deg: float = 24.0
    relocalize_lat_stride_deg: float = 18.0
    relocalize_lat_range_deg: float = 60.0
    relocalize_scales: tuple[float, ...] = (1.0, 1.3)
    relocalize_cooldown_frames: int = 10
    relocalize_chunk_size: int = 8


class PanoOSTrackTracker:
    """ERP panorama tracker: tangent-plane frontend + OSTrack backend."""

    def __init__(
        self,
        model: OSTrackNet,
        config: Optional[OstrackTrackerConfig] = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for the OSTrack backend.") from exc
        self._torch = torch
        self.config = config or OstrackTrackerConfig()
        self.net = model
        self.net.eval()

        self.device = self._resolve_device(self.config.device)
        self.net.to(self.device)
        self._mean = torch.tensor(
            OSTRACK_MEAN, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            OSTRACK_STD, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self._hann = hann2d(
            int(model.config.feat_sz), centered=True, device=self.device
        )

        self.state: Optional[SphereState] = None
        self._init_state: Optional[SphereState] = None
        self.frame_shape: Optional[tuple[int, int]] = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.frame_count = 0
        self.lost_frames = 0
        self.initialized = False
        self._last_relocalize_frame = -10**9

    def _resolve_device(self, requested: str) -> Any:
        torch = self._torch
        normalized = requested.strip().lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(normalized)

    # ------------------------------------------------------------------ #
    # Patch preparation
    # ------------------------------------------------------------------ #

    def _preprocess(self, patch: np.ndarray) -> Any:
        """Convert a float [0, 1] RGB patch to a normalized [1, 3, H, W] tensor."""
        torch = self._torch
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        return (tensor - self._mean) / self._std

    def _preprocess_batch(self, patches: np.ndarray) -> Any:
        torch = self._torch
        if patches.dtype != np.float32:
            patches = patches.astype(np.float32)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(patches))
            .permute(0, 3, 1, 2)
            .to(self.device)
        )
        return (tensor - self._mean) / self._std

    def _template_fov(self, state: SphereState) -> tuple[float, float]:
        return adaptive_fov(
            state,
            enlarge=self.config.template_enlarge,
            min_deg=self.config.min_fov_deg,
            max_deg=self.config.max_fov_deg,
            max_lat_deg=self.config.max_lat_fov_deg,
        )

    def _search_fov(self, state: SphereState) -> tuple[float, float]:
        return adaptive_fov(
            state,
            enlarge=self.config.search_enlarge,
            min_deg=self.config.min_fov_deg,
            max_deg=self.config.max_fov_deg,
            max_lat_deg=self.config.max_lat_fov_deg,
        )

    def _extract_template(self, frame: np.ndarray, state: SphereState) -> np.ndarray:
        fov_x, fov_y = self._template_fov(state)
        size = int(self.net.config.template_size)
        return extract_patch(
            frame,
            float(state.lon) * RAD_TO_DEG,
            float(state.lat) * RAD_TO_DEG,
            fov_x,
            fov_y,
            size,
            size,
        )

    def _refresh_template(self, frame: np.ndarray, state: SphereState) -> None:
        with self._torch.inference_mode():
            self.net.initialize(self._preprocess(self._extract_template(frame, state)))

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, init_bbox_xywh: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        self.state = erp_bbox_to_state(init_bbox_xywh, w, h)
        self._init_state = SphereState(
            lon=self.state.lon,
            lat=self.state.lat,
            equatorial_width=self.state.equatorial_width,
            angular_height=self.state.angular_height,
        )
        self.velocity[:] = 0.0
        self.frame_count = 0
        self.lost_frames = 0
        self._last_relocalize_frame = -10**9
        self._refresh_template(frame, self.state)
        self.initialized = True
        return self._state_to_bbox(self.state)

    def track(self, frame: np.ndarray) -> np.ndarray:
        if self.state is None or self.frame_shape is None:
            raise RuntimeError("PanoOSTrackTracker.initialize must be called first.")
        torch = self._torch
        self.frame_count += 1

        pred_lon = float(wrap_lon(self.state.lon + float(self.velocity[0])))
        pred_lat = float(clamp_lat(self.state.lat + float(self.velocity[1])))

        candidate, score = self._local_track(frame, pred_lon, pred_lat)
        accepted = bool(score >= float(self.config.score_threshold))

        if accepted:
            measured_lon = float(wrap_lon(candidate.lon - self.state.lon))
            measured_lat = float(candidate.lat - self.state.lat)
            momentum = float(np.clip(self.config.motion_momentum, 0.0, 1.0))
            self.velocity[0] = momentum * self.velocity[0] + (1.0 - momentum) * measured_lon
            self.velocity[1] = momentum * self.velocity[1] + (1.0 - momentum) * measured_lat
            self._adopt_candidate(candidate)
            self.lost_frames = 0
            interval = int(self.config.template_update_interval)
            if interval > 0 and self.frame_count % interval == 0:
                if score >= float(self.config.template_update_threshold):
                    self._refresh_template(frame, self.state)
        else:
            self.lost_frames += 1
            self.velocity *= float(np.clip(self.config.velocity_decay, 0.0, 1.0))

        if (
            self.config.relocalize_enabled
            and self.lost_frames >= int(self.config.relocalize_trigger_lost_frames)
            and self.frame_count - self._last_relocalize_frame
            >= int(self.config.relocalize_cooldown_frames)
        ):
            self._last_relocalize_frame = self.frame_count
            recovered = self._global_relocalize(frame)
            if recovered:
                self.lost_frames = 0

        return self._state_to_bbox(self.state)

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

    # ------------------------------------------------------------------ #
    # Core inference
    # ------------------------------------------------------------------ #

    def _local_track(
        self,
        frame: np.ndarray,
        pred_lon: float,
        pred_lat: float,
    ) -> tuple[SphereState, float]:
        torch = self._torch
        state = self.state
        fov_x, fov_y = self._search_fov(state)
        size = int(self.net.config.search_size)
        patch = extract_patch(
            frame,
            float(pred_lon) * RAD_TO_DEG,
            float(pred_lat) * RAD_TO_DEG,
            fov_x,
            fov_y,
            size,
            size,
        )
        x = self._preprocess(patch)
        with torch.inference_mode():
            out = self.net.forward_search(x)
            score_map = out["score_map"]
            influence = float(np.clip(self.config.window_influence, 0.0, 1.0))
            if influence >= 1.0:
                response = score_map * self._hann
            elif influence <= 0.0:
                response = score_map
            else:
                response = score_map * (1.0 - influence) + score_map * self._hann * influence
            bbox_norm, best = self.net.box_head.cal_bbox(
                response, out["size_map"], out["offset_map"], return_score=True
            )
        cx, cy, bw, bh = [float(v) for v in bbox_norm[0].detach().cpu().numpy()]
        px, py = cx * size, cy * size
        pw, ph = max(bw * size, 1.0), max(bh * size, 1.0)
        candidate = patch_box_to_state(
            [px - 0.5 * pw, py - 0.5 * ph, pw, ph],
            float(pred_lon) * RAD_TO_DEG,
            float(pred_lat) * RAD_TO_DEG,
            fov_x,
            fov_y,
            size,
            size,
        )
        return candidate, float(best.detach().cpu().flatten()[0])

    def _adopt_candidate(self, candidate: SphereState) -> None:
        state = self.state
        sema = float(np.clip(self.config.size_ema, 0.0, 1.0))
        eqw = (1.0 - sema) * state.equatorial_width + sema * candidate.equatorial_width
        anh = (1.0 - sema) * state.angular_height + sema * candidate.angular_height
        if self._init_state is not None:
            lo = float(self.config.min_state_ratio)
            hi = float(self.config.max_state_ratio)
            eqw = float(np.clip(eqw, lo * self._init_state.equatorial_width, hi * self._init_state.equatorial_width))
            anh = float(np.clip(anh, lo * self._init_state.angular_height, hi * self._init_state.angular_height))
        self.state = SphereState(
            lon=float(candidate.lon),
            lat=float(candidate.lat),
            equatorial_width=float(eqw),
            angular_height=float(anh),
        )

    def _global_relocalize(self, frame: np.ndarray) -> bool:
        """Coarse lon/lat grid scan using the cached template tokens."""
        torch = self._torch
        if self._init_state is None or self.state is None:
            return False
        lons = list(
            np.arange(
                -180.0,
                180.0 - 1e-6,
                float(self.config.relocalize_stride_deg),
            )
        )
        lat_range = float(self.config.relocalize_lat_range_deg)
        lats = list(
            np.arange(
                -lat_range,
                lat_range + 1e-6,
                float(self.config.relocalize_lat_stride_deg),
            )
        )
        centers = [(float(lon), float(lat)) for lon in lons for lat in lats]

        size = int(self.net.config.search_size)
        best_score = float(self.config.relocalize_accept_score)
        best_candidate: Optional[SphereState] = None

        with torch.inference_mode():
            for scale in self.config.relocalize_scales:
                state_for_fov = SphereState(
                    lon=self._init_state.lon,
                    lat=self._init_state.lat,
                    equatorial_width=self._init_state.equatorial_width * float(scale),
                    angular_height=self._init_state.angular_height * float(scale),
                )
                fov_x, fov_y = self._search_fov(state_for_fov)
                chunk = int(max(self.config.relocalize_chunk_size, 1))
                for start in range(0, len(centers), chunk):
                    group = centers[start : start + chunk]
                    patches = extract_patches_batch(
                        frame, group, fov_x, fov_y, size, size
                    )
                    x = self._preprocess_batch(patches)
                    out = self.net.forward_search(x)
                    score_map = out["score_map"]
                    bboxes, scores = self.net.box_head.cal_bbox(
                        score_map, out["size_map"], out["offset_map"], return_score=True
                    )
                    scores_np = scores.detach().cpu().numpy().reshape(-1)
                    for local_idx, (lon, lat) in enumerate(group):
                        score_v = float(scores_np[local_idx])
                        if score_v <= best_score:
                            continue
                        cx, cy, bw, bh = [
                            float(v)
                            for v in bboxes[local_idx].detach().cpu().numpy()
                        ]
                        px, py = cx * size, cy * size
                        pw, ph = max(bw * size, 1.0), max(bh * size, 1.0)
                        cand = patch_box_to_state(
                            [px - 0.5 * pw, py - 0.5 * ph, pw, ph],
                            lon,
                            lat,
                            fov_x,
                            fov_y,
                            size,
                            size,
                        )
                        if score_v > best_score:
                            best_score = score_v
                            best_candidate = cand

        if best_candidate is None:
            return False
        self.state = best_candidate
        self.velocity[:] = 0.0
        self._refresh_template(frame, self.state)
        return True

    def _state_to_bbox(self, state: SphereState) -> np.ndarray:
        h, w = self.frame_shape
        return state_to_erp_bbox(state, w, h)


def resolve_ostrack_weights(
    weights_path: str | Path | None,
    cache_dir: Optional[str | Path] = None,
    allow_download: bool = True,
) -> Path | None:
    """Resolve a weights path; None means random initialization."""
    if weights_path is None:
        return None
    if str(weights_path) == "auto":
        if cache_dir is None:
            project_cache = Path(__file__).resolve().parents[1] / ".cache" / "ostrack"
            cache_dir = str(project_cache) if project_cache.is_dir() else None
        candidate = None
        if cache_dir is not None:
            p = Path(cache_dir) / "OSTrack_vitb_384_mae_ce_32x4_ep300.safetensors"
            if p.is_file():
                candidate = p
        if candidate is None and allow_download:
            return download_ostrack_384_weights(cache_dir)
        return candidate
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"OSTrack weights not found: {path}")
    return path


def build_ostrack_tracker(
    variant: str = "384",
    weights_path: str | Path | None = "auto",
    device: str = "auto",
    cache_dir: Optional[str | Path] = None,
    allow_download: bool = True,
    tracker_kwargs: Optional[dict[str, Any]] = None,
) -> PanoOSTrackTracker:
    """Build a tangent + OSTrack tracker.

    weights_path accepts an explicit path, "auto" (project cache with optional
    download) or None (random initialization, for unit tests).
    """
    model = build_ostrack(variant)
    path = resolve_ostrack_weights(weights_path, cache_dir=cache_dir, allow_download=allow_download)
    if path is not None:
        missing, unexpected = load_ostrack_checkpoint(model, path, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"OSTrack checkpoint did not load cleanly: missing={missing} "
                f"unexpected={unexpected}"
            )
    else:
        import warnings

        warnings.warn(
            "Building OSTrack tracker without pretrained weights "
            "(random initialization, inference-only for smoke tests).",
            RuntimeWarning,
            stacklevel=2,
        )
    config = OstrackTrackerConfig(device=device)
    if tracker_kwargs:
        for key, value in tracker_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown OstrackTrackerConfig field: {key}")
    return PanoOSTrackTracker(model=model, config=config)
