"""SiamX-style spherical backend for PanoSOT.

This module ports the reusable idea from HuajianUP/360Tracking rather than
copying its obsolete Python 3.7/CUDA extension stack.  The upstream tracker
uses an ``omni`` mode to rectify an ERP neighbourhood before Siamese
template-search matching.  PanoSOT already provides the spherical geometry,
seam-safe sampling, long-term recovery, and Docker contract, so this adapter
combines those pieces with the existing depthwise-XCorr implementation.

Upstream reference (MIT): https://github.com/HuajianUP/360Tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np

from .geometry import SphereState, state_size_to_fov, tangent_patch
from .tracker import PanoSOTTracker, TrackerConfig


@dataclass
class SiamX360Config:
    """Configuration for the ERP-aware SiamX compatibility backend."""

    # SiamX uses an enlarged local context for the template/search pair.
    template_context: float = 2.0
    search_context: float = 4.0
    # Keep enough room for rapid motion while preventing a full-ERP crop.
    max_horizontal_fov_deg: float = 150.0
    max_vertical_fov_deg: float = 120.0
    # Blend the SiamX-style local proposal with PanoSOT's long-term controller.
    local_proposal_weight: float = 0.72
    enabled: bool = True


class ERPRegionCropper:
    """Small, dependency-free equivalent of 360Tracking's omni cropper."""

    def __init__(self, config: SiamX360Config | None = None) -> None:
        self.config = config or SiamX360Config()

    def crop(self, frame: np.ndarray, state: SphereState, *, search: bool = False) -> np.ndarray:
        context = self.config.search_context if search else self.config.template_context
        fov_x, fov_y = state_size_to_fov(state, enlarge=context)
        fov_x = min(fov_x, np.deg2rad(self.config.max_horizontal_fov_deg))
        fov_y = min(fov_y, np.deg2rad(self.config.max_vertical_fov_deg))
        size = 224 if search else 127
        return self.crop_view(frame, state.lon, state.lat, fov_x, fov_y, size)

    def crop_view(
        self,
        frame: np.ndarray,
        lon: float,
        lat: float,
        fov_x: float,
        fov_y: float,
        size: int,
    ) -> np.ndarray:
        """Return a square, rectified ERP view for Siamese matching.

        PanoSOT's tangent sampler maps an ERP neighbourhood to a perspective
        view and wraps longitude, matching the geometric behavior of the
        upstream ``imgLookAt`` omni path while also remaining valid at poles.
        """
        fov_x = min(float(fov_x), np.deg2rad(self.config.max_horizontal_fov_deg))
        fov_y = min(float(fov_y), np.deg2rad(self.config.max_vertical_fov_deg))
        return tangent_patch(frame, lon, lat, fov_x, fov_y, int(size), int(size))


class SiamX360Tracker(PanoSOTTracker):
    """SiamX-style ERP tracker exposed through the PanoSOT API.

    It is a real :class:`PanoSOTTracker` subclass.  Every deep template and
    search patch flows through :class:`ERPRegionCropper`; the inherited
    controller remains responsible for confidence gating, seam-safe state
    updates, occlusion handling, and relocalization.
    """

    backend_name = "siamx360"

    def __init__(
        self,
        config: TrackerConfig | None = None,
        siamx_config: SiamX360Config | None = None,
        deep_extractor: Any = None,
        similarity_head: Any = None,
    ) -> None:
        self.siamx_config = siamx_config or SiamX360Config()
        self.cropper = ERPRegionCropper(self.siamx_config)
        tracker_config = config or TrackerConfig()
        # The upstream SiamX omni path is a learned template-search tracker.
        # Enable the existing deep branch and use its spherical search as the
        # production implementation compatible with Torch 2.x/CUDA 12.x.
        tracker_config.use_deep_features = True
        tracker_config.deep_search_enlarge = max(
            float(tracker_config.deep_search_enlarge),
            self.siamx_config.search_context,
        )
        tracker_config.deep_template_enlarge = max(
            float(tracker_config.deep_template_enlarge),
            self.siamx_config.template_context,
        )
        super().__init__(
            config=tracker_config,
            deep_extractor=deep_extractor,
            similarity_head=similarity_head,
        )

    def _extract_template_feat(self, frame: np.ndarray, state: SphereState) -> Any:
        size = self.config.deep_template_size
        fov_x, fov_y = state_size_to_fov(state, enlarge=self.config.deep_template_enlarge)
        patch = self.cropper.crop_view(frame, state.lon, state.lat, fov_x, fov_y, size)
        return self.deep_extractor.extract_template_feature(patch, assume_normalized=True)

    def _extract_template_feat_bank(
        self,
        frame: np.ndarray,
        state: SphereState,
        trust_location: bool = False,
    ) -> list[Any]:
        size = self.config.deep_template_size
        if self._is_small_target(state):
            enlarge = self.config.small_target_template_enlarge
        else:
            enlarge = self._get_template_enlarge(state.lat) if trust_location else self.config.deep_template_enlarge
        angles = self._get_rotation_angles(state.lat) if trust_location else self.config.deep_template_rotations_deg
        fov_x, fov_y = state_size_to_fov(state, enlarge=enlarge)
        patch = self.cropper.crop_view(frame, state.lon, state.lat, fov_x, fov_y, size)
        return [
            self.deep_extractor.extract_template_feature(
                self._rotate_patch(patch, angle_deg), assume_normalized=True,
            )
            for angle_deg in angles
        ]

    def _extract_search_patch(
        self,
        frame: np.ndarray,
        lon: float,
        lat: float,
        fov_x: float,
        fov_y: float,
        refine: bool = False,
        output_size: int | None = None,
    ) -> np.ndarray:
        size = int(output_size or (
            self.config.refine_search_size if refine else self.config.coarse_search_size
        ))
        return self.cropper.crop_view(frame, lon, lat, fov_x, fov_y, size)
