"""Tangent-plane (切平面) frontend for ERP panoramic frames.

Gnomonic projection of a local region around a spherical center (lon, lat).
This module complements panosot.geometry with the pieces the tracker backends
need:

- latitude-adaptive FOV (polar width compensation),
- pixel <-> (lon, lat) round-trip mapping for box back-projection,
- longitude-wrap-safe box unwrapping across the +-180 degree seam,
- batch patch extraction for global re-localization.

Angles in this module's public API are expressed in **degrees** for
readability; panosot.geometry.SphereState stays in radians.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .geometry import SphereState, clamp_lat, tangent_patch, wrap_lon

__all__ = [
    "adaptive_fov",
    "extract_patch",
    "extract_patches_batch",
    "lonlat_to_patch_pixel",
    "patch_pixel_to_lonlat",
    "patch_box_to_state",
    "patch_box_to_bfov",
    "state_to_patch_bbox",
    "unwrap_near",
    "DEG_TO_RAD",
    "RAD_TO_DEG",
]

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def adaptive_fov(
    state: SphereState,
    enlarge: float = 2.5,
    min_deg: float = 8.0,
    max_deg: float = 150.0,
    max_lat_deg: float = 120.0,
) -> Tuple[float, float]:
    """Latitude-adaptive tangent FOV (degrees) for a spherical state.

    state.equatorial_width is the angular width measured at the equator;
    dividing by cos(lat) already yields the pixel-projected ERP width, so the
    returned horizontal FOV grows toward the poles and keeps the target's
    apparent size roughly constant in patch pixels (polar width compensation).
    """
    cos_lat = max(math.cos(float(state.lat)), 1e-3)
    angular_width = float(state.equatorial_width) / cos_lat
    fov_x = min(
        max(angular_width * float(enlarge), math.radians(float(min_deg))),
        math.radians(float(max_deg)),
    )
    fov_y = min(
        max(float(state.angular_height) * float(enlarge), math.radians(float(min_deg))),
        math.radians(float(max_lat_deg)),
    )
    return float(fov_x * RAD_TO_DEG), float(fov_y * RAD_TO_DEG)


def extract_patch(
    frame: np.ndarray,
    lon_deg: float,
    lat_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Extract a locally rectified gnomonic patch around (lon, lat).

    Sampling is per-pixel with longitude wrap, so patches centered near the
    +-180 degree seam are rendered correctly in a single call.
    """
    return tangent_patch(
        frame,
        float(lon_deg) * DEG_TO_RAD,
        float(lat_deg) * DEG_TO_RAD,
        float(fov_x_deg) * DEG_TO_RAD,
        float(fov_y_deg) * DEG_TO_RAD,
        int(out_h),
        int(out_w),
    )


def _pixel_to_u_v(
    px: float | np.ndarray,
    py: float | np.ndarray,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    u = np.tan((px / max(out_w - 1, 1) * 2.0 - 1.0) * (0.5 * fov_x_deg * DEG_TO_RAD))
    v = np.tan((py / max(out_h - 1, 1) * 2.0 - 1.0) * (0.5 * fov_y_deg * DEG_TO_RAD))
    return u, v


def _basis(
    lon_deg: float, lat_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = float(lon_deg) * DEG_TO_RAD
    lat = float(lat_deg) * DEG_TO_RAD
    center = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    north = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=np.float64,
    )
    return center, east, north


def patch_pixel_to_lonlat(
    px: float | np.ndarray,
    py: float | np.ndarray,
    lon_c_deg: float,
    lat_c_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map patch pixel coordinates back to ERP lon/lat (degrees).

    The returned longitude is wrapped into [-180, 180).  Use unwrap_near
    relative to the patch center for seam-crossing boxes.
    """
    u, v = _pixel_to_u_v(px, py, fov_x_deg, fov_y_deg, out_h, out_w)
    center, east, north = _basis(lon_c_deg, lat_c_deg)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dirs = center + u[..., None] * east + v[..., None] * north
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
    lon = np.arctan2(dirs[..., 1], dirs[..., 0]) * RAD_TO_DEG
    lat = np.arcsin(np.clip(dirs[..., 2], -1.0, 1.0)) * RAD_TO_DEG
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat


def lonlat_to_patch_pixel(
    lon_pt_deg: float | np.ndarray,
    lat_pt_deg: float | np.ndarray,
    lon_c_deg: float,
    lat_c_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map ERP lon/lat (degrees) to continuous patch pixel coordinates.

    Points behind the tangent plane (dot product with the center direction
    <= 0) project to NaN so callers can mask them explicitly.
    """
    center, east, north = _basis(lon_c_deg, lat_c_deg)
    lon = np.asarray(lon_pt_deg, dtype=np.float64) * DEG_TO_RAD
    lat = np.asarray(lat_pt_deg, dtype=np.float64) * DEG_TO_RAD
    cos_lat = np.cos(lat)
    p = np.stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1
    )
    denom = np.einsum("...i,i->...", p, center)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.einsum("...i,i->...", p, east) / denom
        v = np.einsum("...i,i->...", p, north) / denom
    # The pixel grid parameterizes the angle linearly (geometry.tangent_patch
    # samples at tan(uu * fov/2) with uu = linspace(-1, 1)), so the tangent
    # coordinate maps back through atan before scaling by fov/2.
    px = (np.arctan(u) / (0.5 * fov_x_deg * DEG_TO_RAD) + 1.0) * 0.5 * (out_w - 1)
    py = (np.arctan(v) / (0.5 * fov_y_deg * DEG_TO_RAD) + 1.0) * 0.5 * (out_h - 1)
    return px, py


def unwrap_near(values: float | np.ndarray, ref_deg: float) -> float | np.ndarray:
    """Unwrap lon values so they stay within +-180 degrees of a reference.

    Boxes that cross the +-180 seam (e.g. left edge 170, right edge -172)
    become contiguous: -172 unwraps to 188 relative to ref 179.
    """
    values = np.asarray(values, dtype=np.float64)
    return values + 360.0 * np.round((float(ref_deg) - values) / 360.0)


def patch_box_to_state(
    bbox_xywh: np.ndarray | Sequence[float],
    lon_c_deg: float,
    lat_c_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
    min_width_rad: float = 0.02,
    min_height_rad: float = 0.02,
    max_width_rad: float = 1.5,
    max_height_rad: float = 1.2,
) -> SphereState:
    """Back-project a patch-space xywh box to a SphereState.

    Edges are mapped with the exact gnomonic inverse, so polar foreshortening
    is respected, and longitudes are unwrapped relative to the patch center so
    boxes crossing the +-180 seam produce a contiguous angular width.
    """
    x, y, w, h = [float(v) for v in np.asarray(bbox_xywh, dtype=np.float64).reshape(4)]
    w = max(float(w), 1e-3)
    h = max(float(h), 1e-3)
    cx_px = float(x + 0.5 * w)
    cy_px = float(y + 0.5 * h)

    lon_center, lat_center = patch_pixel_to_lonlat(
        cx_px, cy_px, lon_c_deg, lat_c_deg, fov_x_deg, fov_y_deg, out_h, out_w
    )
    lon_center = float(unwrap_near(float(lon_center), lon_c_deg))

    lon_left, lat_left = patch_pixel_to_lonlat(
        x, cy_px, lon_c_deg, lat_c_deg, fov_x_deg, fov_y_deg, out_h, out_w
    )
    lon_right, lat_right = patch_pixel_to_lonlat(
        x + w, cy_px, lon_c_deg, lat_c_deg, fov_x_deg, fov_y_deg, out_h, out_w
    )
    lon_left = float(unwrap_near(float(lon_left), lon_c_deg))
    lon_right = float(unwrap_near(float(lon_right), lon_c_deg))
    width_deg = abs(float(lon_right) - float(lon_left))

    _, lat_top = patch_pixel_to_lonlat(
        cx_px, y, lon_c_deg, lat_c_deg, fov_x_deg, fov_y_deg, out_h, out_w
    )
    _, lat_bottom = patch_pixel_to_lonlat(
        cx_px, y + h, lon_c_deg, lat_c_deg, fov_x_deg, fov_y_deg, out_h, out_w
    )
    height_deg = abs(float(lat_top) - float(lat_bottom))

    lat_center_rad = float(lat_center) * DEG_TO_RAD
    equatorial_width = float(
        width_deg * DEG_TO_RAD * max(math.cos(lat_center_rad), 1e-3)
    )
    angular_height = float(height_deg * DEG_TO_RAD)
    equatorial_width = float(np.clip(equatorial_width, min_width_rad, max_width_rad))
    angular_height = float(np.clip(angular_height, min_height_rad, max_height_rad))
    return SphereState(
        lon=float(wrap_lon(lon_center * DEG_TO_RAD)),
        lat=float(clamp_lat(lat_center * DEG_TO_RAD)),
        equatorial_width=equatorial_width,
        angular_height=angular_height,
    )


def state_to_patch_bbox(
    state: SphereState,
    lon_c_deg: float,
    lat_c_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Project a spherical state into a patch-space xywh box (pixels)."""
    lon_pt = float(unwrap_near(state.lon * RAD_TO_DEG, lon_c_deg))
    lat_pt = float(state.lat * RAD_TO_DEG)
    half_w = (
        float(state.equatorial_width) / max(math.cos(state.lat), 1e-3) * RAD_TO_DEG * 0.5
    )
    half_h = float(state.angular_height) * RAD_TO_DEG * 0.5
    # Gnomonic boxes are not axis-aligned on the sphere.  Build the patch box
    # the way a perfect center head would predict it: the box-center pixel
    # projects to the sphere center, and the edge midpoints project to the
    # sphere box edges.  patch_box_to_state inverts exactly those quantities.
    cx_px, cy_px = lonlat_to_patch_pixel(
        lon_pt,
        lat_pt,
        lon_c_deg,
        lat_c_deg,
        fov_x_deg,
        fov_y_deg,
        out_h,
        out_w,
    )
    left, _ = lonlat_to_patch_pixel(
        lon_pt - half_w,
        lat_pt,
        lon_c_deg,
        lat_c_deg,
        fov_x_deg,
        fov_y_deg,
        out_h,
        out_w,
    )
    right, _ = lonlat_to_patch_pixel(
        lon_pt + half_w,
        lat_pt,
        lon_c_deg,
        lat_c_deg,
        fov_x_deg,
        fov_y_deg,
        out_h,
        out_w,
    )
    # geometry.tangent_patch orients north toward larger py (rows grow
    # toward the south in the sampled patch), so the southern edge
    # (lat_pt - half_h) lands on the smaller row index.
    _, top_edge = lonlat_to_patch_pixel(
        lon_pt,
        lat_pt - half_h,
        lon_c_deg,
        lat_c_deg,
        fov_x_deg,
        fov_y_deg,
        out_h,
        out_w,
    )
    _, bottom_edge = lonlat_to_patch_pixel(
        lon_pt,
        lat_pt + half_h,
        lon_c_deg,
        lat_c_deg,
        fov_x_deg,
        fov_y_deg,
        out_h,
        out_w,
    )
    left = float(np.nan_to_num(np.asarray(left), nan=-1e4).item())
    top_edge = float(np.nan_to_num(np.asarray(top_edge), nan=-1e4).item())
    right = float(np.nan_to_num(np.asarray(right), nan=1e4).item())
    bottom_edge = float(np.nan_to_num(np.asarray(bottom_edge), nan=1e4).item())
    cx_px = float(np.nan_to_num(np.asarray(cx_px), nan=-1e4).item())
    cy_px = float(np.nan_to_num(np.asarray(cy_px), nan=-1e4).item())
    w = float(np.clip(right - left, 1.0, out_w))
    h = float(np.clip(bottom_edge - top_edge, 1.0, out_h))
    x = float(np.clip(cx_px - 0.5 * w, 0.0, out_w - 1.0))
    y = float(np.clip(cy_px - 0.5 * h, 0.0, out_h - 1.0))
    w = float(np.clip(w, 1.0, out_w - x))
    h = float(np.clip(h, 1.0, out_h - y))
    return np.asarray([x, y, w, h], dtype=np.float32)


def patch_box_to_bfov(
    bbox_xywh: np.ndarray | Sequence[float],
    lon_c_deg: float,
    lat_c_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Back-project a patch-space box to a BFoV [clon, clat, fov_h, fov_v]."""
    state = patch_box_to_state(
        bbox_xywh, lon_c_deg, lat_c_deg, fov_x_deg, fov_y_deg, out_h, out_w
    )
    angular_width = float(state.equatorial_width) / max(math.cos(state.lat), 1e-3)
    return np.asarray(
        [
            state.lon * RAD_TO_DEG,
            state.lat * RAD_TO_DEG,
            angular_width * RAD_TO_DEG,
            state.angular_height * RAD_TO_DEG,
        ],
        dtype=np.float32,
    )


def extract_patches_batch(
    frame: np.ndarray,
    centers: Iterable[Tuple[float, float]],
    fov_x_deg: float,
    fov_y_deg: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Extract gnomonic patches for many centers; returns [N, H, W, 3]."""
    patches = [
        extract_patch(frame, lon, lat, fov_x_deg, fov_y_deg, out_h, out_w)
        for lon, lat in centers
    ]
    if not patches:
        return np.zeros((0, out_h, out_w, 3), dtype=np.float32)
    return np.stack(patches, axis=0)
