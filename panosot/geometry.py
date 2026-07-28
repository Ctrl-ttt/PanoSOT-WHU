from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Tuple

import numpy as np


PI = math.pi
TWO_PI = 2.0 * math.pi
HALF_PI = 0.5 * math.pi


@dataclass
class SphereState:
    lon: float
    lat: float
    equatorial_width: float
    angular_height: float


def wrap_lon(lon: float | np.ndarray) -> float | np.ndarray:
    """Wrap longitude to [-pi, pi)."""
    return (lon + PI) % TWO_PI - PI


def clamp_lat(lat: float | np.ndarray, margin: float = 1e-4) -> float | np.ndarray:
    return np.clip(lat, -HALF_PI + margin, HALF_PI - margin)


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def xyz_to_lonlat(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True).clip(min=1e-8)
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    lat = np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0))
    return lon, lat


def erp_bbox_to_state(
    bbox_xywh: np.ndarray,
    image_width: int,
    image_height: int,
) -> SphereState:
    x, y, w, h = bbox_xywh.astype(np.float64)
    cx_px = x + 0.5 * w
    cy_px = y + 0.5 * h
    lon = wrap_lon(cx_px / image_width * TWO_PI - PI)
    lat = clamp_lat(HALF_PI - cy_px / image_height * PI)
    angular_width = max(w / image_width * TWO_PI, 1e-4)
    equatorial_width = angular_width * max(math.cos(float(lat)), 1e-3)
    angular_height = max(h / image_height * PI, 1e-4)
    return SphereState(
        lon=float(lon),
        lat=float(lat),
        equatorial_width=float(equatorial_width),
        angular_height=float(angular_height),
    )


def state_to_erp_bbox(
    state: SphereState,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    cos_lat = max(math.cos(state.lat), 0.01)
    angular_width = state.equatorial_width / cos_lat
    width_px = angular_width / TWO_PI * image_width
    height_px = state.angular_height / PI * image_height
    width_px = min(max(width_px, 10.0), float(image_width))
    height_px = min(max(height_px, 10.0), float(image_height))
    cx_px = (state.lon + PI) / TWO_PI * image_width
    cy_px = (HALF_PI - state.lat) / PI * image_height
    x = (cx_px - 0.5 * width_px) % image_width
    y = np.clip(cy_px - 0.5 * height_px, 0.0, image_height - height_px)
    return np.array([x, y, width_px, height_px], dtype=np.float32)


def lon_distance(a: float, b: float) -> float:
    return float(wrap_lon(a - b))


def state_size_to_fov(state: SphereState, enlarge: float = 2.0) -> Tuple[float, float]:
    angular_width = state.equatorial_width / max(math.cos(state.lat), 1e-3)
    fov_x = min(max(angular_width * enlarge, math.radians(8.0)), math.radians(150.0))
    fov_y = min(max(state.angular_height * enlarge, math.radians(8.0)), math.radians(120.0))
    return fov_x, fov_y


def _compute_tangent_grid(
    fov_x: float,
    fov_y: float,
    out_h: int,
    out_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算切平面投影的基础方向偏移量（不含中心点旋转）。"""
    u = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    delta_x = np.tan(uu * (0.5 * fov_x))
    delta_y = np.tan(vv * (0.5 * fov_y))
    return delta_x, delta_y


@lru_cache(maxsize=16)
def _cached_tangent_grid(
    fov_x_q: int,  # 量化后的整数 key（原始值 × 1000）
    fov_y_q: int,
    out_h: int,
    out_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """LRU 缓存的基础网格，避免同尺寸重复计算。"""
    return _compute_tangent_grid(fov_x_q / 1000.0, fov_y_q / 1000.0, out_h, out_w)


def tangent_patch(
    frame: np.ndarray,
    lon: float,
    lat: float,
    fov_x: float,
    fov_y: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Sample a locally rectified patch around a spherical center."""
    lon = float(wrap_lon(lon))
    lat = float(clamp_lat(lat))
    h, w = frame.shape[:2]

    # 使用缓存的基础网格（FoV 量化到 0.001 弧度精度）
    fov_x_q = int(round(fov_x * 1000))
    fov_y_q = int(round(fov_y * 1000))
    delta_x, delta_y = _cached_tangent_grid(fov_x_q, fov_y_q, out_h, out_w)

    center = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=np.float32,
    )
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float32)
    north = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=np.float32,
    )

    dirs = center + delta_x[..., None] * east + delta_y[..., None] * north
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
    sample_lon, sample_lat = xyz_to_lonlat(dirs)

    sample_x = (sample_lon + PI) / TWO_PI * w
    sample_y = (HALF_PI - sample_lat) / PI * h
    return bilinear_sample(frame, sample_x, sample_y)


def tangent_patches_batch(
    frame: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    fov_x: float,
    fov_y: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """批量提取多个球面中心点的切平面 patch。

    Args:
        frame: [H, W, 3] 图像
        lons: [N] 经度数组
        lats: [N] 纬度数组
        fov_x, fov_y: 视场角
        out_h, out_w: 输出尺寸

    Returns:
        [N, out_h, out_w, 3] 的 patch 数组
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lons = wrap_lon(lons)
    lats = clamp_lat(lats)
    h, w = frame.shape[:2]
    N = len(lons)

    # 使用缓存的基础网格
    fov_x_q = int(round(fov_x * 1000))
    fov_y_q = int(round(fov_y * 1000))
    delta_x, delta_y = _cached_tangent_grid(fov_x_q, fov_y_q, out_h, out_w)

    # 向量化计算 N 个中心点的方向向量
    cos_lats = np.cos(lats)
    sin_lats = np.sin(lats)
    cos_lons = np.cos(lons)
    sin_lons = np.sin(lons)

    centers = np.stack([cos_lats * cos_lons, cos_lats * sin_lons, sin_lats], axis=-1)  # [N, 3]
    easts = np.stack([-sin_lons, cos_lons, np.zeros(N)], axis=-1)  # [N, 3]
    norths = np.stack([
        -sin_lats * cos_lons,
        -sin_lats * sin_lons,
        cos_lats,
    ], axis=-1)  # [N, 3]

    # dirs: [N, out_h, out_w, 3]
    dirs = (
        centers[:, None, None, :]
        + delta_x[None, :, :, None] * easts[:, None, None, :]
        + delta_y[None, :, :, None] * norths[:, None, None, :]
    )
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)

    # 扁平化后统一做 xyz_to_lonlat
    dirs_flat = dirs.reshape(-1, 3)
    sample_lon, sample_lat = xyz_to_lonlat(dirs_flat)
    sample_x = (sample_lon + PI) / TWO_PI * w
    sample_y = (HALF_PI - sample_lat) / PI * h

    # 批量双线性插值
    sampled = bilinear_sample(frame, sample_x, sample_y)  # [N*out_h*out_w, 3]
    return sampled.reshape(N, out_h, out_w, 3).astype(np.float32)


def bilinear_sample(frame: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    xs = np.mod(xs, w).astype(np.float32)
    ys = np.clip(ys, 0.0, h - 1.001).astype(np.float32)

    x0 = np.floor(xs).astype(np.int32) % w
    y0 = np.floor(ys).astype(np.int32)
    x1 = (x0 + 1) % w
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xs - x0
    dy = ys - y0
    wa = (1.0 - dx) * (1.0 - dy)
    wb = dx * (1.0 - dy)
    wc = (1.0 - dx) * dy
    wd = dx * dy

    sampled = (
        frame[y0, x0] * wa[..., None]
        + frame[y0, x1] * wb[..., None]
        + frame[y1, x0] * wc[..., None]
        + frame[y1, x1] * wd[..., None]
    )
    return sampled.astype(np.float32)
