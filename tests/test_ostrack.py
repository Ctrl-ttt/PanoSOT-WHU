"""Tests for the tangent-plane frontend + OSTrack backend upgrade.

Coverage:
- tangent-plane geometry (pixel <-> lon/lat round trip, polar FOV growth,
  seam unwrapping, seam-crossing patch extraction, box back-projection),
- OSTrack network structure (256/384 forward shapes, cal_bbox conventions,
  hann window, checkpoint key remapping),
- PanoOSTrackTracker loop with a deterministic stub network (end-to-end
  patch -> sphere -> ERP round trip on a synthetic moving blob),
- real-weight loading smoke test (slow; only when the safetensors file and
  PANOSOT_RUN_SLOW are present).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from panosot.geometry import SphereState, erp_bbox_to_state, state_to_erp_bbox
from panosot.ostrack import (
    build_ostrack,
    hann2d,
    remap_checkpoint_keys,
)
from panosot import tangent as tangent_mod
from panosot.tangent import (
    adaptive_fov,
    extract_patch,
    lonlat_to_patch_pixel,
    patch_box_to_state,
    patch_pixel_to_lonlat,
    state_to_patch_bbox,
    unwrap_near,
)

torch = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
OSTRACK_384_FILE = (
    PROJECT_DIR / ".cache" / "ostrack" / "OSTrack_vitb_384_mae_ce_32x4_ep300.safetensors"
)
RUN_SLOW = os.environ.get("PANOSOT_RUN_SLOW") == "1"


def make_erp_frames(
    n: int,
    width: int = 320,
    height: int = 160,
    lon0_deg: float = -30.0,
    lat0_deg: float = 10.0,
    dlon_per_frame: float = 2.0,
    radius_deg: float = 6.0,
) -> tuple[list[np.ndarray], list[SphereState]]:
    """Synthetic ERP frames with a spherical blob moving in longitude."""
    lons_deg = np.linspace(0.0, 360.0, width, endpoint=False) - 180.0
    lats_deg = np.linspace(90.0, -90.0, height)
    lon_grid = np.radians(lons_deg)
    lat_grid = np.radians(lats_deg)
    cos_radius = math.cos(math.radians(radius_deg))
    frames: list[np.ndarray] = []
    states: list[SphereState] = []
    for t in range(n):
        lon_t = lon0_deg + t * dlon_per_frame
        lat_t = lat0_deg
        lat_t_r = math.radians(lat_t)
        cosd = (
            np.sin(lat_grid)[:, None] * math.sin(lat_t_r)
            + np.cos(lat_grid)[:, None]
            * math.cos(lat_t_r)
            * np.cos(lon_grid[None, :] - math.radians(lon_t))
        )
        mask = cosd > cos_radius
        img = np.zeros((height, width, 3), dtype=np.float32)
        img[mask] = 1.0
        frames.append(img)
        states.append(
            SphereState(
                lon=math.radians(lon_t),
                lat=lat_t_r,
                equatorial_width=math.radians(2.0 * radius_deg) * math.cos(lat_t_r),
                angular_height=math.radians(2.0 * radius_deg),
            )
        )
    return frames, states


class TangentFrontendTests(unittest.TestCase):
    def test_pixel_lonlat_round_trip(self) -> None:
        cases = [
            (0.0, 0.0, 60.0, 60.0, 64, 64),
            (179.0, 60.0, 90.0, 60.0, 96, 96),
            (-45.0, -35.0, 120.0, 100.0, 128, 128),
        ]
        for lon_c, lat_c, fov_x, fov_y, out_h, out_w in cases:
            for px in (0.0, out_w / 2.0, out_w - 1.0):
                for py in (0.0, out_h / 2.0, out_h - 1.0):
                    lon_pt, lat_pt = patch_pixel_to_lonlat(
                        px, py, lon_c, lat_c, fov_x, fov_y, out_h, out_w
                    )
                    px2, py2 = lonlat_to_patch_pixel(
                        float(lon_pt),
                        float(lat_pt),
                        lon_c,
                        lat_c,
                        fov_x,
                        fov_y,
                        out_h,
                        out_w,
                    )
                    self.assertAlmostEqual(float(px2), px, delta=0.02)
                    self.assertAlmostEqual(float(py2), py, delta=0.02)

    def test_patch_center_maps_to_patch_center(self) -> None:
        for lon_c, lat_c in ((0.0, 0.0), (179.0, 65.0), (-90.0, -40.0)):
            # The continuous grid center of an N-pixel patch is (N-1)/2.
            c = 95.0 / 2.0
            lon, lat = patch_pixel_to_lonlat(c, c, lon_c, lat_c, 80.0, 80.0, 96, 96)
            self.assertAlmostEqual(float(lon), lon_c, delta=1e-3)
            self.assertAlmostEqual(float(lat), lat_c, delta=1e-3)

    def test_adaptive_fov_grows_toward_poles(self) -> None:
        state_eq = SphereState(
            lon=0.0, lat=0.0,
            equatorial_width=math.radians(20.0), angular_height=math.radians(20.0),
        )
        state_polar = SphereState(
            lon=0.0, lat=math.radians(70.0),
            equatorial_width=math.radians(20.0), angular_height=math.radians(20.0),
        )
        fov_eq = adaptive_fov(state_eq, enlarge=2.0)
        fov_polar = adaptive_fov(state_polar, enlarge=2.0)
        self.assertGreater(fov_polar[0], fov_eq[0] * 2.0)
        self.assertAlmostEqual(fov_polar[1], fov_eq[1], delta=1e-4)

    def test_unwrap_near_seam(self) -> None:
        self.assertAlmostEqual(float(unwrap_near(-172.0, 179.0)), 188.0, delta=1e-6)
        self.assertAlmostEqual(float(unwrap_near(170.0, 179.0)), 170.0, delta=1e-6)
        self.assertAlmostEqual(float(unwrap_near(-172.0, -170.0)), -172.0, delta=1e-6)

    def test_patch_box_to_state_round_trip(self) -> None:
        cases = [
            (0.0, 0.0, 80.0, 80.0, 96),
            (179.0, 50.0, 90.0, 70.0, 128),
            (-120.0, -20.0, 110.0, 90.0, 128),
        ]
        for lon_c, lat_c, fov_x, fov_y, size in cases:
            state = SphereState(
                lon=math.radians(lon_c),
                lat=math.radians(lat_c),
                equatorial_width=math.radians(18.0),
                angular_height=math.radians(14.0),
            )
            bbox = state_to_patch_bbox(state, lon_c, lat_c, fov_x, fov_y, size, size)
            recovered = patch_box_to_state(bbox, lon_c, lat_c, fov_x, fov_y, size, size)
            lon_err = abs(
                math.degrees(
                    (recovered.lon - state.lon + math.pi) % (2 * math.pi) - math.pi
                )
            )
            self.assertLess(lon_err, 2.0)
            self.assertLess(abs(math.degrees(recovered.lat - state.lat)), 2.0)
            self.assertLess(
                abs(recovered.equatorial_width - state.equatorial_width)
                / state.equatorial_width,
                0.25,
            )
            self.assertLess(
                abs(recovered.angular_height - state.angular_height)
                / state.angular_height,
                0.25,
            )

    def test_patch_box_to_state_crossing_seam(self) -> None:
        # Box whose right edge crosses +180 (maps to negative wrapped lon).
        lon_c = 179.0
        lat_c = 0.0
        fov_x, fov_y, size = 60.0, 60.0, 96
        state = SphereState(
            lon=math.radians(179.0),
            lat=0.0,
            equatorial_width=math.radians(10.0),
            angular_height=math.radians(10.0),
        )
        bbox = state_to_patch_bbox(state, lon_c, lat_c, fov_x, fov_y, size, size)
        recovered = patch_box_to_state(bbox, lon_c, lat_c, fov_x, fov_y, size, size)
        lon_err = abs(
            math.degrees(
                (recovered.lon - state.lon + math.pi) % (2 * math.pi) - math.pi
            )
        )
        self.assertLess(lon_err, 2.0)

    def test_extract_patch_across_seam(self) -> None:
        width, height = 320, 160
        frame = np.zeros((height, width, 3), dtype=np.float32)
        frame[:, 0:8] = 1.0  # vertical stripe at the seam
        patch = extract_patch(frame, 179.0, 0.0, 40.0, 40.0, 32, 32)
        self.assertGreater(float(patch.max()), 0.5)

    def test_extract_patch_batch_matches_single(self) -> None:
        width, height = 320, 160
        rng = np.random.default_rng(0)
        frame = rng.random((height, width, 3), dtype=np.float32)
        centers = [(0.0, 0.0), (90.0, 30.0), (-90.0, -30.0)]
        batch = tangent_mod.extract_patches_batch(
            frame, centers, 60.0, 60.0, 32, 32
        )
        self.assertEqual(batch.shape, (3, 32, 32, 3))
        for i, (lon, lat) in enumerate(centers):
            single = extract_patch(frame, lon, lat, 60.0, 60.0, 32, 32)
            self.assertLess(float(np.abs(batch[i] - single).max()), 1e-5)


@unittest.skipIf(torch is None, "torch not installed")
class OstrackModelTests(unittest.TestCase):
    def test_256_forward_shapes(self) -> None:
        model = build_ostrack("256")
        model.eval()
        z = torch.randn(1, 3, 128, 128)
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model.forward(z, x)
        self.assertEqual(tuple(out["score_map"].shape), (1, 1, 16, 16))
        self.assertEqual(tuple(out["size_map"].shape), (1, 2, 16, 16))
        self.assertEqual(tuple(out["offset_map"].shape), (1, 2, 16, 16))
        score = out["score_map"]
        self.assertGreaterEqual(float(score.min()), 1e-4)
        self.assertLessEqual(float(score.max()), 1.0 - 1e-4)
        self.assertTrue(torch.isfinite(out["size_map"]).all())
        self.assertTrue(torch.isfinite(out["offset_map"]).all())

    def test_384_forward_shapes_small_batch(self) -> None:
        model = build_ostrack("384")
        model.eval()
        z = torch.randn(2, 3, 192, 192)
        x = torch.randn(2, 3, 384, 384)
        with torch.inference_mode():
            out = model.forward(z, x)
        self.assertEqual(tuple(out["score_map"].shape), (2, 1, 24, 24))
        self.assertEqual(tuple(out["size_map"].shape), (2, 2, 24, 24))

    def test_initialize_then_forward_search_matches(self) -> None:
        model = build_ostrack("256")
        model.eval()
        z = torch.randn(1, 3, 128, 128)
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out_a = model.forward(z, x)
            model.initialize(z)
            out_b = model.forward_search(x)
        self.assertTrue(
            torch.allclose(out_a["score_map"], out_b["score_map"], atol=1e-6)
        )

    def test_forward_search_broadcasts_template_to_batched_search(self) -> None:
        """Grid relocalization batches many search patches against one template."""
        model = build_ostrack("256")
        model.eval()
        z = torch.randn(1, 3, 128, 128)
        x_batch = torch.randn(8, 3, 256, 256)
        with torch.inference_mode():
            model.initialize(z)
            out_batch = model.forward_search(x_batch)
        self.assertEqual(tuple(out_batch["score_map"].shape), (8, 1, 16, 16))
        with torch.inference_mode():
            out_single = model.forward_search(x_batch[:1])
        self.assertTrue(
            torch.allclose(out_batch["score_map"][:1], out_single["score_map"], atol=1e-6)
        )

    def test_cal_bbox_convention(self) -> None:
        model = build_ostrack("256")
        model.eval()
        z = torch.randn(1, 3, 128, 128)
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model.forward(z, x)
        bbox, best = model.box_head.cal_bbox(
            out["score_map"], out["size_map"], out["offset_map"], return_score=True
        )
        bbox = bbox[0].numpy()
        cx, cy, w, h = bbox
        # Raw offset outputs are unbounded (cell units); allow a one-cell margin.
        self.assertGreaterEqual(cx, -0.5)
        self.assertLessEqual(cx, 1.5)
        self.assertGreaterEqual(cy, -0.5)
        self.assertLessEqual(cy, 1.5)
        # With size_sigmoid=True the normalized size is bounded in [1e-4, 1).
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0)
        self.assertTrue(np.all(np.isfinite(bbox)))
        self.assertGreaterEqual(float(best[0]), 1e-4)

    def test_track_window_matches_manual(self) -> None:
        model = build_ostrack("256")
        model.eval()
        z = torch.randn(1, 3, 128, 128)
        x = torch.randn(1, 3, 256, 256)
        window = hann2d(16, centered=True)
        with torch.inference_mode():
            bbox, _, _ = model.track(z, x, window=window)
            out = model.forward(z, x)
            manual, _ = model.box_head.cal_bbox(
                out["score_map"] * window,
                out["size_map"],
                out["offset_map"],
                return_score=True,
            )
        self.assertTrue(torch.allclose(bbox, manual, atol=1e-6))

    def test_hann_window_peaks_at_center(self) -> None:
        window = hann2d(16, centered=True)
        self.assertEqual(tuple(window.shape), (1, 1, 16, 16))
        self.assertAlmostEqual(float(window.max()), 1.0, places=5)
        self.assertAlmostEqual(float(window[0, 0, 8, 8]), float(window.max()), places=5)
        self.assertLess(float(window[0, 0, 0, 0]), 1e-3)

    def test_remap_checkpoint_keys(self) -> None:
        dummy = torch.zeros(3)
        remapped = remap_checkpoint_keys(
            {
                "net.backbone.blocks.0.norm1.weight": dummy,
                "pos_embed_z": dummy,
                "pos_embed_x": dummy,
                "box_head.conv5_ctr.weight": dummy,
            }
        )
        self.assertIn("backbone.blocks.0.norm1.weight", remapped)
        self.assertIn("backbone.pos_embed_z", remapped)
        self.assertIn("backbone.pos_embed_x", remapped)
        self.assertIn("box_head.conv5_ctr.weight", remapped)
        self.assertNotIn("net.backbone.blocks.0.norm1.weight", remapped)


class StubHead:
    def __init__(self, feat_sz: int) -> None:
        self.feat_sz = feat_sz

    def cal_bbox(self, score_map, size_map, offset_map, return_score=False):
        flat = score_map.flatten(1)
        best, idx = flat.max(dim=1)
        ix = idx % self.feat_sz
        iy = idx // self.feat_sz
        size = size_map.flatten(2)[:, :, idx].squeeze(-1)
        offset = offset_map.flatten(2)[:, :, idx].squeeze(-1)
        cx = (ix.float() + offset[:, 0]) / self.feat_sz
        cy = (iy.float() + offset[:, 1]) / self.feat_sz
        wh = size / self.feat_sz
        bbox = torch.stack([cx, cy, wh[:, 0], wh[:, 1]], dim=1)
        if return_score:
            return bbox, best
        return bbox


class StubNet:
    """Deterministic stand-in for OSTrackNet that peaks at the GT location."""

    def __init__(self, holder: dict, search_size: int = 256) -> None:
        self.holder = holder
        self.search_size = search_size
        self.config = SimpleNamespace(
            template_size=128,
            search_size=search_size,
            feat_sz=search_size // 16,
            patch_size=16,
        )
        self.box_head = StubHead(self.config.feat_sz)
        self._z = None

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def initialize(self, z):
        self._z = z
        return z

    def forward_search(self, x, z_tokens=None):
        gt = self.holder["gt_state"]
        lon_c, lat_c, fov_x, fov_y = self.holder["meta"]
        size = self.search_size
        feat = self.config.feat_sz
        bbox_px = state_to_patch_bbox(
            gt, lon_c, lat_c, fov_x, fov_y, size, size
        )
        cx_px = bbox_px[0] + 0.5 * bbox_px[2]
        cy_px = bbox_px[1] + 0.5 * bbox_px[3]
        cell_x = cx_px / size * feat
        cell_y = cy_px / size * feat
        ix = int(np.floor(np.clip(cell_x, 0, feat - 1)))
        iy = int(np.floor(np.clip(cell_y, 0, feat - 1)))
        score = torch.zeros(1, 1, feat, feat, dtype=torch.float32)
        score[0, 0, iy, ix] = 0.95
        offset = torch.zeros(1, 2, feat, feat, dtype=torch.float32)
        offset[0, 0, iy, ix] = float(cell_x - ix)
        offset[0, 1, iy, ix] = float(cell_y - iy)
        size_map = torch.zeros(1, 2, feat, feat, dtype=torch.float32)
        size_map[0, 0, iy, ix] = float(bbox_px[2] / size * feat)
        size_map[0, 1, iy, ix] = float(bbox_px[3] / size * feat)
        return {
            "score_map": score,
            "offset_map": offset,
            "size_map": size_map,
        }


@unittest.skipIf(torch is None, "torch not installed")
class OstrackTrackerStubTests(unittest.TestCase):
    def _make_tracker(self, frames, states):
        from panosot.ostrack_tracker import (
            OstrackTrackerConfig,
            PanoOSTrackTracker,
        )

        holder: dict = {"gt_state": states[0], "meta": (0.0, 0.0, 60.0, 60.0)}
        net = StubNet(holder, search_size=256)
        config = OstrackTrackerConfig(
            device="cpu",
            relocalize_enabled=False,
            score_threshold=0.2,
            size_ema=1.0,
        )
        tracker = PanoOSTrackTracker(model=net, config=config)

        real_extract = tangent_mod.extract_patch
        frame_index = {"t": 0}

        def spy_extract(frame, lon_deg, lat_deg, fov_x_deg, fov_y_deg, out_h, out_w):
            holder["meta"] = (lon_deg, lat_deg, fov_x_deg, fov_y_deg)
            return real_extract(frame, lon_deg, lat_deg, fov_x_deg, fov_y_deg, out_h, out_w)

        orig_module_extract = None
        from panosot import ostrack_tracker as tracker_mod

        orig_module_extract = tracker_mod.extract_patch
        tracker_mod.extract_patch = spy_extract
        try:
            width, height = frames[0].shape[1], frames[0].shape[0]
            init_bbox = state_to_erp_bbox(states[0], width, height)
            for t in range(len(frames)):
                holder["gt_state"] = states[t]
                if t == 0:
                    tracker.initialize(frames[0], init_bbox)
                else:
                    tracker.track(frames[t])
        finally:
            tracker_mod.extract_patch = orig_module_extract
        return tracker, states, holder

    def test_tracks_moving_blob(self) -> None:
        frames, states = make_erp_frames(12)
        tracker, gt_states, _ = self._make_tracker(frames, states)
        width, height = frames[0].shape[1], frames[0].shape[0]
        final_box = tracker._state_to_bbox(tracker.state)
        pred_cx = final_box[0] + 0.5 * final_box[2]
        pred_cy = final_box[1] + 0.5 * final_box[3]
        gt = gt_states[-1]
        gt_cx = (gt.lon + math.pi) / (2 * math.pi) * width
        gt_cy = (math.pi / 2 - gt.lat) / math.pi * height
        self.assertLess(abs(pred_cx - gt_cx), 6.0)
        self.assertLess(abs(pred_cy - gt_cy), 6.0)

    def test_track_sequence_outputs_valid_boxes(self) -> None:
        frames, states = make_erp_frames(8)
        holder = {"gt_state": states[0], "meta": (0.0, 0.0, 60.0, 60.0)}
        net = StubNet(holder, search_size=256)
        from panosot.ostrack_tracker import OstrackTrackerConfig, PanoOSTrackTracker

        tracker = PanoOSTrackTracker(
            model=net,
            config=OstrackTrackerConfig(
                device="cpu", relocalize_enabled=False, score_threshold=0.2
            ),
        )
        real_extract = tangent_mod.extract_patch

        def spy(frame, lon, lat, fx, fy, oh, ow):
            holder["meta"] = (lon, lat, fx, fy)
            holder["gt_state"] = states[tracker.frame_count]
            return real_extract(frame, lon, lat, fx, fy, oh, ow)

        from panosot import ostrack_tracker as tracker_mod

        orig = tracker_mod.extract_patch
        tracker_mod.extract_patch = spy
        try:
            width, height = frames[0].shape[1], frames[0].shape[0]
            init_bbox = state_to_erp_bbox(states[0], width, height)
            boxes = tracker.track_sequence(frames, init_bbox)
        finally:
            tracker_mod.extract_patch = orig
        self.assertEqual(len(boxes), len(frames))
        for box in boxes:
            self.assertTrue(np.all(np.isfinite(box)))
            self.assertGreaterEqual(box[2], 1.0)
            self.assertGreaterEqual(box[3], 1.0)


class StubNetReloc(StubNet):
    """Stub that only peaks when the search patch is centered on the GT."""

    def forward_search(self, x, z_tokens=None):
        gt = self.holder["gt_state"]
        lon_c, lat_c, fov_x, fov_y = self.holder["meta"]
        gt_lon_deg = gt.lon * 180.0 / math.pi
        gt_lat_deg = gt.lat * 180.0 / math.pi
        feat = self.config.feat_sz
        size = self.search_size
        score = torch.full((1, 1, feat, feat), 0.05, dtype=torch.float32)
        offset = torch.zeros(1, 2, feat, feat)
        size_map = torch.zeros(1, 2, feat, feat)
        if abs(lon_c - gt_lon_deg) < 1e-6 and abs(lat_c - gt_lat_deg) < 1e-6:
            bbox_px = state_to_patch_bbox(gt, lon_c, lat_c, fov_x, fov_y, size, size)
            cx_px = bbox_px[0] + 0.5 * bbox_px[2]
            cy_px = bbox_px[1] + 0.5 * bbox_px[3]
            cell_x = cx_px / size * feat
            cell_y = cy_px / size * feat
            ix = int(np.floor(np.clip(cell_x, 0, feat - 1)))
            iy = int(np.floor(np.clip(cell_y, 0, feat - 1)))
            score[0, 0, iy, ix] = 0.9
            offset[0, 0, iy, ix] = float(cell_x - ix)
            offset[0, 1, iy, ix] = float(cell_y - iy)
            size_map[0, 0, iy, ix] = float(bbox_px[2] / size * feat)
            size_map[0, 1, iy, ix] = float(bbox_px[3] / size * feat)
        return {
            "score_map": score,
            "offset_map": offset,
            "size_map": size_map,
        }


@unittest.skipIf(torch is None, "torch not installed")
class OstrackRelocalizeTests(unittest.TestCase):
    def test_global_relocalize_recovers_grid_center(self) -> None:
        from panosot.ostrack_tracker import (
            OstrackTrackerConfig,
            PanoOSTrackTracker,
        )

        frames, states = make_erp_frames(
            3, lon0_deg=-90.0, lat0_deg=0.0, dlon_per_frame=0.0
        )
        # Keep the target off the initial position: tracker starts at lon 0.
        init_state = SphereState(
            lon=0.0,
            lat=states[0].lat,
            equatorial_width=states[0].equatorial_width,
            angular_height=states[0].angular_height,
        )
        holder = {"gt_state": states[0], "meta": (0.0, 0.0, 60.0, 60.0)}
        net = StubNetReloc(holder, search_size=128)
        config = OstrackTrackerConfig(
            device="cpu",
            relocalize_enabled=False,
            relocalize_stride_deg=90.0,
            relocalize_lat_stride_deg=90.0,
            relocalize_lat_range_deg=90.0,
            relocalize_scales=(1.0,),
            relocalize_chunk_size=1,
        )
        tracker = PanoOSTrackTracker(model=net, config=config)
        width, height = frames[0].shape[1], frames[0].shape[0]
        tracker.initialize(frames[0], state_to_erp_bbox(init_state, width, height))
        # The stub sees patch centers through the extract_patch spy.
        from panosot import ostrack_tracker as tracker_mod

        real = tracker_mod.extract_patch

        def spy(frame, lon, lat, fx, fy, oh, ow):
            holder["meta"] = (lon, lat, fx, fy)
            return real(frame, lon, lat, fx, fy, oh, ow)

        tracker_mod.extract_patch = spy
        # extract_patches_batch calls extract_patch through the tangent module
        # namespace, so patch both namespaces the tracker reaches.
        tangent_mod.extract_patch = spy
        try:
            recovered = tracker._global_relocalize(frames[0])
        finally:
            tracker_mod.extract_patch = real
            tangent_mod.extract_patch = real
        self.assertTrue(recovered)
        lon_err = abs(
            math.degrees(
                (tracker.state.lon - states[0].lon + math.pi) % (2 * math.pi) - math.pi
            )
        )
        self.assertLess(lon_err, 5.0)
        self.assertLess(abs(math.degrees(tracker.state.lat - states[0].lat)), 5.0)


@unittest.skipUnless(RUN_SLOW and OSTRACK_384_FILE.is_file(), "slow weight test")
@unittest.skipIf(torch is None, "torch not installed")
class OstrackWeightLoadingTests(unittest.TestCase):
    def test_load_official_384_weights_strict(self) -> None:
        from panosot.ostrack import load_ostrack_checkpoint

        torch.set_num_threads(min(8, torch.get_num_threads()))
        model = build_ostrack("384")
        missing, unexpected = load_ostrack_checkpoint(
            model, OSTRACK_384_FILE, strict=True
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        model.eval()
        z = torch.randn(1, 3, 192, 192)
        x = torch.randn(1, 3, 384, 384)
        with torch.inference_mode():
            model.initialize(z)
            out = model.forward_search(x)
        self.assertEqual(tuple(out["score_map"].shape), (1, 1, 24, 24))
        self.assertTrue(torch.isfinite(out["score_map"]).all())

    def test_real_weights_demo_smoke(self) -> None:
        """Run the full tangent + OSTrack tracker on the 360tracking demo."""
        from panosot.ostrack_tracker import (
            OstrackTrackerConfig,
            PanoOSTrackTracker,
        )

        torch.set_num_threads(min(8, torch.get_num_threads()))
        model = build_ostrack("384")
        from panosot.ostrack import load_ostrack_checkpoint

        load_ostrack_checkpoint(model, OSTRACK_384_FILE, strict=True)
        tracker = PanoOSTrackTracker(
            model=model,
            config=OstrackTrackerConfig(
                device="cpu",
                relocalize_enabled=False,
                template_update_interval=0,
            ),
        )
        demo = PROJECT_DIR / "data" / "360tracking_demo" / "image"
        if not demo.is_dir():
            self.skipTest("demo frames missing")
        paths = sorted(p for p in demo.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:6]
        from panosot.io import load_image, load_boxes

        frames = [load_image(p) for p in paths]
        init = load_boxes(PROJECT_DIR / "data" / "360tracking_demo" / "init_box.txt")[0]
        boxes = tracker.track_sequence(frames, init)
        self.assertEqual(len(boxes), len(frames))
        for box in boxes:
            self.assertTrue(np.all(np.isfinite(box)))
            self.assertGreaterEqual(box[2], 1.0)
            self.assertGreaterEqual(box[3], 1.0)


if __name__ == "__main__":
    unittest.main()
