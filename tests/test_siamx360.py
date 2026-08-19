from __future__ import annotations

import unittest

import numpy as np

from panosot.geometry import SphereState
from panosot.siamx360 import ERPRegionCropper, SiamX360Config


class SiamX360Tests(unittest.TestCase):
    def test_crop_wraps_erp_seam(self) -> None:
        frame = np.zeros((64, 128, 3), dtype=np.float32)
        frame[:, :4] = 1.0
        frame[:, -4:] = 0.5
        state = SphereState(np.pi - 0.01, 0.0, 0.2, 0.15)
        patch = ERPRegionCropper().crop(frame, state, search=True)
        self.assertEqual(patch.shape, (224, 224, 3))
        self.assertTrue(np.isfinite(patch).all())
        self.assertGreater(float(patch.mean()), 0.0)

    def test_context_is_clamped_to_valid_spherical_fov(self) -> None:
        cropper = ERPRegionCropper(SiamX360Config(max_horizontal_fov_deg=90.0))
        state = SphereState(0.0, np.deg2rad(80.0), 0.9, 0.5)
        patch = cropper.crop(np.ones((80, 160, 3), dtype=np.float32), state)
        self.assertEqual(patch.shape, (127, 127, 3))
        self.assertTrue(np.allclose(patch, 1.0, atol=1e-4))

    def test_wrapper_uses_a_real_deep_backend_when_components_are_supplied(self) -> None:
        class FeatureExtractor:
            pass

        tracker = __import__("panosot.siamx360", fromlist=["SiamX360Tracker"]).SiamX360Tracker(
            deep_extractor=FeatureExtractor(), similarity_head=object(),
        )
        self.assertTrue(tracker._deep_mode)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
