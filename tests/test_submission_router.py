from __future__ import annotations

import unittest

import numpy as np

from submission.track import _should_use_deep_box, should_use_deep
from tools.convert_train_dataset import bfov_to_pixel_boxes


class SubmissionRouterTests(unittest.TestCase):
    def test_bfov_conversion_expands_horizontal_extent_at_latitude(self) -> None:
        equator = bfov_to_pixel_boxes(np.array([[0.0, 0.0, 20.0, 10.0]]), 3600, 1800)
        high_lat = bfov_to_pixel_boxes(np.array([[0.0, 60.0, 20.0, 10.0]]), 3600, 1800)
        self.assertAlmostEqual(float(equator[0, 2]), 200.0, places=4)
        self.assertAlmostEqual(float(high_lat[0, 2]), 400.0, places=4)

    def test_routes_medium_tall_targets_to_deep(self) -> None:
        self.assertTrue(should_use_deep([0.0, 0.0, 21.0, 51.0], 1440, 720))
        self.assertTrue(
            _should_use_deep_box(np.array([0.0, 0.0, 84.0, 205.0], dtype=np.float32), 6.0)
        )
        self.assertTrue(should_use_deep([0.0, 0.0, 20.5, 55.6], 1440, 720))

    def test_routes_tiny_moderate_aspect_targets_to_deep(self) -> None:
        self.assertTrue(should_use_deep([0.0, 0.0, 14.8, 26.1], 1440, 720))
        self.assertTrue(
            _should_use_deep_box(np.array([0.0, 0.0, 59.0, 104.0], dtype=np.float32), 1.1)
        )

    def test_rejects_wide_or_tiny_targets(self) -> None:
        self.assertFalse(should_use_deep([0.0, 0.0, 75.0, 33.0], 1440, 720))
        self.assertFalse(should_use_deep([0.0, 0.0, 10.0, 20.0], 1440, 720))
        self.assertFalse(
            _should_use_deep_box(np.array([0.0, 0.0, 294.0, 131.0], dtype=np.float32), -12.0)
        )
        self.assertFalse(
            _should_use_deep_box(np.array([0.0, 0.0, 38.0, 84.0], dtype=np.float32), 2.4)
        )


if __name__ == "__main__":
    unittest.main()
