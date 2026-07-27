from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from panosot.geometry import erp_bbox_to_state, state_to_erp_bbox
from panosot.tracker import PanoSOTTracker
from tools.batch_evaluate import discover_sequences


class HybridTrackerTests(unittest.TestCase):
    def test_optical_flow_accepts_consistent_translation(self) -> None:
        tracker = PanoSOTTracker()
        first = np.zeros((160, 240, 3), dtype=np.float32)
        second = np.zeros_like(first)
        for y in range(58, 103, 8):
            for x in range(78, 123, 8):
                first[y : y + 3, x : x + 3] = 1.0
                second[y + 4 : y + 7, x + 6 : x + 9] = 1.0

        init_box = np.array([72.0, 52.0, 58.0, 58.0], dtype=np.float32)
        tracker.initialize(first, init_box)
        state, reliable = tracker._predict_with_optical_flow(second, tracker.state)

        self.assertTrue(reliable)
        predicted = state_to_erp_bbox(state, 240, 160)
        self.assertAlmostEqual(float(predicted[0] - init_box[0]), 6.0, delta=1.5)
        self.assertAlmostEqual(float(predicted[1] - init_box[1]), 4.0, delta=1.5)

    def test_color_model_tracks_saturated_component(self) -> None:
        tracker = PanoSOTTracker()
        first = np.zeros((180, 260, 3), dtype=np.float32)
        second = np.zeros_like(first)
        first[70:110, 90:130, 0] = 0.8
        second[76:124, 102:152, 0] = 0.8
        init_box = np.array([90.0, 70.0, 40.0, 40.0], dtype=np.float32)
        tracker.initialize(first, init_box)

        anchor = erp_bbox_to_state(init_box, 260, 180)
        state, reliable = tracker._predict_with_color(second, anchor)

        self.assertTrue(reliable)
        predicted = state_to_erp_bbox(state, 260, 180)
        self.assertGreater(float(predicted[0]), float(init_box[0]))
        self.assertGreater(float(predicted[1]), float(init_box[1]))

    def test_ncc_flow_estimates_consistent_translation(self) -> None:
        tracker = PanoSOTTracker()
        first = np.zeros((160, 240, 3), dtype=np.float32)
        second = np.zeros_like(first)
        for y in range(58, 103, 8):
            for x in range(78, 123, 8):
                first[y : y + 3, x : x + 3] = 1.0
                second[y + 5 : y + 8, x + 7 : x + 10] = 1.0

        init_box = np.array([72.0, 52.0, 58.0, 58.0], dtype=np.float32)
        tracker.initialize(first, init_box)
        current_gray = tracker._flow_gray(second)
        delta = tracker._estimate_ncc_flow(current_gray, init_box)

        self.assertIsNotNone(delta)
        self.assertAlmostEqual(float(delta[0]), 7.0, delta=1.5)
        self.assertAlmostEqual(float(delta[1]), 5.0, delta=1.5)

    def test_batch_discovery_supports_smoke_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence = root / "sequence"
            (sequence / "image").mkdir(parents=True)
            (sequence / "image" / "000000.jpg").touch()
            (sequence / "init.txt").write_text("0,0,10,10\n", encoding="utf-8")
            (sequence / "gt.txt").write_text("0,0,10,10\n", encoding="utf-8")

            discovered = discover_sequences(root)

        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0][0], "sequence")
        self.assertEqual(discovered[0][2].name, "init.txt")
        self.assertEqual(discovered[0][3].name, "gt.txt")


if __name__ == "__main__":
    unittest.main()
