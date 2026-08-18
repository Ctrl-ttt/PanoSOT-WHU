from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.train_local_bfov_adapter import (
    bfov_deg_to_erp_bbox,
    read_bfov_file,
    select_pair_specs,
    target_fraction,
)


class LocalBFoVTrainingTests(unittest.TestCase):
    def test_bfov_to_erp_bbox_preserves_wrapped_center(self) -> None:
        box = bfov_deg_to_erp_bbox(np.array([-179.0, 0.0, 20.0, 30.0]), 1440, 720)
        cx = (float(box[0]) + 0.5 * float(box[2])) % 1440.0
        cy = float(box[1]) + 0.5 * float(box[3])
        self.assertAlmostEqual(cx, 4.0, places=3)
        self.assertAlmostEqual(cy, 360.0, places=3)
        self.assertAlmostEqual(float(box[2]), 80.0, places=3)
        self.assertAlmostEqual(float(box[3]), 120.0, places=3)

    def test_bfov_to_erp_bbox_scales_width_with_latitude(self) -> None:
        box = bfov_deg_to_erp_bbox(np.array([0.0, 60.0, 20.0, 30.0]), 1440, 720)
        self.assertAlmostEqual(float(box[2]), 160.0, places=3)
        self.assertAlmostEqual(float(box[3]), 120.0, places=3)

    def test_target_fraction_uses_shortest_longitude_delta(self) -> None:
        previous = bfov_deg_to_erp_bbox(np.array([179.0, 0.0, 20.0, 30.0]), 1440, 720)
        current = bfov_deg_to_erp_bbox(np.array([-179.0, 0.0, 20.0, 30.0]), 1440, 720)
        fraction_x, fraction_y = target_fraction(previous, current, 1440, 720)
        self.assertGreater(fraction_x, 0.5)
        self.assertLess(fraction_x, 0.6)
        self.assertAlmostEqual(fraction_y, 0.5, places=6)

    def test_select_pair_specs_reads_local_sequence_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train_sim"
            sequence = root / "seq_0001"
            sequence.mkdir(parents=True)
            (root / "seqlist.txt").write_text("seq_0001\n", encoding="utf-8")
            (sequence / "video.mp4").write_bytes(b"placeholder")
            (sequence / "groundtruth.txt").write_text(
                "0,0,20,30\n1,0,20,30\n2,0,20,30\n",
                encoding="utf-8",
            )

            with patch("tools.train_local_bfov_adapter.video_info", return_value=(1440, 720, 3)):
                specs = select_pair_specs(
                    [root],
                    None,
                    max_samples=10,
                    max_per_sequence=10,
                    delta=1,
                    sample_step=1,
                    search_enlarge=4.0,
                    seed=0,
                )

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].sequence.name, "train_sim/seq_0001")
        self.assertEqual(specs[0].width, 1440)
        self.assertEqual(specs[0].height, 720)

    def test_read_bfov_file_accepts_commas_and_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundtruth.txt"
            path.write_text("1,2,3,4\n5 6 7 8\n", encoding="utf-8")
            rows = read_bfov_file(path)
        self.assertEqual(rows.shape, (2, 4))
        self.assertTrue(np.array_equal(rows[1], np.array([5, 6, 7, 8], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
