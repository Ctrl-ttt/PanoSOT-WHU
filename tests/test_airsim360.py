from __future__ import annotations

import unittest

import numpy as np

from panosot.airsim360 import decode_argb_instance_ids, extract_instance_boxes


class AirSim360InstanceTests(unittest.TestCase):
    def test_decodes_rgba_bytes_as_airsim_argb_id(self) -> None:
        rgba = np.array([[[0x11, 0x22, 0x33, 0x44]]], dtype=np.uint8)
        self.assertEqual(decode_argb_instance_ids(rgba).item(), 0x44332211)

    def test_extracts_usable_instances_and_skips_background(self) -> None:
        ids = np.zeros((8, 10), dtype=np.uint32)
        ids[2:5, 3:7] = 7
        boxes = extract_instance_boxes(ids, min_pixels=4, max_fraction=0.8)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].instance_id, 7)
        self.assertEqual(boxes[0].bbox_xywh, (3, 2, 4, 3))
        self.assertEqual(boxes[0].pixel_count, 12)


if __name__ == "__main__":
    unittest.main()
