from __future__ import annotations

import unittest

import numpy as np

from panosot.airsim360 import (
    crop_erp_wrapped,
    decode_argb_instance_ids,
    extract_instance_boxes,
)


class AirSim360InstanceTests(unittest.TestCase):
    def test_decodes_rgba_bytes_as_airsim_argb_id(self) -> None:
        rgba = np.array([[[0x11, 0x22, 0x33, 0x44]]], dtype=np.uint8)
        self.assertEqual(decode_argb_instance_ids(rgba).item(), 0x44112233)

    def test_extracts_usable_instances_and_skips_background(self) -> None:
        ids = np.zeros((8, 10), dtype=np.uint32)
        ids[2:5, 3:7] = 7
        boxes = extract_instance_boxes(ids, min_pixels=4, max_fraction=0.8)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].instance_id, 7)
        self.assertEqual(boxes[0].bbox_xywh, (3, 2, 4, 3))
        self.assertEqual(boxes[0].pixel_count, 12)

    def test_extracts_short_box_across_erp_seam(self) -> None:
        ids = np.zeros((6, 10), dtype=np.uint32)
        ids[2:4, [0, 1, 8, 9]] = 7
        boxes = extract_instance_boxes(ids, min_pixels=4, max_fraction=0.9)
        self.assertEqual(boxes[0].bbox_xywh, (8, 2, 4, 2))

    def test_wrapped_crop_continues_at_erp_seam(self) -> None:
        image = np.tile(np.arange(6, dtype=np.uint8), (3, 1))[..., None]
        crop = crop_erp_wrapped(image, (5, 1, 2, 1), context=1.0)
        self.assertTrue(np.array_equal(crop[..., 0], np.array([[5, 0]], dtype=np.uint8)))

    def test_training_archive_pair_lookup_uses_instance_name_as_key(self) -> None:
        raw_name, instance_name = ("raw/panorama_1.png", "labels/panorama_1.png")
        pairs = {instance: raw for raw, instance in [(raw_name, instance_name)]}
        self.assertEqual(pairs[instance_name], raw_name)


if __name__ == "__main__":
    unittest.main()
