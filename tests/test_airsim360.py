from __future__ import annotations

import unittest

import numpy as np

from panosot.airsim360 import (
    crop_erp_wrapped,
    decode_argb_instance_ids,
    extract_instance_boxes,
    temporal_target_fraction,
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

    def test_training_resize_makes_variable_crops_batchable(self) -> None:
        from tools.train_airsim360_adapter import _resize_patch

        first = _resize_patch(np.zeros((8, 13, 3), dtype=np.float32))
        second = _resize_patch(np.zeros((20, 7, 3), dtype=np.float32))
        self.assertEqual(first.shape, (128, 128, 3))
        self.assertEqual(second.shape, (128, 128, 3))

    def test_localization_loss_target_is_the_response_center(self) -> None:
        import torch

        from tools.train_airsim360_adapter import localization_loss

        response = torch.full((1, 1, 3, 5), -8.0)
        response[0, 0, 1, 2] = 8.0
        self.assertLess(localization_loss(response, torch, [(0.5, 0.5)]).item(), 0.01)

    def test_localization_loss_uses_shifted_target_location(self) -> None:
        import torch

        from tools.train_airsim360_adapter import localization_loss

        response = torch.full((1, 1, 8, 8), -8.0)
        response[0, 0, 4, 6] = 8.0
        self.assertLess(localization_loss(response, torch, [(0.65, 0.5)]).item(), 0.01)

    def test_temporal_pairs_are_built_only_for_consecutive_numbers(self) -> None:
        numbered = {0: "panorama_0.png", 1: "panorama_1.png", 3: "panorama_3.png"}
        consecutive = [(number, number + 1) for number in sorted(numbered) if number + 1 in numbered]
        self.assertEqual(consecutive, [(0, 1)])

    def test_temporal_target_fraction_tracks_wrapped_motion(self) -> None:
        fraction = temporal_target_fraction((1990, 100, 20, 20), (2, 100, 20, 20), 2048)
        self.assertAlmostEqual(fraction[0], 1.25)
        self.assertAlmostEqual(fraction[1], 0.5)


if __name__ == "__main__":
    unittest.main()
