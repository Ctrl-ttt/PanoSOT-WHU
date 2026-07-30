from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from panosot.geometry import bilinear_sample, erp_bbox_to_state, state_to_erp_bbox, tangent_patch
from panosot.models import build_similarity_head
from panosot.tracker import PanoSOTTracker, TrackerConfig
from tools.batch_evaluate import discover_sequences


class HybridTrackerTests(unittest.TestCase):
    def test_handcrafted_result_uses_handcrafted_confidence_scale(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(use_deep_features=True))
        tracker._deep_mode = True
        tracker.runtime_stats.last_psr = 1.0

        self.assertLess(tracker._state_trust(0.24), 0.5)
        self.assertEqual(tracker._state_trust(0.24, handcrafted_result=True), 1.0)
        self.assertEqual(tracker._high_confidence(True), tracker.config.handcrafted_high_confidence)
        self.assertEqual(
            tracker._relocalize_confidence_threshold(True),
            tracker.config.handcrafted_relocalize_confidence_threshold,
        )

    def test_deep_probe_interval_retries_after_low_psr(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(deep_probe_interval=5))
        tracker._last_deep_probe_frame = 3

        tracker._frame_count = 7
        self.assertFalse(tracker._deep_probe_due())
        tracker._frame_count = 8
        self.assertTrue(tracker._deep_probe_due())

    def test_deep_probe_interval_backs_off_and_recovers(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            deep_probe_interval=5,
            deep_probe_backoff_after=2,
            deep_probe_max_interval=20,
        ))

        tracker._record_deep_probe_psr(1.2)
        self.assertEqual(tracker._current_deep_probe_interval(), 5)
        tracker._record_deep_probe_psr(1.3)
        self.assertEqual(tracker._current_deep_probe_interval(), 10)
        tracker._record_deep_probe_psr(1.4)
        tracker._record_deep_probe_psr(1.5)
        self.assertEqual(tracker._current_deep_probe_interval(), 20)

        tracker._record_deep_probe_psr(2.1)
        self.assertEqual(tracker._current_deep_probe_interval(), 5)

    def test_deep_relocalization_rejects_low_psr_candidate(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            relocalize_score_margin=0.15,
            deep_relocalize_accept_psr_threshold=1.75,
        ))
        tracker._deep_mode = True

        self.assertFalse(tracker._accept_relocalization_candidate(0.70, 0.40, 1.50, False, True))
        self.assertTrue(tracker._accept_relocalization_candidate(0.70, 0.40, 1.90, False, True))
        self.assertTrue(tracker._accept_relocalization_candidate(0.70, 0.40, 0.00, False, False))
        self.assertFalse(tracker._accept_relocalization_candidate(0.50, 0.40, 2.10, False, True))
        self.assertFalse(tracker._accept_relocalization_candidate(0.70, 0.40, 2.10, True, True))

    def test_batched_response_scores_handle_tiny_maps(self) -> None:
        import torch

        tracker = PanoSOTTracker()
        tracker.deep_extractor = type("TorchHolder", (), {"_torch": torch})()
        responses = torch.tensor([
            [[[0.1, 0.2], [0.3, 0.8]]],
            [[[0.5, 0.4], [0.2, 0.1]]],
        ])

        scores, _, metadata = tracker._response_scores_offsets_batch(responses, None, None)

        for index, response in enumerate(responses):
            response_np = response.squeeze().numpy()
            peak_index = int(response_np.argmax())
            peak_y, peak_x = divmod(peak_index, response_np.shape[1])
            expected, _, expected_psr, expected_apce = tracker._response_confidence(
                response_np, peak_y, peak_x,
            )
            self.assertAlmostEqual(scores[index], expected, places=6)
            self.assertAlmostEqual(metadata[index]["psr"], expected_psr, places=6)
            self.assertAlmostEqual(metadata[index]["apce"], expected_apce, places=6)

    def test_bilinear_sample_wraps_across_erp_seam(self) -> None:
        frame = np.zeros((2, 4, 3), dtype=np.float32)
        frame[:, 0] = 1.0
        xs = np.array([[3.5]], dtype=np.float32)
        ys = np.array([[0.5]], dtype=np.float32)

        sampled = bilinear_sample(frame, xs, ys)

        self.assertTrue(np.allclose(sampled, 0.5, atol=0.02))

    def test_relocalization_attempt_uses_its_own_cooldown(self) -> None:
        config = TrackerConfig(relocalize_min_interval=10)
        tracker = PanoSOTTracker(config)

        tracker._frame_count = 20
        tracker._last_relocalize_attempt_frame = tracker._frame_count
        tracker._last_relocalize_frame = -config.relocalize_min_interval

        self.assertFalse(tracker._relocalize_cooldown_elapsed())
        tracker._frame_count += config.relocalize_min_interval
        self.assertTrue(tracker._relocalize_cooldown_elapsed())
        self.assertEqual(tracker._last_relocalize_frame, -config.relocalize_min_interval)

    def test_deep_relocalization_interval_backs_off_after_rejects(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            relocalize_min_interval=10,
            deep_relocalize_backoff_after=2,
            deep_relocalize_max_interval=80,
        ))
        tracker._deep_mode = True

        self.assertEqual(tracker._current_relocalization_interval(), 10)
        tracker._consecutive_relocalization_rejects = 2
        self.assertEqual(tracker._current_relocalization_interval(), 20)
        tracker._consecutive_relocalization_rejects = 6
        self.assertEqual(tracker._current_relocalization_interval(), 80)

    def test_template_bank_batch_matches_individual_responses(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(7)
        templates = torch.randn(4, 3, 3, 3, generator=generator)
        searches = torch.randn(5, 3, 7, 7, generator=generator)
        head = build_similarity_head("depthwise_xcorr")

        batched = head.forward_bank_to_searches(templates, searches)
        expected = torch.stack([
            torch.cat([
                head(templates[index : index + 1], searches[search : search + 1])
                for index in range(templates.shape[0])
            ], dim=0)
            for search in range(searches.shape[0])
        ], dim=0)

        self.assertTrue(torch.equal(batched, expected))

    def test_tangent_grid_cache_preserves_patch_values(self) -> None:
        rng = np.random.default_rng(11)
        frame = rng.random((80, 160, 3), dtype=np.float32)

        first = tangent_patch(frame, 0.4, -0.2, 0.7, 0.5, 48, 64)
        second = tangent_patch(frame, 0.4, -0.2, 0.7, 0.5, 48, 64)

        self.assertTrue(np.array_equal(first, second))

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

    def test_ncc_low_score_scale_uses_reliable_flow_bounds(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_flow_scale_tolerance=0.10,
        ))
        previous = np.array([100.0, 80.0, 180.0, 80.0], dtype=np.float64)
        candidate = np.array([80.0, 85.0, 180.0, 68.0], dtype=np.float64)

        guarded = tracker._guard_ncc_candidate_scale(
            candidate, previous, np.array([1.12, 1.03]), 0.30,
        )

        self.assertGreater(float(guarded[2]), float(candidate[2]))
        self.assertGreater(float(guarded[3]), float(candidate[3]))
        self.assertAlmostEqual(
            float(guarded[0] + 0.5 * guarded[2]),
            float(candidate[0] + 0.5 * candidate[2]),
        )
        self.assertTrue(np.array_equal(
            tracker._guard_ncc_candidate_scale(candidate, previous, None, 0.30),
            candidate,
        ))
        self.assertTrue(np.array_equal(
            tracker._guard_ncc_candidate_scale(
                candidate, previous, np.array([1.12, 1.03]), 0.75,
            ),
            candidate,
        ))
        tracker._deep_mode = True
        self.assertTrue(np.array_equal(
            tracker._guard_ncc_candidate_scale(
                candidate, previous, np.array([1.12, 1.03]), 0.30,
            ),
            candidate,
        ))

    def test_ncc_rejects_low_score_candidate_opposing_flow(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_flow_disagreement_ratio=0.35,
            handcrafted_ncc_flow_disagreement_score=0.55,
        ))
        box = np.array([0.0, 0.0, 170.0, 90.0], dtype=np.float64)
        flow = np.array([-30.0, 4.0], dtype=np.float64)

        self.assertFalse(tracker._ncc_candidate_consistent(
            np.array([43.0, 0.0]), flow, box, 0.28,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([-25.0, 2.0]), flow, box, 0.28,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([43.0, 0.0]), flow, box, 0.75,
        ))
        self.assertFalse(tracker._ncc_candidate_consistent(
            np.array([77.0, 13.0]), np.array([-23.0, 2.0]), box, 0.23,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([3.0, 0.0]), np.array([-2.0, 0.0]), box, 0.23,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([6.0, 4.5]),
            np.array([-6.7, 3.0]),
            np.array([0.0, 0.0, 26.0, 12.0]),
            0.49,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([9.5, -2.0]),
            np.array([19.2, -8.5]),
            np.array([0.0, 0.0, 20.0, 48.0]),
            0.36,
            strict_axes=True,
        ))
        self.assertFalse(tracker._ncc_candidate_consistent(
            np.array([-20.2, 20.8]),
            np.array([-12.5, 4.1]),
            np.array([0.0, 0.0, 173.0, 68.0]),
            0.23,
            strict_axes=True,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([-20.2, 10.0]),
            np.array([-12.5, 4.1]),
            np.array([0.0, 0.0, 173.0, 68.0]),
            0.23,
            strict_axes=True,
        ))
        self.assertTrue(tracker._ncc_candidate_consistent(
            np.array([-20.2, 20.8]),
            np.array([-12.5, 4.1]),
            np.array([0.0, 0.0, 173.0, 68.0]),
            0.23,
            strict_axes=False,
        ))

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
