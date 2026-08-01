from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from panosot.geometry import bilinear_sample, erp_bbox_to_state, state_to_erp_bbox, tangent_patch
from panosot.models import build_similarity_head
from panosot.tracker import PanoSOTTracker, TrackerConfig
from tools.batch_evaluate import SeqResult, discover_sequences, summarize_results


class HybridTrackerTests(unittest.TestCase):
    def test_handcrafted_result_uses_handcrafted_confidence_scale(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(use_deep_features=True))
        tracker._deep_mode = True
        tracker.runtime_stats.last_psr = 1.0

        self.assertAlmostEqual(tracker._handcrafted_confidence(0.18, "ncc"), 0.18)
        self.assertEqual(
            tracker._handcrafted_confidence(0.18, "flow"),
            tracker.config.handcrafted_high_confidence,
        )
        self.assertEqual(tracker.config.handcrafted_ncc_low_score_commit_threshold, 0.55)
        self.assertEqual(tracker.config.handcrafted_ncc_max_scale_step, 1.12)
        self.assertLess(tracker._state_trust(0.24), 0.5)
        self.assertEqual(tracker._state_trust(0.24, handcrafted_result=True), 1.0)
        self.assertEqual(tracker._high_confidence(True), tracker.config.handcrafted_high_confidence)
        self.assertEqual(
            tracker._relocalize_confidence_threshold(True),
            tracker.config.handcrafted_relocalize_confidence_threshold,
        )

    def test_ncc_rejects_low_texture_template(self) -> None:
        first = np.zeros((180, 260, 3), dtype=np.float32)
        first[70:110, 90:140, 0] = 1.0
        second = np.zeros_like(first)
        init_box = np.array([90.0, 70.0, 50.0, 40.0], dtype=np.float32)
        tracker = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_parallel_workers=1,
            handcrafted_ncc_flow_motion_trigger=10.0,
            handcrafted_ncc_flow_score_trigger=-1.0,
        ))
        tracker.initialize(first, init_box)

        _, score, reliable = tracker._predict_with_ncc(second)

        self.assertFalse(reliable)
        self.assertFalse(tracker._last_ncc_reliable)
        self.assertEqual(score, -1.0)
        self.assertTrue(np.array_equal(tracker._ncc_bbox, init_box))

    def test_ncc_rejects_low_score_missing_textured_target(self) -> None:
        rng = np.random.default_rng(456)
        first = rng.random((180, 260, 3), dtype=np.float32) * 0.02
        second = rng.random((180, 260, 3), dtype=np.float32)
        first[70:110, 90:140] = rng.random((40, 50, 3), dtype=np.float32)
        init_box = np.array([90.0, 70.0, 50.0, 40.0], dtype=np.float32)
        tracker = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_parallel_workers=1,
            handcrafted_ncc_flow_motion_trigger=10.0,
            handcrafted_ncc_flow_score_trigger=-1.0,
            handcrafted_ncc_reliable_score=0.35,
        ))
        tracker.initialize(first, init_box)

        _, score, reliable = tracker._predict_with_ncc(second)

        self.assertFalse(reliable)
        self.assertLess(score, tracker.config.handcrafted_ncc_reliable_score)
        self.assertTrue(np.array_equal(tracker._ncc_bbox, init_box))

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

    def test_ncc_crop_wraps_across_erp_seam(self) -> None:
        tracker = PanoSOTTracker()
        frame = np.tile(np.arange(5, dtype=np.uint8), (4, 1))

        crop = tracker._crop_erp_box(
            frame,
            np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32),
        )

        self.assertIsNotNone(crop)
        self.assertEqual(crop.tolist(), [[3, 4, 0, 1], [3, 4, 0, 1]])

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

    def test_deep_mode_relocalizes_with_deep_candidates_after_handcrafted_fallback(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(use_deep_features=True))
        tracker._deep_mode = True
        self.assertTrue(tracker._use_deep_relocalization())

        tracker._deep_mode = False
        self.assertFalse(tracker._use_deep_relocalization())

    def test_relocalization_large_jump_waits_for_a_real_loss_streak(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            relocalize_jump_gate_frames=30,
            relocalize_jump_gate_lost_frames=10,
            relocalize_max_lon_jump_deg=40.0,
            relocalize_max_lat_jump_deg=18.0,
        ))
        tracker._frame_count = 100
        tracker.lost_frames = 0
        tracker._consecutive_low_ncc = 1
        self.assertTrue(tracker._relocalization_jump_is_gated(90.0, 25.0))

        tracker._consecutive_low_ncc = 10
        self.assertFalse(tracker._relocalization_jump_is_gated(90.0, 25.0))
        self.assertFalse(tracker._relocalization_jump_is_gated(20.0, 10.0))

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

    def test_ncc_parallel_matches_serial_prediction(self) -> None:
        rng = np.random.default_rng(19)
        first = rng.random((240, 360, 3), dtype=np.float32) * 0.05
        second = first.copy()
        patch = rng.random((70, 110, 3), dtype=np.float32)
        first[80:150, 120:230] = patch
        second[80:150, 120:230] = 0.0
        second[86:156, 129:239] = patch
        init_box = np.array([120.0, 80.0, 110.0, 70.0], dtype=np.float32)
        serial = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_parallel_workers=1,
            handcrafted_ncc_flow_motion_trigger=10.0,
            handcrafted_ncc_flow_score_trigger=-1.0,
        ))
        parallel = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_parallel_workers=8,
            handcrafted_ncc_flow_motion_trigger=10.0,
            handcrafted_ncc_flow_score_trigger=-1.0,
        ))
        serial.initialize(first, init_box)
        parallel.initialize(first, init_box)

        serial_state, serial_score, serial_reliable = serial._predict_with_ncc(second)
        parallel_state, parallel_score, parallel_reliable = parallel._predict_with_ncc(second)

        self.assertEqual(parallel_reliable, serial_reliable)
        self.assertEqual(parallel_score, serial_score)
        self.assertEqual(parallel_state, serial_state)
        self.assertTrue(np.array_equal(parallel._ncc_bbox, serial._ncc_bbox))
        self.assertTrue(np.array_equal(parallel._ncc_velocity, serial._ncc_velocity))
        serial.close()
        parallel.close()

    def test_ncc_parallel_matches_serial_across_erp_seam(self) -> None:
        rng = np.random.default_rng(29)
        first = rng.random((220, 360, 3), dtype=np.float32) * 0.03
        second = first.copy()
        patch = rng.random((56, 48, 3), dtype=np.float32)
        first_x = 334
        second_x = 342
        y = 82
        first[y : y + patch.shape[0], np.mod(
            np.arange(first_x, first_x + patch.shape[1]), first.shape[1],
        )] = patch
        second[y : y + patch.shape[0], np.mod(
            np.arange(first_x, first_x + patch.shape[1]), second.shape[1],
        )] = 0.0
        second[y : y + patch.shape[0], np.mod(
            np.arange(second_x, second_x + patch.shape[1]), second.shape[1],
        )] = patch
        init_box = np.array([first_x, y, patch.shape[1], patch.shape[0]], dtype=np.float32)
        serial = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_parallel_workers=1,
            handcrafted_ncc_flow_motion_trigger=10.0,
            handcrafted_ncc_flow_score_trigger=-1.0,
        ))
        parallel = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_parallel_workers=8,
            handcrafted_ncc_flow_motion_trigger=10.0,
            handcrafted_ncc_flow_score_trigger=-1.0,
        ))
        serial.initialize(first, init_box)
        parallel.initialize(first, init_box)

        serial_result = serial._predict_with_ncc(second)
        parallel_result = parallel._predict_with_ncc(second)

        self.assertEqual(parallel_result, serial_result)
        self.assertTrue(np.array_equal(parallel._ncc_bbox, serial._ncc_bbox))
        self.assertAlmostEqual(float(parallel._ncc_bbox[0]), second_x, delta=1.0)
        self.assertGreaterEqual(float(parallel._ncc_bbox[0]), 0.0)
        self.assertLess(float(parallel._ncc_bbox[0]), second.shape[1])
        serial.close()
        parallel.close()

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
        deep_guarded = tracker._guard_ncc_candidate_scale(
            candidate, previous, np.array([1.12, 1.03]), 0.30,
        )
        self.assertGreater(float(deep_guarded[2]), float(candidate[2]))
        self.assertGreater(float(deep_guarded[3]), float(candidate[3]))

    def test_ncc_low_score_scale_keeps_recent_confident_anchor(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            handcrafted_ncc_scale_anchor_score=0.55,
            handcrafted_ncc_scale_anchor_min_ratio=0.65,
        ))
        tracker._ncc_scale_anchor = np.array([180.0, 90.0], dtype=np.float64)
        previous = np.array([100.0, 80.0, 170.0, 80.0], dtype=np.float64)
        candidate = np.array([120.0, 95.0, 120.0, 45.0], dtype=np.float64)

        guarded = tracker._guard_ncc_scale_anchor(candidate, previous, 0.30)

        self.assertGreaterEqual(float(guarded[2]), 117.0)
        self.assertGreaterEqual(float(guarded[3]), 58.5)
        self.assertAlmostEqual(
            float(guarded[0] + 0.5 * guarded[2]),
            float(candidate[0] + 0.5 * candidate[2]),
        )
        high_score_guarded = tracker._guard_ncc_scale_anchor(candidate, previous, 0.75)
        self.assertGreaterEqual(float(high_score_guarded[2]), 117.0)
        self.assertGreaterEqual(float(high_score_guarded[3]), 58.5)
        tracker._deep_mode = True
        deep_guarded = tracker._guard_ncc_scale_anchor(candidate, previous, 0.30)
        self.assertGreaterEqual(float(deep_guarded[2]), 117.0)
        self.assertGreaterEqual(float(deep_guarded[3]), 58.5)

    def test_ncc_temporal_scale_step_is_bounded(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(handcrafted_ncc_max_scale_step=1.12))
        previous = np.array([100.0, 80.0, 180.0, 80.0], dtype=np.float64)
        candidate = np.array([110.0, 85.0, 100.0, 44.0], dtype=np.float64)
        size = np.maximum(previous[2:4], 1.0)
        bounded = np.clip(candidate[2:4] / size, 1.0 / 1.12, 1.12)
        self.assertTrue(np.allclose(bounded, [1.0 / 1.12, 1.0 / 1.12]))

    def test_ncc_scale_anchor_does_not_shrink_on_high_score(self) -> None:
        tracker = PanoSOTTracker()
        tracker._ncc_scale_anchor = np.array([180.0, 90.0], dtype=np.float64)
        tracker._ncc_scale_anchor = np.maximum(
            tracker._ncc_scale_anchor, np.array([120.0, 60.0]),
        )
        self.assertTrue(np.array_equal(
            tracker._ncc_scale_anchor, np.array([180.0, 90.0]),
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

    def test_semantic_proposal_requires_streak_and_cooldown(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            use_deep_features=True,
            deep_semantic_trigger_frames=3,
            deep_semantic_min_interval=20,
        ))
        tracker._deep_mode = True
        tracker._semantic_history = [object()]
        tracker._semantic_size_anchor = erp_bbox_to_state(
            np.array([10.0, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
        )
        tracker._frame_count = 40
        tracker._last_semantic_proposal_frame = 10

        tracker._consecutive_low_ncc = 2
        self.assertFalse(tracker._semantic_proposal_due())
        tracker._consecutive_low_ncc = 3
        self.assertTrue(tracker._semantic_proposal_due())
        tracker._last_semantic_proposal_frame = 30
        self.assertFalse(tracker._semantic_proposal_due())

    def test_semantic_topk_suppresses_nearby_candidates(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            deep_semantic_min_score=0.58,
            deep_semantic_topk=2,
            deep_semantic_nms_deg=18.0,
        ))
        states = [
            erp_bbox_to_state(np.array([10.0, 20.0, 20.0, 10.0]), 360, 180),
            erp_bbox_to_state(np.array([15.0, 20.0, 20.0, 10.0]), 360, 180),
            erp_bbox_to_state(np.array([100.0, 20.0, 20.0, 10.0]), 360, 180),
        ]

        selected = tracker._select_semantic_topk(states, [0.90, 0.85, 0.80])

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0][0], states[0])
        self.assertEqual(selected[1][0], states[2])

    def test_semantic_verification_rejection_restores_quality_state(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            deep_semantic_verify_score_margin=0.08,
            deep_semantic_verify_psr=1.75,
        ))
        current = erp_bbox_to_state(
            np.array([10.0, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
        )
        proposal = erp_bbox_to_state(
            np.array([100.0, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
        )
        tracker.runtime_stats.last_score = 0.40
        tracker.runtime_stats.last_peak = 1.0
        tracker.runtime_stats.last_psr = 1.2
        tracker.runtime_stats.last_apce = 2.0
        def fake_search(frame: np.ndarray, state) -> tuple:
            result_state, score, psr = current, 0.60, 2.0
            tracker.runtime_stats.last_score = score
            tracker.runtime_stats.last_peak = score + 1.0
            tracker.runtime_stats.last_psr = psr
            tracker.runtime_stats.last_apce = score + 2.0
            return result_state, score

        tracker._local_search_deep = fake_search
        tracker._refine_semantic_proposals = lambda frame, states, anchor: [
            (proposal, 0.80, 0.65, 2.1, 0.65),
        ]
        result = tracker._verify_semantic_proposals(
            np.zeros((10, 20, 3), dtype=np.float32),
            current,
            0.40,
            [(proposal, 0.80)],
        )

        self.assertIsNone(result)
        self.assertEqual(
            (
                tracker.runtime_stats.last_score,
                tracker.runtime_stats.last_peak,
                tracker.runtime_stats.last_psr,
                tracker.runtime_stats.last_apce,
            ),
            (0.40, 1.0, 1.2, 2.0),
        )

    def test_semantic_verification_accepts_clear_ranked_advantage(self) -> None:
        tracker = PanoSOTTracker(TrackerConfig(
            deep_semantic_verify_score_margin=0.08,
            deep_semantic_verify_psr=1.75,
        ))
        current = erp_bbox_to_state(
            np.array([10.0, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
        )
        proposal = erp_bbox_to_state(
            np.array([100.0, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
        )
        tracker._local_search_deep = lambda frame, state: (current, 0.48)
        tracker.runtime_stats.last_psr = 1.6
        tracker._refine_semantic_proposals = lambda frame, states, anchor: [
            (proposal, 0.82, 0.64, 2.4, 0.59),
        ]

        result = tracker._verify_semantic_proposals(
            np.zeros((10, 20, 3), dtype=np.float32),
            current,
            0.48,
            [(proposal, 0.80)],
        )

        self.assertIsNotNone(result)
        assert result is not None
        recovered, score, psr = result
        self.assertEqual((recovered.lon, recovered.lat), (proposal.lon, proposal.lat))
        self.assertEqual(
            (recovered.equatorial_width, recovered.angular_height),
            (current.equatorial_width, current.angular_height),
        )
        self.assertEqual((score, psr), (0.64, 2.4))
        self.assertEqual(tracker.runtime_stats.last_semantic_score, 0.82)

    def test_semantic_verification_batches_all_proposals(self) -> None:
        tracker = PanoSOTTracker()
        current = erp_bbox_to_state(
            np.array([10.0, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
        )
        proposals = [
            erp_bbox_to_state(
                np.array([x, 10.0, 40.0, 30.0], dtype=np.float32), 200, 100,
            )
            for x in (60.0, 100.0, 140.0)
        ]
        tracker._local_search_deep = lambda frame, state: (current, 0.60)
        calls = []

        def fake_batch(frame, states, anchor):
            calls.append(list(states))
            return [(state, 0.70, 0.61, 2.0, 0.61) for state in states]

        tracker._refine_semantic_proposals = fake_batch
        tracker._verify_semantic_proposals(
            np.zeros((10, 20, 3), dtype=np.float32),
            current,
            0.60,
            [(state, 0.70) for state in proposals],
        )

        self.assertEqual(calls, [proposals])
        self.assertEqual(tracker.runtime_stats.semantic_refine_attempts, 3)

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

    def test_batch_summary_handles_all_failed_sequences(self) -> None:
        results = [
            SeqResult("broken", 0, 0.0, 0.0, 0.0, 0.0, error="init_box: invalid"),
        ]

        valid, summary = summarize_results(results)

        self.assertEqual(valid, [])
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["succeeded"], 0)
        self.assertEqual(summary["failed"], 1)
        self.assertIsNone(summary["avg_success_rate"])
        self.assertIsNone(summary["avg_auc"])
        self.assertIsNone(summary["avg_mean_iou"])
        self.assertEqual(summary["total_elapsed_sec"], 0.0)
        self.assertIsNone(summary["avg_fps"])


if __name__ == "__main__":
    unittest.main()
