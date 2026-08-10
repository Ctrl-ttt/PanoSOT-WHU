from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_360vots_zips.py"
SPEC = importlib.util.spec_from_file_location("benchmark_360vots_zips", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Benchmark360VOTZipTests(unittest.TestCase):
    def test_evaluate_predictions_reports_drift_and_invalid_boxes(self) -> None:
        predictions = np.asarray(
            [[0.0, 0.0, 10.0, 10.0], [95.0, 0.0, 10.0, 10.0], [np.nan, 0.0, 10.0, 10.0]],
            dtype=np.float32,
        )
        targets = np.asarray(
            [[0.0, 0.0, 10.0, 10.0], [-5.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
            dtype=np.float32,
        )

        metrics = benchmark.evaluate_predictions(predictions, targets, 100.0)

        self.assertEqual(metrics["scored_frames"], 3)
        self.assertEqual(metrics["invalid_predictions"], 1)
        self.assertAlmostEqual(float(metrics["invalid_prediction_rate"]), 1.0 / 3.0)
        self.assertGreater(float(metrics["mean_iou_prefix_10"]), float(metrics["mean_iou_suffix_10"]))
        self.assertLess(float(metrics["center_error_px_median"]), 1e-4)

    def test_summary_is_rebuilt_from_sequence_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary)
            results = out_root / "sequence_results"
            results.mkdir()
            benchmark._atomic_write_json(
                results / "0002.json",
                {"sequence": "0002", "frames": 2, "auc": 0.2},
            )
            benchmark._atomic_write_json(
                results / "0001.json",
                {"sequence": "0001", "frames": 1, "auc": 0.1},
            )

            self.assertEqual(benchmark.write_summary(out_root), 2)
            summary_lines = (out_root / "summary.csv").read_text(encoding="utf-8").splitlines()

            self.assertEqual(summary_lines[1].split(",")[0], "0001")
            self.assertEqual(summary_lines[2].split(",")[0], "0002")
            self.assertIn('"sequence": "0001"', (out_root / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
