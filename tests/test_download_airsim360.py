from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.download_airsim360 import build_allow_patterns, download_with_curl


class AirSim360DownloadTests(unittest.TestCase):
    def test_builds_official_nyc_training_asset_paths(self) -> None:
        self.assertEqual(
            build_allow_patterns(["NYC"], ["raw", "instance"]),
            [
                "Omni360-Scene/NYC/nyc_Raw.zip",
                "Omni360-Scene/NYC/nyc_instance_panorama.zip",
            ],
        )

    def test_scene_and_component_order_is_preserved(self) -> None:
        self.assertEqual(
            build_allow_patterns(["DTW", "CITYPARK"], ["semantic"]),
            [
                "Omni360-Scene/DTW/dtw_seg_panorama.zip",
                "Omni360-Scene/CityPark/citypark_seg_panorama.zip",
            ],
        )

    def test_citypark_raw_is_split_into_three_official_archives(self) -> None:
        self.assertEqual(
            build_allow_patterns(["CITYPARK"], ["raw"]),
            [
                "Omni360-Scene/CityPark/citypark_Raw_Part1.zip",
                "Omni360-Scene/CityPark/citypark_Raw_Part2.zip",
                "Omni360-Scene/CityPark/citypark_Raw_Part3.zip",
            ],
        )

    def test_curl_download_keeps_partial_file_on_failure(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "raw.zip"
            with patch("tools.download_airsim360.subprocess.run") as run:
                run.return_value.returncode = 28
                with self.assertRaises(RuntimeError):
                    download_with_curl("https://example.invalid/raw.zip", destination)
                self.assertFalse(destination.exists())
                command = run.call_args.args[0]
                self.assertTrue(str(command[command.index("--output") + 1]).endswith("raw.zip.part"))


if __name__ == "__main__":
    unittest.main()
