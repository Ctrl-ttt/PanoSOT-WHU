from __future__ import annotations

import unittest

from tools.download_airsim360 import build_allow_patterns


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


if __name__ == "__main__":
    unittest.main()
