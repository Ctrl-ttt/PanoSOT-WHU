from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests


TEST_SEQUENCES = {
    "sphere1": {
        "url": "https://github.com/OpenPTrack/open_ptrack/raw/master/data/sample_data/person1_001.zip",
        "frames_dir": "frames",
        "init_box": [100, 100, 150, 200],
    }
}


def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {output_path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def create_init_box_file(output_path: Path, box: list[int]) -> None:
