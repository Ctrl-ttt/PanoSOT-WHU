from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


SAMPLE_DATA = {
    "test1": {
        "url": "https://github.com/xuyzshaun/PanoSOT/raw/main/data/test_sample.zip",
        "filename": "test_sample.zip",
    }
}


def download_and_extract(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "sample_data.zip"
    
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = (downloaded / total_size) * 100
                print(f"\rProgress: {progress:.1f}%", end="")
    print("\nDownload complete!")
    
    print(f"Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    zip_path.unlink()
    print(f"Extraction complete!")
    
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample test data for PanoSOT")
    parser.add_argument("--output-dir", default="data/test", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    try:
        download_and_extract(SAMPLE_DATA["test1"]["url"], output_dir)
        print(f"\nSample data downloaded to {output_dir}")
        
        seq_dirs = list(output_dir.glob("*"))
        if seq_dirs:
            for seq in seq_dirs:
                if seq.is_dir():
                    frames = list(seq.glob("*.jpg")) + list(seq.glob("*.png"))
                    print(f"  - {seq.name}: {len(frames)} frames")
                    
                    init_file = seq / "init.txt"
                    if init_file.exists():
                        with open(init_file) as f:
                            print(f"    Init box: {f.read().strip()}")
    except Exception as e:
        print(f"Error downloading sample data: {e}")
        print("Trying alternative download...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        init_box = [500, 300, 200, 200]
        init_file = output_dir / "init.txt"
        init_file.write_text(f"{','.join(map(str, init_box))}")
        print(f"Created sample init box: {init_box}")


if __name__ == "__main__":
    main()