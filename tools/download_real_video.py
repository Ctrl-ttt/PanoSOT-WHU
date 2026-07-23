from __future__ import annotations

import requests
import os
import zipfile
from pathlib import Path


def download_file(url: str, save_path: Path, timeout: int = 30):
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\rDownloading: {progress:.1f}% ({downloaded/1024/1024:.2f} MB)", end='')
        
        print(f"\nDownload completed: {downloaded/1024/1024:.2f} MB")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_360vots_sample(output_dir: Path = Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    urls = [
        {
            "name": "360VOTS_test",
            "url": "https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOTS_test.zip",
            "size": "~100MB"
        },
        {
            "name": "OTB_Bolt",
            "url": "https://cloud.tsinghua.edu.cn/f/6c52f2d5b9624a5e8c8a/?dl=1",
            "size": "~5MB"
        },
        {
            "name": "OpenCV_vtest",
            "url": "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi",
            "size": "~2MB"
        },
        {
            "name": "GOT10k_sample",
            "url": "https://drive.google.com/uc?export=download&id=1G228wU71kOq3E7g8dMRL82KX98H8q8dX",
            "size": "~50MB"
        }
    ]
    
    for item in urls:
        zip_path = output_dir / f"{item['name']}.zip" if ".zip" in item["url"].lower() else output_dir / f"{item['name']}.avi"
        
        if zip_path.exists():
            print(f"File {item['name']} already exists, skipping...")
            continue
        
        print(f"\nTrying to download {item['name']} ({item['size']})...")
        print(f"URL: {item['url']}")
        
        if download_file(item["url"], zip_path):
            if zip_path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(str(output_dir))
                    print(f"Extracted to {output_dir}")
                    os.remove(str(zip_path))
                except Exception as e:
                    print(f"Failed to extract zip: {e}")
            
            print(f"\nSuccess! File saved to {zip_path}")
            return item["name"], str(zip_path)
    
    print("\nAll downloads failed.")
    return None, None


if __name__ == "__main__":
    name, path = download_360vots_sample()
    if name:
        print(f"\nDownload successful!")
        print(f"File: {name}")
        print(f"Path: {path}")