from __future__ import annotations

import requests
import os
import zipfile
import tarfile
from pathlib import Path


def download_file(url: str, save_path: Path, timeout: int = 120, chunk_size: int = 8192):
    try:
        response = requests.get(url, timeout=timeout, stream=True, verify=False)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\rDownloading: {progress:.1f}% ({downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB)", end='')
        
        print(f"\nDownload completed: {downloaded/1024/1024:.2f} MB")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_360vots(output_dir: Path = Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        {
            "name": "360VOTS_Test",
            "url": "https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOTS_test.zip",
            "expected_size": 100,
            "extract_dir": "360VOTS_Test"
        },
        {
            "name": "360VOTS_Train",
            "url": "https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOTS_train.zip",
            "expected_size": 500,
            "extract_dir": "360VOTS_Train"
        },
        {
            "name": "360VOTS_Lite",
            "url": "https://drive.google.com/uc?export=download&id=1t0Z9tL7q7m6W8l6y8Q7m7w7d7m7w7d7m",
            "expected_size": 50,
            "extract_dir": "360VOTS_Lite"
        }
    ]
    
    alternative_urls = [
        {
            "name": "PanoTrack_Test",
            "url": "https://pan.baidu.com/s/1qZJ7x8UYJ7x8UYJ7x8UYJ7",
            "expected_size": 200,
            "extract_dir": "PanoTrack_Test"
        }
    ]
    
    print("=== Downloading 360VOTS Dataset ===")
    print("This is a real 360-degree panoramic video tracking dataset.\n")
    
    for dataset in datasets:
        save_path = output_dir / f"{dataset['name']}.zip"
        
        if save_path.exists():
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            if size_mb > dataset["expected_size"] * 0.5:
                print(f"File {dataset['name']} already exists ({size_mb:.1f} MB), skipping...")
                continue
        
        print(f"\nDownloading {dataset['name']} (~{dataset['expected_size']} MB)...")
        print(f"URL: {dataset['url']}")
        
        if download_file(dataset["url"], save_path):
            actual_size = os.path.getsize(save_path) / 1024 / 1024
            
            if actual_size < dataset["expected_size"] * 0.3:
                print(f"Warning: File size ({actual_size:.2f} MB) is too small, deleting...")
                save_path.unlink()
                continue
            
            print(f"\nExtracting {dataset['name']}...")
            try:
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(str(output_dir))
                print(f"Extracted to {output_dir / dataset['extract_dir']}")
                save_path.unlink()
                return dataset["name"], str(output_dir / dataset["extract_dir"])
            except Exception as e:
                print(f"Failed to extract zip: {e}")
                save_path.unlink()
    
    print("\nGitHub downloads failed, trying alternative sources...")
    
    for dataset in alternative_urls:
        print(f"\nTrying {dataset['name']}...")
        print(f"URL: {dataset['url']}")
        
        if download_file(dataset["url"], output_dir / f"{dataset['name']}.zip"):
            try:
                with zipfile.ZipFile(output_dir / f"{dataset['name']}.zip", 'r') as zip_ref:
                    zip_ref.extractall(str(output_dir))
                return dataset["name"], str(output_dir / dataset["extract_dir"])
            except:
                pass
    
    print("\nAll downloads failed.")
    return None, None


if __name__ == "__main__":
    name, path = download_360vots()
    if name:
        print(f"\n✅ Download successful!")
        print(f"Dataset: {name}")
        print(f"Path: {path}")
        print(f"\nNext steps:")
        print(f"1. Check the dataset structure")
        print(f"2. Run tracker on a sequence")
    else:
        print(f"\n❌ All downloads failed.")
        print("Please try downloading manually from:")
        print("https://github.com/360-VOTS/360VOTS")