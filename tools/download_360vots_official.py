from __future__ import annotations

import requests
import os
import zipfile
from pathlib import Path


def download_file(url: str, save_path: Path, timeout: int = 300, chunk_size: int = 8192):
    try:
        session = requests.Session()
        session.verify = False
        
        response = session.get(url, timeout=timeout, stream=True)
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
                        print(f"\rDownloading: {progress:.1f}% ({downloaded/1024/1024:.2f} MB)", end='')
        
        print(f"\nDownload completed: {downloaded/1024/1024:.2f} MB")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_360vots_official(output_dir: Path = Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://360vots.hkustvgd.com/dataset/"
    
    datasets = [
        {
            "name": "360VOT_Test",
            "url": base_url + "360VOT_test.zip",
            "expected_size": 500,
            "extract_dir": "360VOT_test"
        },
        {
            "name": "360VOT_Train",
            "url": base_url + "360VOT_train.zip",
            "expected_size": 2000,
            "extract_dir": "360VOT_train"
        },
        {
            "name": "360VOS_Test",
            "url": base_url + "360VOS_test.zip",
            "expected_size": 1000,
            "extract_dir": "360VOS_test"
        },
        {
            "name": "360VOS_Train",
            "url": base_url + "360VOS_train.zip",
            "expected_size": 5000,
            "extract_dir": "360VOS_train"
        }
    ]
    
    print("=== Downloading 360VOTS Dataset from Official Website ===")
    print("Official site: https://360vots.hkustvgd.com/\n")
    
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
            
            if actual_size < 10:
                print(f"Warning: File size ({actual_size:.2f} MB) is too small")
                
                try:
                    with open(save_path, 'r') as f:
                        content = f.read()
                        if "404" in content or "Not Found" in content:
                            print("The file is a 404 error page, deleting...")
                            save_path.unlink()
                            continue
                except:
                    pass
            
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
    
    print("\nAll downloads failed.")
    print("\nAlternative download methods:")
    print("1. Visit the official website: https://360vots.hkustvgd.com/")
    print("2. Download manually from: https://github.com/360-VOTS/360VOTS")
    print("3. Use Baidu Netdisk if available")
    
    return None, None


if __name__ == "__main__":
    name, path = download_360vots_official()
    if name:
        print(f"\n✅ Download successful!")
        print(f"Dataset: {name}")
        print(f"Path: {path}")
    else:
        print(f"\n❌ All downloads failed.")