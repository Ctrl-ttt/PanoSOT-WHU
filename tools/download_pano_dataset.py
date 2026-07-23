from __future__ import annotations

import requests
import os
import zipfile
from pathlib import Path


def download_file(url: str, save_path: Path, timeout: int = 300):
    try:
        session = requests.Session()
        session.verify = False
        
        response = session.get(url, timeout=timeout, stream=True)
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
        if save_path.exists():
            save_path.unlink()
        return False


def download_pano_dataset(output_dir: Path = Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        {
            "name": "360VOT_Test",
            "url": "https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOT_test.zip",
            "expected_size": 500
        },
        {
            "name": "360VOT_Train",
            "url": "https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOT_train.zip",
            "expected_size": 2000
        },
        {
            "name": "360VOS_Test",
            "url": "https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOS_test.zip",
            "expected_size": 1000
        },
        {
            "name": "PanoTrack_Sample",
            "url": "https://www.cse.cuhk.edu.hk/~leojia/projects/panotrack/dataset/PanoTrack_test.zip",
            "expected_size": 200
        },
        {
            "name": "OmniTrack_Sample",
            "url": "https://github.com/yuhuan-wu/OmniTrack/releases/download/v1.0/sample_data.zip",
            "expected_size": 50
        },
        {
            "name": "OpenPano_Sample",
            "url": "https://github.com/facebookresearch/OpenPano/releases/download/v1.0/sample_images.zip",
            "expected_size": 50
        }
    ]
    
    print("=== Downloading 360° Panoramic Video Dataset ===")
    print("Trying multiple sources...\n")
    
    for dataset in datasets:
        save_path = output_dir / f"{dataset['name']}.zip"
        
        if save_path.exists():
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            if size_mb > dataset["expected_size"] * 0.3:
                print(f"File {dataset['name']} already exists ({size_mb:.1f} MB), skipping...")
                continue
        
        print(f"\nTrying {dataset['name']}...")
        print(f"URL: {dataset['url']}")
        
        if download_file(dataset["url"], save_path):
            actual_size = os.path.getsize(save_path) / 1024 / 1024
            
            if actual_size < 5:
                print(f"Warning: File too small ({actual_size:.2f} MB), checking if it's an HTML error...")
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "404" in content or "Not Found" in content or "<html" in content.lower():
                            print("This is an HTML error page, deleting...")
                            save_path.unlink()
                            continue
                except:
                    pass
            
            print(f"\nExtracting {dataset['name']}...")
            try:
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(str(output_dir))
                print(f"Extracted successfully!")
                save_path.unlink()
                
                extracted_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
                if extracted_dirs:
                    print(f"Extracted to: {extracted_dirs[-1]}")
                    return dataset["name"], str(extracted_dirs[-1])
                else:
                    print(f"Extracted to: {output_dir}")
                    return dataset["name"], str(output_dir)
            except Exception as e:
                print(f"Failed to extract: {e}")
                save_path.unlink()
    
    print("\n❌ All downloads failed.")
    print("\nManual download options:")
    print("1. 360VOTS official: https://360vots.hkustvgd.com/")
    print("2. GitHub repo: https://github.com/360-VOTS/360VOTS")
    print("3. PanoTrack: https://www.cse.cuhk.edu.hk/~leojia/projects/panotrack")
    
    return None, None


if __name__ == "__main__":
    name, path = download_pano_dataset()
    if name:
        print(f"\n✅ Download successful!")
        print(f"Dataset: {name}")
        print(f"Path: {path}")