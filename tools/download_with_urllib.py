from __future__ import annotations

import urllib.request
import os
import ssl
import zipfile
from pathlib import Path


def download_file_urllib(url: str, save_path: Path, timeout: int = 300):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
        
        response = urllib.request.urlopen(url, timeout=timeout)
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
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
            "url": "https://gh.api.99988866.xyz/https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOT_test.zip",
            "expected_size": 500
        },
        {
            "name": "360VOT_Train",
            "url": "https://gh.api.99988866.xyz/https://github.com/360-VOTS/360VOTS/releases/download/v1.0/360VOT_train.zip",
            "expected_size": 2000
        },
        {
            "name": "Sample_Pano_Images",
            "url": "https://raw.githubusercontent.com/facebookresearch/OpenPano/master/samples/equirectangular/input.jpg",
            "expected_size": 1
        },
        {
            "name": "PanoTrack_Demo",
            "url": "https://ghproxy.com/https://github.com/leoxiaobin/panotrack/releases/download/v1.0/demo_data.zip",
            "expected_size": 100
        }
    ]
    
    print("=== Downloading 360° Panoramic Dataset ===")
    print("Using GitHub proxy and urllib...\n")
    
    for dataset in datasets:
        ext = ".zip" if ".zip" in dataset["url"].lower() else ".jpg"
        save_path = output_dir / f"{dataset['name']}{ext}"
        
        if save_path.exists():
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            if size_mb > dataset["expected_size"] * 0.3:
                print(f"File {dataset['name']} already exists ({size_mb:.1f} MB), skipping...")
                continue
        
        print(f"\nTrying {dataset['name']}...")
        print(f"URL: {dataset['url']}")
        
        if download_file_urllib(dataset["url"], save_path):
            actual_size = os.path.getsize(save_path) / 1024 / 1024
            
            if actual_size < 1 and ext == ".zip":
                print(f"Warning: File too small ({actual_size:.2f} MB)")
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "404" in content or "<html" in content.lower():
                            print("This is an HTML error page, deleting...")
                            save_path.unlink()
                            continue
                except:
                    pass
            
            if ext == ".zip":
                print(f"\nExtracting {dataset['name']}...")
                try:
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(str(output_dir))
                    print(f"Extracted successfully!")
                    save_path.unlink()
                    return dataset["name"], str(output_dir)
                except Exception as e:
                    print(f"Failed to extract: {e}")
                    save_path.unlink()
            else:
                print(f"\nSaved to: {save_path}")
                return dataset["name"], str(save_path)
    
    print("\n❌ All downloads failed.")
    print("\nManual download options:")
    print("1. Visit: https://360vots.hkustvgd.com/")
    print("2. GitHub: https://github.com/360-VOTS/360VOTS")
    
    return None, None


if __name__ == "__main__":
    name, path = download_pano_dataset()
    if name:
        print(f"\n✅ Download successful!")
        print(f"Dataset: {name}")
        print(f"Path: {path}")