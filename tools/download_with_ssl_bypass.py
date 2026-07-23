from __future__ import annotations

import requests
import os
import zipfile
from pathlib import Path
import ssl


def download_file_ssl_bypass(url: str, save_path: Path, timeout: int = 60):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
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
        return False


def download_real_tracking_video(output_dir: Path = Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    urls = [
        {
            "name": "OpenCV_vtest",
            "url": "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi",
            "expected_size": 2
        },
        {
            "name": "OTB_David",
            "url": "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/David.zip",
            "expected_size": 30
        },
        {
            "name": "OTB_Bolt",
            "url": "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Bolt.zip",
            "expected_size": 50
        },
        {
            "name": "OTB_Socc",
            "url": "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Socc.zip",
            "expected_size": 20
        }
    ]
    
    for item in urls:
        ext = ".zip" if ".zip" in item["url"].lower() else ".avi"
        save_path = output_dir / f"{item['name']}{ext}"
        
        if save_path.exists():
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            if size_mb > item["expected_size"] * 0.5:
                print(f"File {item['name']} already exists ({size_mb:.1f} MB), skipping...")
                continue
        
        print(f"\nTrying to download {item['name']} (~{item['expected_size']} MB)...")
        print(f"URL: {item['url']}")
        
        if download_file_ssl_bypass(item["url"], save_path):
            actual_size = os.path.getsize(save_path) / 1024 / 1024
            
            if actual_size < item["expected_size"] * 0.5:
                print(f"Warning: File size ({actual_size:.2f} MB) is smaller than expected")
                os.remove(str(save_path))
                continue
            
            if save_path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(str(output_dir))
                    print(f"Extracted to {output_dir}")
                    os.remove(str(save_path))
                    seq_name = item["name"].replace("OTB_", "")
                    seq_dir = output_dir / seq_name
                    if seq_dir.exists():
                        print(f"Sequence saved to {seq_dir}")
                except Exception as e:
                    print(f"Failed to extract zip: {e}")
            
            print(f"\nSuccess! File saved to {save_path}")
            return item["name"], str(save_path)
    
    print("\nAll downloads failed.")
    return None, None


if __name__ == "__main__":
    name, path = download_real_tracking_video()
    if name:
        print(f"\nDownload successful!")
        print(f"File: {name}")
        print(f"Path: {path}")