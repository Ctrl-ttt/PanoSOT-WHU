"""数据集与视频下载工具。

支持多种数据源：
  - 360VOTS 官方数据集（GitHub Release / 官网）
  - Pexels 360° 全景视频
  - 合成全景数据（网络不可用时的替代方案）

用法:
  python tools/download_dataset.py --source 360vots
  python tools/download_dataset.py --source pexels
  python tools/download_dataset.py --source synthetic
"""
from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 下载辅助
# ---------------------------------------------------------------------------

def _download_urllib(url: str, save_path: Path, timeout: int = 300) -> bool:
    """使用 urllib 下载文件，禁用 SSL 验证以兼容受限网络。"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)

        response = urllib.request.urlopen(url, timeout=timeout)
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(save_path, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  进度: {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end="")

        print(f"\n  下载完成: {downloaded/1024/1024:.1f} MB")
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def _download_powershell(url: str, save_path: Path, timeout: int = 300) -> bool:
    """使用 PowerShell 的 Invoke-WebRequest 下载。"""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"Invoke-WebRequest -Uri '{url}' -OutFile '{save_path}' -TimeoutSec {timeout}"],
            capture_output=True, text=True, timeout=timeout + 60,
        )
        if result.returncode == 0 and save_path.exists():
            print(f"  下载完成: {save_path.stat().st_size/1024/1024:.1f} MB")
            return True
        print(f"  PowerShell 下载失败")
        return False
    except Exception as e:
        print(f"  PowerShell 下载异常: {e}")
        return False


def download_file(url: str, save_path: Path, timeout: int = 300) -> bool:
    """尝试多种方式下载文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  URL: {url}")
    if _download_urllib(url, save_path, timeout):
        return True
    if _download_powershell(url, save_path, timeout):
        return True
    return False


def _extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """解压 zip 文件并删除压缩包。"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(output_dir))
        print(f"  解压完成: {output_dir}")
        zip_path.unlink()
        return True
    except Exception as e:
        print(f"  解压失败: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


# ---------------------------------------------------------------------------
# 各数据源下载逻辑
# ---------------------------------------------------------------------------

# GitHub 代理列表（国内加速）
_GH_PROXIES = [
    "https://gh.api.99988866.xyz/",
    "https://ghproxy.com/",
    "",
]

_SOURCES_360VOTS = [
    {
        "name": "360VOT_test",
        "gh_path": "360-VOTS/360VOTS/releases/download/v1.0/360VOT_test.zip",
        "expected_mb": 500,
    },
    {
        "name": "360VOT_train",
        "gh_path": "360-VOTS/360VOTS/releases/download/v1.0/360VOT_train.zip",
        "expected_mb": 2000,
    },
]


def download_360vots(output_dir: Path) -> bool:
    """下载 360VOTS 数据集。"""
    print("\n=== 下载 360VOTS 数据集 ===")
    for item in _SOURCES_360VOTS:
        save_path = output_dir / f"{item['name']}.zip"
        if save_path.exists() and save_path.stat().st_size > item["expected_mb"] * 1024 * 1024 * 0.3:
            print(f"  {item['name']} 已存在，跳过")
            continue
        for proxy in _GH_PROXIES:
            url = f"{proxy}https://github.com/{item['gh_path']}"
            print(f"\n  尝试下载 {item['name']} ...")
            if download_file(url, save_path):
                _extract_zip(save_path, output_dir)
                return True
    print("  ❌ 所有下载源均失败")
    print("  请手动访问 https://360vots.hkustvgd.com/ 下载数据集")
    return False


def download_pexels_360(output_dir: Path, video_id: str = "36157408") -> bool:
    """下载 Pexels 360° 全景视频。"""
    print(f"\n=== 下载 Pexels 360° 视频 (ID: {video_id}) ===")
    save_path = output_dir / "external_videos" / f"tracking_pexels_360_{video_id}.mp4"
    if save_path.exists():
        print(f"  视频已存在: {save_path}")
        return True

    url = f"https://www.pexels.com/video/{video_id}/download/"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if download_file(url, save_path):
        print(f"  ✅ 下载成功: {save_path}")
        return True

    print("  ❌ 下载失败，请手动从 https://www.pexels.com/search/videos/360/ 下载")
    return False


def generate_synthetic_pano(output_dir: Path) -> bool:
    """生成合成全景视频（网络不可用时的替代方案）。"""
    print("\n=== 生成合成全景测试数据 ===")
    script = PROJECT_ROOT / "examples" / "generate_test_data.py"
    if not script.exists():
        print(f"  脚本不存在: {script}")
        return False
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output_dir)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print("  ✅ 合成数据生成成功")
        return True
    print(f"  ❌ 生成失败: {result.stderr[:200]}")
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PanoSOT 数据集下载工具")
    parser.add_argument(
        "--source", choices=["360vots", "pexels", "synthetic", "all"],
        default="all", help="数据源 (默认: all)",
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data"), help="输出目录")
    parser.add_argument("--pexels-id", default="36157408", help="Pexels 视频 ID")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.source in ("360vots", "all"):
        results["360vots"] = download_360vots(output_dir)
    if args.source in ("pexels", "all"):
        results["pexels"] = download_pexels_360(output_dir, args.pexels_id)
    if args.source in ("synthetic", "all"):
        results["synthetic"] = generate_synthetic_pano(output_dir)

    print("\n=== 下载结果 ===")
    for name, ok in results.items():
        print(f"  {name}: {'✅ 成功' if ok else '❌ 失败'}")


if __name__ == "__main__":
    main()
