from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import httpx
from huggingface_hub import HfApi


REPO_ID = "xuyzshaun/360VOTS"
REPO_TYPE = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download 360VOT-test split directly from Hugging Face.")
    parser.add_argument("--token", required=True, help="Hugging Face read token.")
    parser.add_argument("--output-dir", default=r"D:\360VOTS", help="Target root directory.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum number of missing zip files to download in this run.",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        help="Optional filename like 0003.zip to start from.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pause between files to reduce retry pressure.",
    )
    parser.add_argument(
        "--missing-only-from",
        default=None,
        help="Optional existing 360VOT-test directory used to skip files already downloaded elsewhere.",
    )
    return parser.parse_args()


def list_files(token: str) -> list[str]:
    api = HfApi(token=token)
    return sorted(
        f
        for f in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        if f.startswith("360VOT-test/") and f.endswith(".zip")
    )


def direct_download(token: str, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/360VOT-test/{filename}?download=true"
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)

    tmp_path.replace(out_path)
    return out_path


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir)
    target_dir = root / "360VOT-test"
    target_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(f).name for f in list_files(args.token)]
    done = {p.name for p in target_dir.glob("*.zip")}

    if args.start_from:
        files = [name for name in files if name >= args.start_from]

    if args.missing_only_from:
        existing_dir = Path(args.missing_only_from)
        done.update(p.name for p in existing_dir.glob("*.zip"))

    pending = [name for name in files if name not in done]
    print(f"already_downloaded={len(done)} pending={len(pending)}")

    for index, name in enumerate(pending[: args.batch_size], start=1):
        path = direct_download(args.token, name, target_dir)
        size_gb = path.stat().st_size / (1024 ** 3)
        print(f"{index}/{min(len(pending), args.batch_size)} {path.name} {size_gb:.2f} GB")
        time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
