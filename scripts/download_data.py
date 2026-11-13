"""Download the EA Sports FC 24 Complete Player Dataset into data/raw/.

The dataset is public, so Kaggle's download endpoint works without an account
or API key. Only ``male_players.csv`` is extracted, because that is the only
table the study uses.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET = "stefanoleone992/ea-sports-fc-24-complete-player-dataset"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
WANTED = "male_players.csv"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / WANTED
    if target.exists():
        print(f"{target} already exists, nothing to do.")
        return 0

    archive = RAW_DIR / "ea-sports-fc-24-complete-player-dataset.zip"
    if not archive.exists():
        print(f"Downloading {DATASET} ...")
        download(URL, archive)

    with zipfile.ZipFile(archive) as zf:
        print("Archive contents:")
        for info in zf.infolist():
            print(f"  {info.filename:28s} {info.file_size / 1e6:9.1f} MB")
        if WANTED not in zf.namelist():
            print(f"ERROR: {WANTED} not found in the archive.", file=sys.stderr)
            return 1
        zf.extract(WANTED, RAW_DIR)

    print(f"Extracted {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
