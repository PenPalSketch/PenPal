"""Download SketchRNN .npz files for each category and pre-rasterize them.

Caches bitmaps as uint8 .npy files under data/cache/ so train.py can skip
rasterization on every epoch.

Usage:
    python prepare.py                     # default 10 categories, 30k/class
    python prepare.py --per-class 10000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from rasterize import IMG_SIZE, rasterize


DEFAULT_CATEGORIES = [
    "cat", "dog", "rabbit", "mouse", "lion",
    "pig", "bird", "fish", "elephant", "horse",
]

BUCKET_URL = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{name}.npz"

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"


def download_npz(category: str) -> Path:
    """Fetch the .npz for a category. Uses local sketchrnn_cat.npz if present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / f"sketchrnn_{category}.npz"
    if dst.exists():
        return dst

    # Reuse the sketchrnn_cat.npz the user already has.
    local = ROOT / f"sketchrnn_{category}.npz"
    if local.exists():
        print(f"[{category}] copying local file")
        dst.write_bytes(local.read_bytes())
        return dst

    url = BUCKET_URL.format(name=category)
    print(f"[{category}] downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                f.write(chunk)
                pbar.update(len(chunk))
    return dst


def build_cache(category: str, per_class: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{category}_{IMG_SIZE}.npy"
    if cache_file.exists():
        existing = np.load(cache_file, mmap_mode="r")
        if existing.shape[0] >= per_class:
            print(f"[{category}] cache already has {existing.shape[0]} samples — skipping")
            return

    npz_path = download_npz(category)
    data = np.load(npz_path, allow_pickle=True, encoding="latin1")
    train = data["train"]
    n = min(per_class, len(train))
    print(f"[{category}] rasterizing {n} of {len(train)} samples -> {cache_file.name}")

    out = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in tqdm(range(n), unit="img"):
        out[i] = rasterize(train[i])
    np.save(cache_file, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--per-class", type=int, default=30000)
    args = ap.parse_args()

    for cat in args.categories:
        build_cache(cat, args.per_class)

    # Write the class list so train/compare share the same ordering.
    (ROOT / "data" / "classes.txt").write_text("\n".join(args.categories) + "\n")
    print(f"\nready. classes: {args.categories}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
