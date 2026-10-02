#!/usr/bin/env python3
"""
fetch_model.py — download the LaMa inpainting model (Apache-2.0) and verify it.

Run once:  python scripts/fetch_model.py
Skips the download if a valid copy is already in models/.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEST = MODEL_DIR / "lama.onnx"

SOURCES = [
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
]
# Published checksum for the opencv/inpainting_lama export.
SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"
MIN_BYTES = 40 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if DEST.exists() and DEST.stat().st_size >= MIN_BYTES:
        digest = sha256(DEST)
        if digest == SHA256:
            print(f"Already present and verified: {DEST} ({DEST.stat().st_size // 1048576} MB)")
            return 0
        print("Existing file failed the checksum — re-downloading.")
        DEST.unlink()

    for url in SOURCES:
        print(f"Downloading {url}\n  …")
        tmp = DEST.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as out:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done // total
                        print(f"\r  {done // 1048576}/{total // 1048576} MB ({pct}%)", end="", flush=True)
            print()
        except Exception as exc:
            print(f"  failed: {exc}")
            tmp.unlink(missing_ok=True)
            continue

        if tmp.stat().st_size < MIN_BYTES:
            print("  download too small, trying the next mirror.")
            tmp.unlink(missing_ok=True)
            continue

        digest = sha256(tmp)
        if digest != SHA256:
            print(f"  checksum mismatch ({digest[:16]}…), trying the next mirror.")
            tmp.unlink(missing_ok=True)
            continue

        tmp.rename(DEST)
        print(f"Verified and saved: {DEST}")
        return 0

    print("Could not obtain the model. You can still use the temporal, diffusion")
    print("and blur modes; only 'Neural (LaMa)' needs this file.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
