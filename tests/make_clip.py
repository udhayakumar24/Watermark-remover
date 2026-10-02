#!/usr/bin/env python3
"""
make_clip.py — synthesise a test clip with a MOVING watermark.

The watermark travels on a Lissajous path over an animated (non-static) background,
which is exactly the hard case: a fixed box will not work, and temporal filling has
to cope with background motion too. Ground-truth box positions are written next to
the video so the tracker can be scored.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.ff import ffmpeg_exe  # noqa: E402

W, H = 320, 180
FPS = 25
SECONDS = 2.0
TEXT = "SAMPLE"
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.72
THICK = 2


def _text_size():
    (tw, th), baseline = cv2.getTextSize(TEXT, FONT, SCALE, THICK)
    return tw, th + baseline


def watermark_center(t: float) -> tuple[float, float]:
    """Lissajous path across the frame."""
    cx = W / 2 + (W / 2 - 60) * math.sin(2 * math.pi * 0.5 * t)
    cy = H / 2 + (H / 2 - 34) * math.sin(2 * math.pi * 0.33 * t + 1.1)
    return cx, cy


def background(t: float) -> np.ndarray:
    """Animated gradient with drifting blobs — deliberately NOT static."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 40 + 60 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.4 * t))
    g = 90 + 70 * (0.5 + 0.5 * math.cos(2 * math.pi * 0.25 * t + 1.0))
    b = 120 + 80 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.3 * t + 2.0))
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = r
    img[..., 1] = g * (0.4 + 0.6 * (x / W))
    img[..., 2] = b * (0.4 + 0.6 * (y / H))

    for k in range(3):
        bx = (W * (0.2 + 0.3 * k) + 40 * math.sin(2 * math.pi * 0.5 * t + k)) % W
        by = (H * (0.3 + 0.25 * k) + 30 * math.cos(2 * math.pi * 0.4 * t + k)) % H
        cv2.circle(img, (int(bx), int(by)), 26 + 8 * k, (250, 240, 200), -1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def stamp_watermark(img: np.ndarray, cx: float, cy: float, alpha: float = 0.65) -> tuple:
    tw, th = _text_size()
    x0 = int(round(cx - tw / 2))
    y0 = int(round(cy - th / 2))
    x1, y1 = x0 + tw, y0 + th

    tx0, ty0 = max(0, x0), max(0, y0)
    tx1, ty1 = min(W, x1), min(H, y1)
    if tx1 <= tx0 or ty1 <= ty0:
        return (x0, y0, tw, th)

    overlay = img.copy()
    cv2.putText(overlay, TEXT, (tx0 + 1, ty1 - 5), FONT, SCALE, (255, 255, 255), THICK, cv2.LINE_AA)
    cv2.rectangle(overlay, (tx0 + 1, ty1 - 3), (tx1 - 2, ty1 - 1), (255, 255, 255), 1, cv2.LINE_AA)
    img[ty0:ty1, tx0:tx1] = cv2.addWeighted(
        overlay[ty0:ty1, tx0:tx1], alpha, img[ty0:ty1, tx0:tx1], 1 - alpha, 0
    )
    return (x0, y0, tw, th)


def _encode(video: Path, gen, n: int, audio: bool = True) -> None:
    """Pipe `n` BGR frames from gen(i) into an H.264+AAC file."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency=440:sample_rate=44100:duration={n / FPS:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [str(video)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(gen(i).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed while building a test clip")


# --------------------------------------------------------------------------- #
# Clip A — chaotic: oscillating colours plus three independently moving blobs.
# No single motion model explains it, so temporal filling is at the mercy of how
# fast the background itself changes. Deliberately pessimistic.
# --------------------------------------------------------------------------- #

def main(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "clip.mp4"
    n = int(round(FPS * SECONDS))

    gt = []

    def gen(i):
        t = i / FPS
        frame = background(t)
        cx, cy = watermark_center(t)
        box = stamp_watermark(frame, cx, cy)
        gt.append({"frame": i, "t": round(t, 4), "box": [int(v) for v in box],
                   "center": [round(cx, 2), round(cy, 2)]})
        return frame

    _encode(video, gen, n)
    (out_dir / "clip_gt.json").write_text(json.dumps({
        "w": W, "h": H, "fps": FPS, "frames": n, "gt": gt,
    }))
    return video


# --------------------------------------------------------------------------- #
# Clip B — realistic: a detailed static scene filmed with a steady camera pan.
# One global translation explains all the motion, which is the common case for real
# footage and exactly what phase-correlation alignment is built for. Temporal
# filling should recover the true background almost exactly here.
# --------------------------------------------------------------------------- #

PAN_PX_PER_FRAME = 4.0
_SCENE_CACHE: dict = {}


def _scene() -> np.ndarray:
    if "s" in _SCENE_CACHE:
        return _SCENE_CACHE["s"]
    wide = W + int(PAN_PX_PER_FRAME * FPS * SECONDS) + 8
    rng = np.random.default_rng(1234)
    base = rng.integers(40, 210, size=(H, wide, 3), dtype=np.uint8).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 2.2)          # smooth, detailed backdrop
    for k in range(26):                                 # hard-edged detail to key on
        x = int(rng.integers(0, wide - 20))
        y = int(rng.integers(0, H - 14))
        col = tuple(int(v) for v in rng.integers(20, 255, size=3))
        cv2.rectangle(base, (x, y), (x + 14, y + 9), col, -1)
    for k in range(6):
        cv2.circle(base, (int(rng.integers(10, wide - 10)), int(rng.integers(10, H - 10))),
                   int(rng.integers(6, 16)), (235, 235, 235), 2, cv2.LINE_AA)
    scene = np.clip(base, 0, 255).astype(np.uint8)
    _SCENE_CACHE["s"] = scene
    return scene


def pan_background(i: int) -> np.ndarray:
    x0 = int(round(i * PAN_PX_PER_FRAME))
    return _scene()[:, x0:x0 + W].copy()


def main_pan(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "clip_pan.mp4"
    n = int(round(FPS * SECONDS))
    gt = []

    def gen(i):
        t = i / FPS
        frame = pan_background(i)
        cx, cy = watermark_center(t)
        box = stamp_watermark(frame, cx, cy)
        gt.append({"frame": i, "t": round(t, 4), "box": [int(v) for v in box],
                   "center": [round(cx, 2), round(cy, 2)]})
        return frame

    _encode(video, gen, n)
    (out_dir / "clip_pan_gt.json").write_text(json.dumps({
        "w": W, "h": H, "fps": FPS, "frames": n, "gt": gt, "pan_px_per_frame": PAN_PX_PER_FRAME,
    }))
    return video



if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "fixtures"
    p = main(target)
    print(p)
