"""
textmask.py — erase the TEXT, not the rectangle around it.

The single biggest quality problem with box-based removal is that the box covers
whatever is behind the watermark — a face, a mouth, a product — and repainting that
whole rectangle destroys it. Watermark text, however, is a set of thin, high-contrast
strokes. This module isolates exactly those strokes so downstream inpainting only ever
touches a few pixels of the subject and the rest is left bit-for-bit untouched.

Method: text strokes are the high-frequency structure inside the box (skin, sky, walls
are smooth). A high-pass filter (frame minus a blurred copy of itself) lights the
strokes up and leaves smooth subjects near zero; an Otsu threshold turns that into a
tight glyph mask. When several frames are supplied, only structure that is *stable*
across them is kept, which drops moving subject detail (a talking mouth, hair) while
the watermark — which we crop at its tracked position, so it is the one thing that
stays put — survives the vote.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def highpass(gray: np.ndarray, k: int = 15) -> np.ndarray:
    """Local-contrast map: bright on thin strokes, ~0 on smooth areas."""
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.absdiff(gray, blur)


def _clean(mask: np.ndarray, min_frac: float = 0.002) -> np.ndarray:
    """Drop specks, bridge tiny stroke gaps, fatten strokes by one pixel."""
    n = mask.size
    if mask.sum() == 0:
        return mask
    # remove isolated specks smaller than a fraction of the box
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(8, int(n * min_frac))
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    # close small breaks within strokes, then a 1px safety dilation
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def text_mask(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    frames: Optional[list[np.ndarray]] = None,
    boxes: Optional[list[tuple[int, int, int, int]]] = None,
    stability: float = 0.6,
    max_area_frac: float = 0.5,
) -> tuple[np.ndarray, bool]:
    """Return (full-frame uint8 mask tight to the glyphs, whether detection was clean).

    `frames`/`boxes` (optional) are neighbouring frames cropped at their tracked boxes;
    structure that is not present in most of them is treated as moving subject detail
    and discarded, so a talking mouth or waving hand is not mistaken for text.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = (int(v) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((h, w), np.uint8), False

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    crop = gray[y0:y1, x0:x1]
    hp = highpass(crop)

    thr, base = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 6:  # essentially flat — nothing to isolate
        return np.zeros((h, w), np.uint8), False
    # Otsu alone over-thresholds and leaves half the glyph behind (thin anti-aliased
    # strokes over a dark subject read as weak high-pass). Clamp it to a band that
    # keeps recall high; the stability vote and speck-clean below handle precision.
    thr = float(np.clip(thr, 8, 12))
    m = (hp >= thr).astype(np.uint8) * 255

    # Temporal stability vote: keep only strokes seen in most neighbour frames.
    if frames and boxes and len(frames) >= 3:
        votes = np.zeros_like(m, np.int16)
        total = 0
        ch, cw = m.shape
        for nf, nb_ in zip(frames, boxes):
            nx, ny, nw, nh = (int(v) for v in nb_)
            ng = cv2.cvtColor(nf, cv2.COLOR_RGB2GRAY)
            nc = ng[max(0, ny):max(0, ny) + ch, max(0, nx):max(0, nx) + cw]
            if nc.shape != (ch, cw):
                nc = cv2.resize(nc, (cw, ch), interpolation=cv2.INTER_AREA)
            nhp = highpass(nc)
            votes += (nhp >= max(thr, 10)).astype(np.int16)
            total += 1
        if total:
            m = ((votes / total) >= stability).astype(np.uint8) * 255

    m = _clean(m)

    npx = int((m > 0).sum())
    area_frac = float(npx) / max(1, m.size)
    if npx < 24 or area_frac > max_area_frac:
        # Too much of the box lit up: this is not clean text over a smooth subject,
        # so a tight mask cannot be trusted. Caller should fall back.
        return np.zeros((h, w), np.uint8), False

    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = m
    return full, True


def feather(mask: np.ndarray, px: int = 2) -> np.ndarray:
    if px <= 0:
        return mask
    k = int(px) * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)
