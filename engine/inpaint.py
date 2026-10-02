"""
inpaint.py — three ways to fill the hole left by a removed watermark.

`lama`     Neural inpainting (LaMa, ONNX). Reconstructs plausible texture from the
           surrounding context. Slowest, but works on arbitrary backgrounds.

`temporal` Rebuilds the hidden pixels from *other frames* of the same clip. Because
           the watermark moves, the pixels it hides at frame t are visible at frame
           t±k, so the true background can often be recovered exactly rather than
           hallucinated. Cheap, and usually the best result for moving logos.
           Camera motion is compensated with phase correlation.

`classic`  OpenCV Telea/Navier-Stokes diffusion inpainting. Fast fallback.

`auto`     temporal first, LaMa only for pixels temporal could not recover — the
           quality/speed sweet spot.

LaMa's ONNX graph has a fixed 512x512 input, so instead of squashing the whole frame
(the usual approach, which blurs the repair) we crop a square around the watermark,
run the model on that, and blend it back. The repair therefore happens at full
effective resolution.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

LAMA_SIZE = 512


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #

def make_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    margin: float = 0.25,
    feather: int = 3,
) -> np.ndarray:
    """Soft-edged uint8 mask (0/255) for a box expanded by `margin`."""
    h, w = shape[:2]
    x, y, bw, bh = (int(v) for v in box)
    mx = int(round(bw * margin))
    my = int(round(bh * margin))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + bw + mx)
    y1 = min(h, y + bh + my)

    mask = np.zeros((h, w), np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    if feather > 0:
        k = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


# --------------------------------------------------------------------------- #
# LaMa (neural)
# --------------------------------------------------------------------------- #

class LamaInpainter:
    """LaMa via ONNX Runtime. Thread-safe enough for single-threaded render loops."""

    _instances: dict[str, "LamaInpainter"] = {}
    _lock = threading.Lock()

    def __init__(self, model_path: str, threads: int = 0):
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa model not found at {model_path}")

        so = ort.SessionOptions()
        n = threads or max(1, (os.cpu_count() or 2) - 1)
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3

        self.session = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"]
        )
        self.in_names = [i.name for i in self.session.get_inputs()]
        self.model_path = model_path

    @classmethod
    def get(cls, model_path: str, threads: int = 0) -> "LamaInpainter":
        with cls._lock:
            key = os.path.abspath(model_path)
            if key not in cls._instances:
                cls._instances[key] = cls(key, threads)
            return cls._instances[key]

    # ------------------------------------------------------------------ #

    def fill(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        pad_ratio: float = 1.2,
        min_side: int = 96,
    ) -> tuple[np.ndarray, bool]:
        """Return (repaired frame, whether anything was done).

        Only the masked pixels are replaced; the rest of the frame is untouched, so
        there is no global quality loss from the round-trip through the network.
        """
        ys, xs = np.nonzero(mask > 40)
        if len(xs) == 0:
            return frame, False

        h, w = frame.shape[:2]
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        side = max(min_side, int(np.ceil(max(bw, bh) * (1.0 + pad_ratio))))
        side = int(min(side, max(h, w) * 2))

        half = side / 2.0
        sx0 = int(round(cx - half))
        sy0 = int(round(cy - half))

        # Reflect-pad if the square spills outside the frame.
        pad_l = max(0, -sx0)
        pad_t = max(0, -sy0)
        pad_r = max(0, sx0 + side - w)
        pad_b = max(0, sy0 + side - h)

        crop = frame[
            max(0, sy0):min(h, sy0 + side),
            max(0, sx0):min(w, sx0 + side),
        ]
        if crop.size == 0:
            return frame, False
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(
                crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
            )
        crop = crop[:side, :side]
        if crop.shape[0] != side or crop.shape[1] != side:
            crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)

        m_crop = mask[
            max(0, sy0):min(h, sy0 + side),
            max(0, sx0):min(w, sx0 + side),
        ]
        if pad_l or pad_t or pad_r or pad_b:
            m_crop = cv2.copyMakeBorder(
                m_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0
            )
        m_crop = m_crop[:side, :side]
        if m_crop.shape[0] != side or m_crop.shape[1] != side:
            m_crop = cv2.resize(m_crop, (side, side), interpolation=cv2.INTER_NEAREST)

        img_in = cv2.resize(crop, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
        msk_in = cv2.resize(m_crop, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
        msk_in = (msk_in > 40).astype(np.float32)
        if not msk_in.any():
            return frame, False

        img_t = (img_in.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        msk_t = msk_in[None, None]
        out = self.session.run(None, {"image": img_t, "mask": msk_t})[0]

        recon = np.transpose(out[0], (1, 2, 0))
        recon = np.clip(recon, 0, 255).astype(np.uint8)
        recon = cv2.resize(recon, (side, side), interpolation=cv2.INTER_CUBIC)

        # Map the repaired crop back onto the frame. When the square spilled off an
        # edge the crop was reflect-padded, so frame row 0 corresponds to crop row
        # pad_t — skipping that offset would paste the repair in the wrong place.
        y0s, y1s = max(0, sy0), min(h, sy0 + side)
        x0s, x1s = max(0, sx0), min(w, sx0 + side)
        if y1s <= y0s or x1s <= x0s:
            return frame, False
        off_t, off_l = y0s - sy0, x0s - sx0
        rh, rw = y1s - y0s, x1s - x0s

        alpha = m_crop[off_t:off_t + rh, off_l:off_l + rw].astype(np.float32) / 255.0
        alpha[alpha < 0.03] = 0.0          # leave untouched pixels byte-identical
        alpha = alpha[..., None]
        recon_sub = recon[off_t:off_t + rh, off_l:off_l + rw].astype(np.float32)

        out_full = frame.copy()
        region = out_full[y0s:y1s, x0s:x1s].astype(np.float32)
        out_full[y0s:y1s, x0s:x1s] = np.clip(
            region * (1 - alpha) + recon_sub * alpha, 0, 255
        ).astype(np.uint8)
        return out_full, True


# --------------------------------------------------------------------------- #
# Classic diffusion inpainting
# --------------------------------------------------------------------------- #

class ClassicInpainter:
    def __init__(self, radius: int = 5, method: str = "telea"):
        self.radius = radius
        self.flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def fill(
        self, frame: np.ndarray, mask: np.ndarray, radius_pad: int = 2
    ) -> tuple[np.ndarray, bool]:
        hard = (mask > 40).astype(np.uint8) * 255
        if not hard.any():
            return frame, False
        hard = cv2.dilate(hard, np.ones((radius_pad * 2 + 1,) * 2, np.uint8))
        out = cv2.inpaint(frame, hard, self.radius, self.flag)
        return out, True


# --------------------------------------------------------------------------- #
# Temporal background reconstruction
# --------------------------------------------------------------------------- #

class TemporalFiller:
    """Recover hidden pixels from neighbouring frames.

    For each frame we look at a window of frames around it, shift each neighbour to
    line up with the centre frame (phase correlation on the visible pixels, so
    camera motion does not smear the result), and take the per-pixel median of the
    samples that were *not* covered by a watermark. A moving watermark means most
    pixels get several clean samples; a static watermark gives none, and the caller
    falls back to LaMa.
    """

    def __init__(
        self,
        window: int = 25,
        min_samples: int = 3,
        use_align: bool = True,
        align_max_shift: float = 40.0,
    ):
        self.window = window if window % 2 == 1 else window + 1
        self.min_samples = min_samples
        self.use_align = use_align
        self.align_max_shift = align_max_shift

    # ------------------------------------------------------------------ #

    def _shift(
        self, src: np.ndarray, dx: float, dy: float
    ) -> np.ndarray:
        h, w = src.shape[:2]
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    def _estimate_shift(
        self,
        center_patch: np.ndarray,
        nb_patch: np.ndarray,
        hole: np.ndarray,
    ) -> tuple[float, float, float]:
        """Translation that takes the neighbour patch onto the centre patch.

        Pixels covered by a watermark in either frame are replaced with the patch
        mean before correlating. Leaving them in would be a mistake: the hole edges
        are the strongest feature in the patch and they sit in *different* places in
        the two frames, so the cross-power spectrum locks onto the mask instead of
        the scene. A Hanning window keeps the patch border from doing the same.
        """
        keep = ~hole
        if int(keep.sum()) < 200:
            return 0.0, 0.0, 0.0

        a = center_patch.copy()
        b = nb_patch.copy()
        a[hole] = a[keep].mean()
        b[hole] = b[keep].mean()
        a -= a.mean()
        b -= b.mean()

        h, w = a.shape
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        try:
            (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        except cv2.error:
            return 0.0, 0.0, 0.0
        if not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0, 0.0, 0.0
        if abs(dx) > self.align_max_shift or abs(dy) > self.align_max_shift:
            return 0.0, 0.0, 0.0
        return float(dx), float(dy), float(resp)

    def _align_offset(
        self, center_gray: np.ndarray, nb_gray: np.ndarray, valid: np.ndarray
    ) -> tuple[float, float]:
        try:
            a = cv2.multiply(center_gray, valid, dtype=cv2.CV_32F)
            b = cv2.multiply(nb_gray, valid, dtype=cv2.CV_32F)
            (dx, dy), _resp = cv2.phaseCorrelate(a, b)
            if abs(dx) > self.align_max_shift or abs(dy) > self.align_max_shift:
                return 0.0, 0.0
            return float(dx), float(dy)
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------ #

    def fill(
        self,
        center: np.ndarray,
        center_mask: np.ndarray,
        window_frames: list[np.ndarray],
        window_masks: list[np.ndarray],
        center_index_in_window: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (filled frame, coverage mask in 0..255 of what we could recover).

        Work is scoped to the pixels that actually need filling, so a small logo in a
        4K frame costs almost nothing.
        """
        need = center_mask > 40
        coverage = np.zeros(center.shape[:2], np.uint8)
        if not need.any():
            return center.copy(), coverage

        h, w = center.shape[:2]
        ys, xs = np.nonzero(need)
        npix = len(xs)

        # Local region used for motion estimation and warping.
        pad = int(self.align_max_shift) + 6
        rx0, rx1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
        ry0, ry1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)

        center_gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY).astype(np.float32)
        center_valid = (center_mask <= 40).astype(np.float32)

        samples: list[np.ndarray] = []
        valids: list[np.ndarray] = []

        for j, (nb, nb_mask) in enumerate(zip(window_frames, window_masks)):
            if j == center_index_in_window or nb.shape[:2] != (h, w):
                continue

            if self.use_align:
                hole = (
                    (center_mask[ry0:ry1, rx0:rx1] > 40)
                    | (nb_mask[ry0:ry1, rx0:rx1] > 40)
                )
                nb_gray = cv2.cvtColor(
                    nb[ry0:ry1, rx0:rx1], cv2.COLOR_RGB2GRAY
                ).astype(np.float32)
                dx, dy, _resp = self._estimate_shift(
                    center_gray[ry0:ry1, rx0:rx1], nb_gray, hole
                )
            else:
                dx = dy = 0.0

            if abs(dx) > 0.3 or abs(dy) > 0.3:
                # phaseCorrelate reports the neighbour's offset *from* the centre
                # frame, so undoing it means warping by the negated shift.
                nb = self._shift(nb, -dx, -dy)
                nb_mask = self._shift(nb_mask, -dx, -dy)

            free = nb_mask <= 40
            if not free[ys, xs].any():
                continue
            samples.append(nb[ys, xs])            # (npix, 3)
            valids.append(free[ys, xs])           # (npix,)

        out = center.copy()
        if not samples:
            return out, coverage

        S = np.stack(samples, axis=0).astype(np.uint16)   # (n, npix, 3)
        V = np.stack(valids, axis=0)                      # (n, npix)
        count = V.sum(axis=0)                             # (npix,)
        good = count >= self.min_samples
        if not good.any():
            return out, coverage

        # Invalid samples sort to the top, so the median of the valid ones sits at
        # index count//2 without needing a per-pixel ragged array.
        filled = np.where(V[..., None] > 0, S, np.uint16(65535))
        filled = np.sort(filled, axis=0)          # invalid samples sink to the top
        midx = np.broadcast_to((count // 2)[None, :, None], filled.shape)
        median = np.take_along_axis(filled, midx, axis=0)[0]

        gy, gx = ys[good], xs[good]
        out[gy, gx] = np.clip(median[good], 0, 255).astype(np.uint8)
        coverage[gy, gx] = 255

        # Feather the repaired pixels into the surrounding frame.
        alpha = (center_mask.astype(np.float32)[..., None] / 255.0) * (
            coverage.astype(np.float32)[..., None] / 255.0
        )
        out = np.clip(
            out.astype(np.float32) * alpha + center.astype(np.float32) * (1 - alpha),
            0, 255,
        ).astype(np.uint8)
        return out, coverage
