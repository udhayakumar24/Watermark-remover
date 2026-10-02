"""
render.py — drive the removal pipeline end to end.

Decodes the source, applies the tracked mask, fills it with the chosen engine, and
re-encodes to H.264 while muxing the original audio back in. Frames are streamed, so
memory stays flat regardless of clip length.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .ff import FrameReader, FrameWriter, extract_frame, probe
from .inpaint import ClassicInpainter, LamaInpainter, TemporalFiller, make_mask
from .textmask import text_mask, feather


@dataclass
class RenderConfig:
    mode: str = "auto"            # auto | lama | temporal | classic | blur
    margin: float = 0.25
    feather: int = 3
    crf: int = 18
    preset: str = "medium"
    keep_audio: bool = True
    lama_pad: float = 1.2
    temporal_window: int = 25
    temporal_min_samples: int = 3
    temporal_align: bool = True
    blur_strength: int = 25
    start_frame: int = 0
    end_frame: Optional[int] = None   # exclusive
    scale: float = 1.0                # working resolution multiplier
    low_conf_override: bool = True    # use lama when the tracker lost its lock
    tight_text: bool = True           # erase glyph strokes only, never the box


@dataclass
class RenderStats:
    frames: int = 0
    lama_used: int = 0
    temporal_used: int = 0
    classic_used: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #

def _working_size(info, scale: float) -> tuple[int, int]:
    w = max(16, int(round(info.width * scale / 2)) * 2)
    h = max(16, int(round(info.height * scale / 2)) * 2)
    return w, h


def _boxes_for(track_boxes: list, n: int, as_int: bool = True) -> list:
    """Stretch or clamp a per-frame list so it lines up with the decoded frames."""
    if not track_boxes:
        raise ValueError("No watermark track supplied.")
    if len(track_boxes) == n:
        out = list(track_boxes)
    elif len(track_boxes) == 1:
        out = list(track_boxes) * n
    else:
        src = np.asarray(track_boxes, dtype=np.float32)
        idx = np.clip(np.round(np.linspace(0, len(src) - 1, n)).astype(int), 0, len(src) - 1)
        out = [tuple(src[i]) for i in idx]
    if as_int:
        out = [tuple(int(round(v)) for v in item) for item in out]
    return out


def _fill_engine(cfg: RenderConfig, model_path: Optional[str]):
    lama = None
    if cfg.mode in ("auto", "lama"):
        if not model_path or not os.path.exists(model_path):
            if cfg.mode == "lama":
                raise FileNotFoundError(
                    "The LaMa model file is missing. Download it with "
                    "`python scripts/fetch_model.py` or switch to another mode."
                )
            lama = None
        else:
            lama = LamaInpainter.get(model_path)
    return lama


def _mask_for(frame: np.ndarray, box: tuple, cfg: RenderConfig) -> np.ndarray:
    """The region to repair. Prefer a tight glyph-only mask so the subject behind the
    watermark (a face, a mouth, a product) is never repainted; fall back to the full
    safety-margin box only when clean text cannot be isolated."""
    if cfg.tight_text:
        tm, ok = text_mask(frame, box)
        if ok:
            return feather(tm, 1)
    return make_mask(frame.shape[:2], box, cfg.margin, cfg.feather)


# --------------------------------------------------------------------------- #

def render_video(
    src_path: str,
    out_path: str,
    track_boxes: list,
    track_confidence: Optional[list] = None,
    cfg: Optional[RenderConfig] = None,
    model_path: Optional[str] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> RenderStats:
    cfg = cfg or RenderConfig()
    info = probe(src_path)

    w, h = _working_size(info, cfg.scale)
    stats = RenderStats()
    t_start = time.time()

    start_frame = max(0, int(cfg.start_frame))
    total = info.nb_frames
    end_frame = total if cfg.end_frame is None else min(total, int(cfg.end_frame))
    n_target = max(1, end_frame - start_frame)

    lama = _fill_engine(cfg, model_path)
    temporal = (
        TemporalFiller(
            window=cfg.temporal_window,
            min_samples=cfg.temporal_min_samples,
            use_align=cfg.temporal_align,
        )
        if cfg.mode in ("auto", "temporal")
        else None
    )
    classic = ClassicInpainter()

    # Memory guard for the temporal buffer.
    win = 1
    if temporal is not None:
        per = w * h * 3
        budget = 600 * 1024 * 1024
        win = int(max(3, min(cfg.temporal_window, budget // max(1, per * 2))))
        win = win if win % 2 == 1 else win - 1
        temporal.window = max(3, win)

    start_t = start_frame / max(1e-6, info.fps)

    audio_src = src_path if (cfg.keep_audio and info.has_audio) else None
    writer = FrameWriter(
        out_path, w, h, info.fps,
        audio_source=audio_src, start_t=start_t,
        crf=cfg.crf, preset=cfg.preset,
    )

    # Read a few frames past the requested end purely as temporal context, so the
    # last output frames are not repaired from a half-empty window. They are never
    # written out.
    half_ctx = temporal.window // 2 if temporal is not None else 0
    read_limit = n_target + half_ctx

    try:
        with FrameReader(
            src_path, w, h, start_t=start_t, max_frames=read_limit
        ) as reader:
            # Boxes / confidences are aligned to the DECODED frame count, not the
            # read-ahead limit. Indexing them by the look-ahead position (an earlier
            # bug) put each mask where the watermark had been several frames before,
            # so the filler sampled the wrong pixels.
            boxes = _boxes_for(track_boxes, n_target)
            confs = (
                _boxes_for([(c, 0, 0, 0) for c in track_confidence], n_target,
                           as_int=False)
                if track_confidence
                else None
            )

            def box_at(idx: int) -> tuple:
                return boxes[min(idx, n_target - 1)]

            def conf_at(idx: int) -> float:
                return confs[min(idx, n_target - 1)][0] if confs else 1.0

            # `buf` is a sliding window over input frames [front .. j]. Frames are
            # emitted strictly in input order: input frame k goes out as output frame
            # k, once its `half` frames of look-ahead have been read. Emitting in any
            # other order scrambles the video (an earlier version wrote the buffered
            # centre first, shifting the whole output by half a window).
            half = temporal.window // 2 if temporal is not None else 0
            buf: deque = deque()
            front = 0
            next_emit = 0

            def emit(idx: int) -> None:
                lo = max(front, idx - half)
                hi = min(front + len(buf) - 1, idx + half)
                window = [buf[i - front] for i in range(lo, hi + 1)]
                ci = idx - lo
                frame, mask = window[ci]
                cval = conf_at(idx)

                out, used = _repair_one(
                    frame, mask, cval, cfg, lama, temporal, classic, window, ci
                )
                if used == "lama":
                    stats.lama_used += 1
                elif used == "temporal":
                    stats.temporal_used += 1
                elif used in ("classic", "blur"):
                    stats.classic_used += 1

                writer.write(out)
                if progress:
                    progress(idx + 1, n_target, used)

            for j, frame in enumerate(reader.frames()):
                if j >= read_limit:
                    break
                if stop_flag and stop_flag():
                    stats.notes.append("Stopped by user.")
                    break
                if temporal is None:
                    mask = _mask_for(frame, box_at(j), cfg)
                    cval = conf_at(j)
                    out, used = _repair_one(frame, mask, cval, cfg, lama, None, classic, [], 0)
                    if used == "lama":
                        stats.lama_used += 1
                    elif used in ("classic", "blur"):
                        stats.classic_used += 1
                    writer.write(out)
                    if progress:
                        progress(j + 1, n_target, used)
                    next_emit = j + 1
                    continue

                # Drop frames too old to belong to any window we will still emit.
                while buf and front < next_emit - half:
                    buf.popleft()
                    front += 1
                buf.append((frame, _mask_for(frame, box_at(j), cfg)))
                while next_emit < n_target and j >= next_emit + half:
                    emit(next_emit)
                    next_emit += 1

            # Drain the tail: look-ahead ran out, use the truncated window.
            while next_emit < n_target:
                emit(next_emit)
                next_emit += 1
            stats.frames = next_emit
    finally:
        try:
            writer.close()
        except Exception as exc:
            stats.notes.append(f"Encode warning: {exc}")

    stats.seconds = time.time() - t_start
    return stats


def _repair_one(
    frame: np.ndarray,
    mask: np.ndarray,
    confidence: float,
    cfg: RenderConfig,
    lama: Optional[LamaInpainter],
    temporal: Optional[TemporalFiller],
    classic: ClassicInpainter,
    buf: deque,
    center_i: int,
) -> tuple[np.ndarray, str]:
    if cfg.mode == "blur":
        k = max(3, int(cfg.blur_strength) | 1)
        blurred = cv2.GaussianBlur(frame, (k * 2 + 1, k * 2 + 1), 0)
        pix = cv2.resize(
            cv2.resize(frame, (max(1, frame.shape[1] // 12), max(1, frame.shape[0] // 12)),
                       interpolation=cv2.INTER_LINEAR),
            (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        blended = cv2.addWeighted(blurred, 0.5, pix, 0.5, 0)
        a = (mask.astype(np.float32)[..., None] / 255.0)
        return np.clip(frame * (1 - a) + blended * a, 0, 255).astype(np.uint8), "blur"

    lost = cfg.low_conf_override and confidence < 0.28

    if cfg.mode == "lama" or (cfg.mode == "auto" and lost):
        if lama is not None:
            out, _ok = lama.fill(frame, mask, pad_ratio=cfg.lama_pad)
            return out, "lama"

    if temporal is not None and cfg.mode in ("auto", "temporal") and len(buf) > 1:
        frames = [f for f, _ in buf]
        masks = [m for _, m in buf]
        out, cov = temporal.fill(frame, mask, frames, masks, center_i)

        # Only the *core* of the mask counts as an unfilled hole. The mask edges are
        # deliberately feathered, so a thin rim is always partly uncovered; treating
        # that rim as a hole made the neural fallback fire on every single frame
        # (~0.15 fps) even when temporal had recovered the whole watermark.
        hole = np.where((mask > 160) & (cov <= 40), 255, 0).astype(np.uint8)
        if not hole.any():
            return out, "temporal"

        if lama is not None and cfg.mode == "auto":
            out, _ok = lama.fill(out, hole, pad_ratio=cfg.lama_pad)
            return out, "lama"
        out, _ok = classic.fill(out, hole, radius_pad=1)
        return out, "classic"

    if lama is not None and cfg.mode in ("auto", "lama"):
        out, _ok = lama.fill(frame, mask, pad_ratio=cfg.lama_pad)
        return out, "lama"

    out, _ok = classic.fill(frame, mask)
    return out, "classic"


# --------------------------------------------------------------------------- #
# Single-frame preview (before / after)
# --------------------------------------------------------------------------- #

def preview_frame(
    src_path: str,
    t: float,
    box: tuple[int, int, int, int],
    cfg: RenderConfig,
    model_path: Optional[str] = None,
    info=None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Process ONE frame and return (before, after, engine_used)."""
    info = info or probe(src_path)
    w, h = _working_size(info, cfg.scale)

    lama = _fill_engine(cfg, model_path)
    classic = ClassicInpainter()
    temporal = None
    if cfg.mode in ("auto", "temporal"):
        temporal = TemporalFiller(
            window=cfg.temporal_window,
            min_samples=cfg.temporal_min_samples,
            use_align=cfg.temporal_align,
        )

    before = extract_frame(src_path, t, w, h)
    mask = _mask_for(before, box, cfg)

    if cfg.mode in ("auto", "temporal") and temporal is not None:
        half = temporal.window // 2
        dt = 1.0 / max(1e-6, info.fps)
        frames, masks, ci = [], [], half
        for j in range(temporal.window):
            tt = max(0.0, t + (j - half) * dt)
            try:
                f = extract_frame(src_path, tt, w, h)
            except Exception:
                f = before
            frames.append(f)
            masks.append(mask)
        after, cov = temporal.fill(before, mask, frames, masks, half)
        used = "temporal"
        hole = np.where((mask > 160) & (cov <= 40), 255, 0).astype(np.uint8)
        if hole.any():
            if lama is not None and cfg.mode == "auto":
                after, _ = lama.fill(after, hole, pad_ratio=cfg.lama_pad)
                used = "auto (temporal + neural)"
            else:
                after, _ = classic.fill(after, hole, radius_pad=1)
                used = ("auto (temporal + diffusion)" if cfg.mode == "auto"
                        else "temporal + diffusion")
        return before, after, used

    if cfg.mode == "lama" and lama is not None:
        after, _ = lama.fill(before, mask, pad_ratio=cfg.lama_pad)
        return before, after, "lama"

    if cfg.mode == "blur":
        after, _ = _repair_one(before, mask, 1.0, cfg, lama, None, classic, deque(), 0)
        return before, after, "blur"

    after, _ = classic.fill(before, mask)
    return before, after, "classic"
