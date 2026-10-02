"""
ff.py — FFmpeg plumbing: locate the binary, probe files, read and write raw frames.

Everything is local. No network calls are made anywhere in this module.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Iterator, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Binary location
# --------------------------------------------------------------------------- #

_ffmpeg_path: Optional[str] = None


def ffmpeg_exe() -> str:
    """Return a usable ffmpeg executable, preferring a system one, then the
    wheel-bundled static build shipped by `imageio-ffmpeg`."""
    global _ffmpeg_path
    if _ffmpeg_path:
        return _ffmpeg_path

    env = os.environ.get("WMREMOVER_FFMPEG")
    if env and os.path.exists(env):
        _ffmpeg_path = env
        return _ffmpeg_path

    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        _ffmpeg_path = sys_ff
        return _ffmpeg_path

    try:
        import imageio_ffmpeg  # type: ignore

        _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        return _ffmpeg_path
    except Exception as exc:  # pragma: no cover - surfaced as a clean error
        raise RuntimeError(
            "No ffmpeg found. Install it (`brew install ffmpeg`, "
            "`apt install ffmpeg`) or `pip install imageio-ffmpeg`."
        ) from exc


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_exe(), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #

_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RE_VIDEO = re.compile(
    r"Stream #\d+:\d+.*?:\s*Video:\s*([\w]+).*?,\s*(\d{2,5})x(\d{2,5})"
)
_RE_FPS = re.compile(r"(\d+(?:\.\d+)?)\s+fps")
_RE_TBR = re.compile(r"(\d+(?:\.\d+)?)\s+tbr")
_RE_NBFRAMES = re.compile(r"(\d+)\s+frames")
_RE_AUDIO = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:\s*([\w]+)")
_RE_ROTATE = re.compile(r"rotate\s*:\s*(-?\d+)")
_RE_DISPLAYMATRIX = re.compile(r"displaymatrix:\s*rotation of\s*(-?\d+(?:\.\d+)?)")


@dataclass
class MediaInfo:
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    nb_frames: int = 0
    vcodec: str = ""
    acodec: str = ""
    has_audio: bool = False
    rotation: int = 0
    size_bytes: int = 0

    @property
    def display_width(self) -> int:
        """Rotation is already folded into width/height by probe()."""
        return self.width

    @property
    def display_height(self) -> int:
        return self.height

    def to_dict(self) -> dict:
        d = asdict(self)
        d["display_width"] = self.display_width
        d["display_height"] = self.display_height
        return d


def probe(path: str) -> MediaInfo:
    """Read media metadata by parsing `ffmpeg -i` output.

    The bundled static build has no ffprobe, so parsing is the portable path.
    """
    proc = _run(["-hide_banner", "-i", path])
    text = proc.stderr.decode("utf-8", "replace")

    info = MediaInfo(path=path, size_bytes=os.path.getsize(path))

    m = _RE_DURATION.search(text)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info.duration = h * 3600 + mi * 60 + s

    m = _RE_VIDEO.search(text)
    if m:
        info.vcodec = m.group(1)
        info.width, info.height = int(m.group(2)), int(m.group(3))
    else:
        raise ValueError("No video stream found in this file.")

    m = _RE_FPS.search(text) or _RE_TBR.search(text)
    if m:
        info.fps = float(m.group(1))
    if not info.fps or info.fps <= 0:
        info.fps = 25.0

    m = _RE_NBFRAMES.search(text)
    if m:
        info.nb_frames = int(m.group(1))
    if info.nb_frames <= 0 and info.duration > 0:
        info.nb_frames = max(1, int(round(info.duration * info.fps)))

    m = _RE_AUDIO.search(text)
    if m:
        info.acodec = m.group(1)
        info.has_audio = True

    m = _RE_DISPLAYMATRIX.search(text) or _RE_ROTATE.search(text)
    if m:
        info.rotation = int(round(float(m.group(1))))

    # ffmpeg auto-rotates on decode, so report dimensions the way they will
    # actually come out of FrameReader — otherwise phone-shot (rotated) clips
    # would be processed sideways.
    if abs(info.rotation) in (90, 270):
        info.width, info.height = info.height, info.width

    return info


# --------------------------------------------------------------------------- #
# Single-frame extraction
# --------------------------------------------------------------------------- #

def extract_frame(path: str, t: float, width: int, height: int) -> np.ndarray:
    """Decode one frame at time `t`, scaled to exactly (width, height) as RGB."""
    cmd = [
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, t):.4f}",
        "-i", path,
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vf", f"scale={width}:{height}",
        "-",
    ]
    proc = _run(cmd, timeout=180)
    buf = proc.stdout
    need = width * height * 3
    if len(buf) < need:
        raise RuntimeError(
            f"Could not decode a frame at t={t}s: {proc.stderr.decode('utf-8','replace')[:300]}"
        )
    return np.frombuffer(buf[:need], dtype=np.uint8).reshape(height, width, 3)


def encode_jpeg(frame: np.ndarray, quality: int = 88) -> bytes:
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Sequential frame readers
# --------------------------------------------------------------------------- #

class FrameReader:
    """Stream decoded RGB frames from a video, optionally starting at `start_t`."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        start_t: float = 0.0,
        max_frames: Optional[int] = None,
    ):
        self.path = path
        self.w = width
        self.h = height
        self.frame_bytes = width * height * 3
        self.max_frames = max_frames

        cmd = ["-hide_banner", "-loglevel", "error", "-nostdin"]
        if start_t > 0:
            cmd += ["-ss", f"{start_t:.4f}"]
        cmd += [
            "-i", path,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-vf", f"scale={width}:{height}:flags=area",
            "-",
        ]
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def frames(self) -> Iterator[np.ndarray]:
        count = 0
        stream = self.proc.stdout
        while True:
            if self.max_frames is not None and count >= self.max_frames:
                break
            buf = _read_exact(stream, self.frame_bytes)
            if buf is None:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)
            count += 1


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly n bytes, or None at clean EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining == n:
        return None
    if remaining > 0:  # truncated tail frame — drop it
        return None
    return b"".join(chunks)


class FrameWriter:
    """Encode RGB frames to H.264, muxing the audio track from a source file."""

    def __init__(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: Optional[str] = None,
        start_t: float = 0.0,
        crf: int = 18,
        preset: str = "medium",
    ):
        # yuv420p needs even dimensions
        self.w = width + (width % 2)
        self.h = height + (height % 2)

        cmd = [
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}", "-r", f"{fps:.6f}",
            "-i", "-",
        ]
        if audio_source:
            cmd += ["-i", audio_source]
            if start_t > 0:
                # realign audio to the first frame we are writing
                cmd += ["-map_metadata", "-1"]
            cmd += ["-map", "0:v:0", "-map", "1:a:0?"]
        else:
            cmd += ["-map", "0:v:0"]

        cmd += [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-movflags", "+faststart",
        ]
        if audio_source:
            cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += [out_path]

        self.proc = subprocess.Popen(
            [ffmpeg_exe(), *cmd],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._err: list[bytes] = []

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            import cv2

            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        err = self.proc.stderr.read() if self.proc.stderr else b""
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed ({self.proc.returncode}): "
                f"{err.decode('utf-8', 'replace')[-600:]}"
            )

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
