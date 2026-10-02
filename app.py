#!/usr/bin/env python3
"""
Watermark Remover — local-first web app.

Everything happens on your own machine: uploads are written to ./data, the neural
model runs through ONNX Runtime on your CPU/GPU, and no frame ever leaves the box.
There are no outbound network calls in the request path.

Run:  python app.py            →  http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory, abort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.ff import encode_jpeg, extract_frame, ffmpeg_exe, probe   # noqa: E402
from engine.jobs import JobManager                                     # noqa: E402
from engine.render import RenderConfig, preview_frame, render_video     # noqa: E402
from engine.tracker import track_watermark                             # noqa: E402

DATA = HERE / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
MODELS = HERE / "models"
LAMA_PATH = MODELS / "lama.onnx"
for d in (UPLOADS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8 GB
jobs = JobManager(max_workers=1)

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts", ".mts"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mid_path(mid: str) -> Path:
    if not mid or "/" in mid or "\\" in mid or ".." in mid:
        abort(400)
    matches = list(UPLOADS.glob(f"{mid}.*"))
    if not matches:
        abort(404, "Unknown clip. Upload it again.")
    return matches[0]


def _safe_name(name: str) -> str:
    return "".join(c for c in Path(name).stem if c.isalnum() or c in "._-")[:60] or "clip"


def _model_status() -> dict:
    if LAMA_PATH.exists():
        return {"present": True, "size_mb": round(LAMA_PATH.stat().st_size / 1048576, 1)}
    return {"present": False, "size_mb": 0}


def _cleanup(max_age_s: float = 6 * 3600) -> None:
    now = time.time()
    for folder in (UPLOADS, OUTPUTS):
        for p in folder.iterdir():
            try:
                if p.is_file() and now - p.stat().st_mtime > max_age_s:
                    p.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Static / PWA
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/manifest.webmanifest")
def manifest():
    return send_from_directory(app.static_folder, "manifest.webmanifest")


@app.get("/sw.js")
def sw():
    resp = send_from_directory(app.static_folder, "sw.js")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "ffmpeg": ffmpeg_exe(),
        "model": _model_status(),
        "cpu_count": os.cpu_count(),
    })


# --------------------------------------------------------------------------- #
# Upload + media
# --------------------------------------------------------------------------- #

@app.post("/api/upload")
def upload():
    _cleanup()
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported container '{ext}'. Use mp4/mov/mkv/webm/avi."}), 400

    mid = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{mid}{ext}"
    f.save(dest)

    try:
        info = probe(str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return jsonify({"error": f"Could not read that video: {exc}"}), 400

    return jsonify({
        "id": mid,
        "name": Path(f.filename).name,
        "ext": ext,
        "info": info.to_dict(),
        "model": _model_status(),
    })


@app.get("/api/media/<mid>")
def media(mid: str):
    p = _mid_path(mid)
    return send_file(p, conditional=True, mimetype="video/mp4")


@app.get("/api/frame/<mid>")
def frame(mid: str):
    p = _mid_path(mid)
    t = float(request.args.get("t", 0))
    long_side = int(request.args.get("w", 1280))
    info = probe(str(p))
    ratio = info.height / max(1, info.width)
    w = min(long_side, info.width)
    h = max(2, int(round(w * ratio / 2)) * 2)
    img = extract_frame(str(p), t, w, h)
    q = int(request.args.get("q", 88))
    from flask import Response
    return Response(encode_jpeg(img, q), mimetype="image/jpeg")


@app.post("/api/preview")
def preview():
    body = request.get_json(force=True)
    mid = body.get("id")
    p = _mid_path(mid)
    info = probe(str(p))
    b = body.get("box") or [0, 0, 10, 10]
    box = tuple(int(round(v * info.width / float(body.get("ref_w", info.width)))) for v in b)
    t = float(body.get("t", 0))

    cfg = _cfg_from(body)
    model = str(LAMA_PATH) if LAMA_PATH.exists() else None
    if cfg.mode == "lama" and not model:
        return jsonify({"error": "The LaMa model is not installed."}), 400

    def work(job):
        before, after, used = preview_frame(str(p), t, box, cfg, model, info)
        return {
            "before": encode_jpeg(before, 92),
            "after": encode_jpeg(after, 92),
            "engine": used,
            "w": before.shape[1],
            "h": before.shape[0],
        }

    job = jobs.submit("preview", work)
    return jsonify({"job": job.id})


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

@app.post("/api/track")
def track():
    body = request.get_json(force=True)
    mid = body.get("id")
    p = _mid_path(mid)
    info = probe(str(p))

    ref_w = float(body.get("ref_w") or info.width)
    k = info.width / ref_w
    b = body.get("box") or []
    if len(b) != 4:
        return jsonify({"error": "Draw a box around the watermark first."}), 400
    box = tuple(int(round(v * k)) for v in b)

    t = float(body.get("t", 0))
    moving = bool(body.get("moving", True))

    def work(job):
        def prog(done, total):
            job.current = done
            job.progress = min(0.99, done / max(1, info.nb_frames))
            job.stage = "scanning frames"

        res = track_watermark(
            str(p), box, t,
            info.width, info.height, info.fps,
            moving=moving, progress=prog,
        )
        return res.to_dict()

    job = jobs.submit("track", work)
    return jsonify({"job": job.id})


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def _cfg_from(body: dict) -> RenderConfig:
    mode = body.get("mode", "auto")
    if mode not in ("auto", "lama", "temporal", "classic", "blur"):
        mode = "auto"
    return RenderConfig(
        mode=mode,
        margin=float(body.get("margin", 0.25)),
        feather=int(body.get("feather", 3)),
        crf=int(body.get("crf", 18)),
        preset=body.get("preset", "medium"),
        keep_audio=bool(body.get("keep_audio", True)),
        lama_pad=float(body.get("lama_pad", 1.2)),
        temporal_window=int(body.get("temporal_window", 13)),
        temporal_min_samples=int(body.get("temporal_min_samples", 3)),
        temporal_align=bool(body.get("temporal_align", True)),
        blur_strength=int(body.get("blur_strength", 25)),
        start_frame=int(body.get("start_frame", 0)),
        end_frame=body.get("end_frame"),
        scale=float(body.get("scale", 1.0)),
    )


@app.post("/api/render")
def render():
    body = request.get_json(force=True)
    mid = body.get("id")
    p = _mid_path(mid)
    info = probe(str(p))

    boxes = body.get("boxes") or []
    if not boxes:
        return jsonify({"error": "No watermark track supplied."}), 400

    ref_w = float(body.get("ref_w") or info.width)
    k = info.width / ref_w
    boxes = [[int(round(v * k)) for v in bx] for bx in boxes]
    confs = body.get("confidence") or None

    cfg = _cfg_from(body)
    model = str(LAMA_PATH) if LAMA_PATH.exists() else None
    if cfg.mode == "lama" and not model:
        return jsonify({"error": "The LaMa model is not installed. Pick another mode."}), 400

    job_id = uuid.uuid4().hex[:10]
    out = OUTPUTS / f"{_safe_name(body.get('name', 'clean'))}-{job_id}.mp4"

    def work(job):
        def prog(done, total, used):
            job.current = done
            job.total = total
            job.progress = done / max(1, total)
            job.stage = f"repairing ({used})"

        stats = render_video(
            str(p), str(out), boxes, confs, cfg, model,
            progress=prog, stop_flag=lambda: job.cancelled,
        )
        return {
            "url": f"/api/output/{out.name}",
            "filename": out.name,
            "size_mb": round(out.stat().st_size / 1048576, 2),
            "frames": stats.frames,
            "lama_frames": stats.lama_used,
            "temporal_frames": stats.temporal_used,
            "classic_frames": stats.classic_used,
            "seconds": round(stats.seconds, 1),
            "notes": stats.notes,
        }

    job = jobs.submit("render", work)
    return jsonify({"job": job.id})


@app.get("/api/output/<path:name>")
def output(name: str):
    p = OUTPUTS / Path(name).name
    if not p.exists():
        abort(404)
    return send_file(p, conditional=True, mimetype="video/mp4", as_attachment=False)


@app.post("/api/cancel/<job_id>")
def cancel(job_id: str):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    job.cancel()
    return jsonify({"ok": True})


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    data = job.public()
    # Images are binary — hand them back base64 so the JSON stays valid.
    if job.state == "done" and isinstance(job.result, dict) and "before" in job.result:
        import base64
        r = dict(job.result)
        r["before"] = base64.b64encode(r["before"]).decode()
        r["after"] = base64.b64encode(r["after"]).decode()
        data["result"] = r
    return jsonify(data)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Local watermark remover")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"ffmpeg : {ffmpeg_exe()}")
    m = _model_status()
    print(f"LaMa   : {'ready (' + str(m['size_mb']) + ' MB)' if m['present'] else 'NOT INSTALLED — run scripts/fetch_model.py'}")
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
