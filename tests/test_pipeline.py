#!/usr/bin/env python3
"""
test_pipeline.py — end-to-end checks for the watermark remover.

The test clip has a *moving* watermark over an *animated* background, and the test
also renders the same background with no watermark at all. That gives a true
reference image, so "did it work?" is measured as error against the real background
rather than eyeballed.

Run:  python tests/test_pipeline.py      (or `pytest tests/ -q`)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import tests.make_clip as mk  # noqa: E402
from engine.ff import FrameReader, encode_jpeg, extract_frame, probe  # noqa: E402
from engine.inpaint import ClassicInpainter, LamaInpainter, TemporalFiller, make_mask  # noqa: E402
from engine.render import RenderConfig, preview_frame, render_video  # noqa: E402
from engine.tracker import track_watermark  # noqa: E402

LAMA = ROOT / "models" / "lama.onnx"
FIXTURES = HERE / "fixtures"

STATE: dict = {}


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

def setup_module(_mod=None) -> None:
    if STATE.get("ready"):
        return
    video = mk.main(FIXTURES)
    meta = json.loads((FIXTURES / "clip_gt.json").read_text())
    info = probe(str(video))

    pan = mk.main_pan(FIXTURES)
    pan_meta = json.loads((FIXTURES / "clip_pan_gt.json").read_text())

    STATE.update(video=str(video), meta=meta, info=info,
                 pan=str(pan), pan_meta=pan_meta, pan_info=probe(str(pan)))
    STATE["ready"] = True


def _gt_frame(t: float) -> np.ndarray:
    """The true chaotic background at time t — no watermark."""
    return cv2.cvtColor(mk.background(t), cv2.COLOR_BGR2RGB)


def _pan_gt_frame(i: int) -> np.ndarray:
    """The true panning-scene background at frame i — no watermark."""
    return cv2.cvtColor(mk.pan_background(i), cv2.COLOR_BGR2RGB)


def _region_error(img: np.ndarray, ref: np.ndarray, box, margin: float = 0.0) -> float:
    """Mean absolute error inside the box, expanded by margin."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1 = min(img.shape[1], x + w + mx)
    y1 = min(img.shape[0], y + h + my)
    a = img[y0:y1, x0:x1].astype(np.float32)
    b = ref[y0:y1, x0:x1].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


# --------------------------------------------------------------------------- #
# 1. Plumbing
# --------------------------------------------------------------------------- #

def test_probe_reports_video_and_audio():
    setup_module()
    info = STATE["info"]
    assert (info.width, info.height) == (mk.W, mk.H), f"got {info.width}x{info.height}"
    assert abs(info.fps - mk.FPS) < 0.5, f"fps {info.fps}"
    assert abs(info.duration - mk.SECONDS) < 0.4, f"duration {info.duration}"
    assert info.nb_frames >= mk.FPS * mk.SECONDS - 3, f"nb_frames {info.nb_frames}"
    assert info.has_audio, "audio track not detected"
    assert info.acodec == "aac", f"acodec {info.acodec}"
    print(f"  probe: {info.width}x{info.height} {info.fps}fps {info.duration:.2f}s "
          f"{info.nb_frames}f audio={info.acodec}")


def test_frame_extraction_and_jpeg():
    setup_module()
    f = extract_frame(STATE["video"], 0.5, mk.W, mk.H)
    assert f.shape == (mk.H, mk.W, 3), f.shape
    assert f.dtype == np.uint8
    jpg = encode_jpeg(f, 90)
    assert jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9"
    print(f"  extract_frame -> {f.shape}, jpeg {len(jpg)} bytes")


def test_frame_reader_streams_all_frames():
    setup_module()
    with FrameReader(STATE["video"], mk.W, mk.H) as r:
        n = sum(1 for fr in r.frames() if fr.shape == (mk.H, mk.W, 3))
    assert abs(n - mk.FPS * mk.SECONDS) <= 3, f"decoded {n} frames"
    print(f"  FrameReader decoded {n} frames")


# --------------------------------------------------------------------------- #
# 2. Tracker — the watermark genuinely moves, so this must follow it
# --------------------------------------------------------------------------- #

def test_tracker_follows_the_moving_watermark():
    setup_module()
    meta, info = STATE["meta"], STATE["info"]
    anchor_i = 0
    box0 = meta["gt"][anchor_i]["box"]

    res = track_watermark(
        STATE["video"], tuple(box0), anchor_i / info.fps,
        info.width, info.height, info.fps, moving=True,
    )

    assert res.frames >= mk.FPS * mk.SECONDS - 3, f"only {res.frames} frames tracked"

    errs = []
    for i, g in enumerate(meta["gt"]):
        if i >= len(res.boxes):
            break
        bx, by, bw, bh = res.boxes[i]
        tracked = (bx + bw / 2, by + bh / 2)
        errs.append(float(np.hypot(tracked[0] - g["center"][0], tracked[1] - g["center"][1])))
    errs = np.asarray(errs)

    # The watermark travels ~150px across the clip; a stuck box would be off by
    # that much. We require the median error to be a small fraction of the box.
    med, p95 = float(np.median(errs)), float(np.percentile(errs, 95))
    travel = res.max_displacement
    assert travel > 40, f"tracker thinks the logo is static (travel={travel:.1f}px)"
    assert med < 12, f"median tracking error {med:.1f}px too large"
    assert p95 < 40, f"95th-percentile error {p95:.1f}px too large"
    print(f"  tracker: travel {travel:.0f}px, median err {med:.1f}px, "
          f"p95 {p95:.1f}px, mean match {res.mean_confidence:.2f}, lost {res.lost_frames}")


def test_tracker_fixed_mode_pins_the_box():
    setup_module()
    meta, info = STATE["meta"], STATE["info"]
    box0 = meta["gt"][0]["box"]
    res = track_watermark(
        STATE["video"], tuple(box0), 0.0,
        info.width, info.height, info.fps, moving=False,
    )
    assert all(b == res.boxes[0] for b in res.boxes), "fixed mode moved the box"
    print(f"  fixed mode: {len(res.boxes)} identical boxes")


# --------------------------------------------------------------------------- #
# 3. Inpainting — measured against the true background
# --------------------------------------------------------------------------- #

def _case_frame(t: float = 0.5):
    setup_module()
    meta = STATE["meta"]
    i = int(round(t * mk.FPS))
    box = tuple(meta["gt"][i]["box"])
    src = extract_frame(STATE["video"], t, mk.W, mk.H)
    ref = _gt_frame(t)
    return src, ref, box, i


def test_make_mask_geometry():
    m = make_mask((100, 200), (50, 40, 30, 20), margin=0.0, feather=0)
    assert m.shape == (100, 200)
    assert m[45, 60] == 255 and m[5, 5] == 0
    m2 = make_mask((100, 200), (50, 40, 30, 20), margin=0.5, feather=0)
    assert m2.sum() > m.sum(), "margin did not enlarge the mask"
    print(f"  mask: base {int((m>0).sum())}px -> margined {int((m2>0).sum())}px")


def test_classic_inpaint_on_an_opaque_mark():
    """Diffusion inpainting on the case it suits: an opaque mark, textured ground.

    Note the flip side, which the numbers below make concrete: on a *semi-transparent*
    watermark the same operation makes the pixel error worse (measured 14.8 -> 30.0
    on the test clip), because the covered pixels still hold part of the real
    background and diffusion throws that away. Every replace-the-region method —
    diffusion and LaMa alike — has this property. It is why the default mode recovers
    pixels from other frames first and only synthesises what is genuinely missing.
    """
    setup_module()
    box = tuple(STATE["pan_meta"]["gt"][0]["box"])
    clean = cv2.cvtColor(mk.pan_background(0), cv2.COLOR_BGR2RGB).copy()
    broken = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
    mk.stamp_watermark(broken, box[0] + box[2] / 2, box[1] + box[3] / 2, alpha=1.0)
    broken = cv2.cvtColor(broken, cv2.COLOR_BGR2RGB)

    core = np.zeros(clean.shape[:2], bool)
    core[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = True
    e = lambda img: float(np.mean(np.abs(img[core].astype(np.float32) - clean[core].astype(np.float32))))

    mask = make_mask(clean.shape[:2], box, margin=0.2, feather=2)
    before = e(broken)
    out, did = ClassicInpainter().fill(broken, mask)
    assert did
    after = e(out)
    assert after < before * 0.8, f"classic on opaque mark: {before:.1f} -> {after:.1f}"
    print(f"  classic : opaque mark core err {before:.1f} -> {after:.1f}")


def test_semitransparent_watermark_prefers_recovery_over_synthesis():
    """Regression guard for the mode ordering, stated as a measured fact."""
    setup_module()
    meta = STATE["pan_meta"]
    i = 12
    box = tuple(meta["gt"][i]["box"])
    with FrameReader(STATE["pan"], mk.W, mk.H) as r:
        allf = list(r.frames())
    src, ref = allf[i], _pan_gt_frame(i)
    mask = make_mask(src.shape[:2], box, margin=0.2, feather=0)
    core = np.zeros(src.shape[:2], bool)
    core[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = True
    e = lambda img: float(np.mean(np.abs(img[core].astype(np.float32) - ref[core].astype(np.float32))))

    frames = [allf[min(max(0, i + k), len(allf) - 1)] for k in range(-12, 13)]
    masks = [make_mask(src.shape[:2], tuple(meta["gt"][min(max(0, i + k), len(meta["gt"]) - 1)]["box"]),
                       margin=0.2, feather=0) for k in range(-12, 13)]
    temporal_out, _ = TemporalFiller(window=25, min_samples=3).fill(src, mask, frames, masks, 12)
    classic_out, _ = ClassicInpainter().fill(src, mask)

    t_err, c_err = e(temporal_out), e(classic_out)
    assert t_err < c_err, (
        f"recovery ({t_err:.1f}) should beat synthesis ({c_err:.1f}) on a semi-transparent mark")
    print(f"  modes   : semi-transparent mark -> recovery {t_err:.1f} vs synthesis {c_err:.1f} "
          f"(recovery wins, so it runs first)")


def test_temporal_recovers_the_true_background():
    """Realistic case: a steady camera pan over a detailed scene.

    One global translation explains the motion, so neighbouring frames really do
    contain the hidden pixels and the filler should get them almost exactly right.
    """
    setup_module()
    meta = STATE["pan_meta"]
    i = 12
    box = tuple(meta["gt"][i]["box"])
    with FrameReader(STATE["pan"], mk.W, mk.H) as r:
        allf = list(r.frames())
    src, ref = allf[i], _pan_gt_frame(i)
    mask = make_mask(src.shape[:2], box, margin=0.2, feather=0)

    win, half = 25, 12
    frames = [allf[min(max(0, i + k), len(allf) - 1)] for k in range(-half, half + 1)]
    masks = [make_mask(src.shape[:2], tuple(meta["gt"][min(max(0, i + k), len(meta["gt"]) - 1)]["box"]),
                       margin=0.2, feather=0) for k in range(-half, half + 1)]

    before = _region_error(src, ref, box, 0.2)
    out, cov = TemporalFiller(window=win, min_samples=3, use_align=True).fill(
        src, mask, frames, masks, half)
    after = _region_error(out, ref, box, 0.2)
    coverage = float((cov > 40).sum() / max(1, (mask > 40).sum()))

    assert coverage > 0.85, f"only {coverage:.0%} of the hole was covered"
    assert after < 4.0, f"temporal left {after:.1f} error (source {before:.1f})"
    print(f"  temporal: err {before:.1f} -> {after:.1f} vs true background, "
          f"coverage {coverage:.0%} (pan clip)")


def test_temporal_on_incoherent_motion_hits_the_drift_floor():
    """Chaotic case, measured against what is physically achievable.

    Here the background itself changes by more than the watermark does, so no
    temporal method can beat the source on a per-pixel error metric. The honest
    assertion is that the filler reaches the drift floor of the window it was given
    rather than adding artefacts on top of it.
    """
    setup_module()
    meta = STATE["meta"]
    i = 12
    box = tuple(meta["gt"][i]["box"])
    with FrameReader(STATE["video"], mk.W, mk.H) as r:
        allf = list(r.frames())
    src, ref = allf[i], _gt_frame(i / mk.FPS)
    mask = make_mask(src.shape[:2], box, margin=0.2, feather=0)

    win, half = 13, 6
    frames = [allf[min(max(0, i + k), len(allf) - 1)] for k in range(-half, half + 1)]
    masks = [make_mask(src.shape[:2], tuple(meta["gt"][min(max(0, i + k), len(meta["gt"]) - 1)]["box"]),
                       margin=0.2, feather=0) for k in range(-half, half + 1)]

    # The best any temporal filler could do is substitute a neighbouring frame's
    # true background, so that is the floor this method is scored against.
    floor = min(_region_error(_gt_frame((i + k) / mk.FPS), ref, box, 0.2)
                for k in range(-half, half + 1) if k)
    source_err = _region_error(src, ref, box, 0.2)
    out, cov = TemporalFiller(window=win, min_samples=2, use_align=True).fill(
        src, mask, frames, masks, half)
    after = _region_error(out, ref, box, 0.2)

    assert after < floor * 1.6 + 3, (
        f"temporal added artefacts: {after:.1f} vs drift floor {floor:.1f}")
    print(f"  temporal: chaotic bg -> err {after:.1f} vs source {source_err:.1f}, "
          f"drift floor {floor:.1f} (background moves more than the watermark here)")


def test_lama_pipeline_is_geometrically_correct():
    """Controlled case: a known gradient with a hole punched in it.

    This isolates the crop / 512x512 round-trip / paste-back geometry from the
    question of whether the network guesses well. If the offsets or the normalisation
    were wrong this would blow up even though the model is fine.
    """
    if not LAMA.exists():
        print("  lama    : SKIPPED (model not downloaded)")
        return
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    scene = np.stack([x, y, (x + y) / 2], -1).astype(np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    mask[100:140, 80:180] = 255
    broken = scene.copy()
    broken[100:140, 80:180] = 255

    out, did = LamaInpainter.get(str(LAMA)).fill(broken, mask, pad_ratio=0.6)
    assert did
    hole = mask > 0
    before = float(np.mean(np.abs(broken[hole].astype(np.float32) - scene[hole].astype(np.float32))))
    after = float(np.mean(np.abs(out[hole].astype(np.float32) - scene[hole].astype(np.float32))))
    assert np.array_equal(out[~hole], broken[~hole]), "LaMa touched pixels outside the mask"
    assert after < 8.0, f"geometry/normalisation looks wrong: err {before:.1f} -> {after:.1f}"
    print(f"  lama    : controlled gradient err {before:.1f} -> {after:.1f}, "
          f"outside-mask byte-identical ✓")


def test_lama_removes_an_opaque_watermark():
    """LaMa's proper job: an OPAQUE mark on a detailed scene.

    Worth stating plainly, because it drives the mode choice: when a watermark is
    semi-transparent the original pixels still hold part of the truth, so replacing
    them wholesale with a plausible invention can score worse than doing less. That
    is why `auto` prefers temporal recovery and only reaches for the network where it
    is genuinely the better tool.
    """
    if not LAMA.exists():
        print("  lama    : SKIPPED (model not downloaded)")
        return
    setup_module()
    box = tuple(STATE["pan_meta"]["gt"][0]["box"])
    clean = cv2.cvtColor(mk.pan_background(0), cv2.COLOR_BGR2RGB).copy()
    broken = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
    mk.stamp_watermark(broken, (box[0] + box[2] / 2), (box[1] + box[3] / 2), alpha=1.0)
    broken = cv2.cvtColor(broken, cv2.COLOR_BGR2RGB)

    mask = make_mask(clean.shape[:2], box, margin=0.25, feather=3)
    core = np.zeros(clean.shape[:2], bool)
    core[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = True

    def e(img):
        return float(np.mean(np.abs(img[core].astype(np.float32) - clean[core].astype(np.float32))))

    before = e(broken)
    out, did = LamaInpainter.get(str(LAMA)).fill(broken, mask, pad_ratio=1.2)
    assert did
    after = e(out)
    assert before > 60, f"fixture is not opaque (source err only {before:.1f})"
    assert after < before * 0.5, f"lama on opaque mark: {before:.1f} -> {after:.1f}"
    print(f"  lama    : opaque mark on detailed scene, core err {before:.1f} -> {after:.1f}")


# --------------------------------------------------------------------------- #
# 4. Full render
# --------------------------------------------------------------------------- #

def test_render_end_to_end_keeps_audio_and_frames():
    setup_module()
    meta, info = STATE["meta"], STATE["info"]
    boxes = [g["box"] for g in meta["gt"]]

    out = Path(tempfile.mkdtemp(prefix="wmout-")) / "clean.mp4"
    cfg = RenderConfig(mode="temporal", margin=0.2, feather=2, crf=20,
                       preset="ultrafast", keep_audio=True, temporal_window=9,
                       start_frame=10, end_frame=40)

    seen = []
    stats = render_video(
        STATE["video"], str(out), boxes, None, cfg,
        str(LAMA) if LAMA.exists() else None,
        progress=lambda d, t, u: seen.append((d, t, u)),
    )

    assert out.exists() and out.stat().st_size > 1000, "no output file"
    got = probe(str(out))
    want = 30
    assert abs(got.nb_frames - want) <= 3, f"wrote {got.nb_frames} frames, expected ~{want}"
    assert (got.width, got.height) == (mk.W, mk.H), f"{got.width}x{got.height}"
    assert got.has_audio, "audio was dropped"
    assert stats.frames == want, f"stats.frames {stats.frames}"
    assert len(seen) == want, f"progress fired {len(seen)} times"
    print(f"  render  : {stats.frames} frames -> {got.nb_frames}f, "
          f"{got.width}x{got.height}, audio={got.acodec}, "
          f"{out.stat().st_size // 1024} KB in {stats.seconds:.1f}s")

    shutil.rmtree(out.parent, ignore_errors=True)


def test_render_quality_beats_the_source():
    setup_module()
    meta = STATE["pan_meta"]
    boxes = [g["box"] for g in meta["gt"]]
    out = Path(tempfile.mkdtemp(prefix="wmq-")) / "q.mp4"
    cfg = RenderConfig(mode="temporal", margin=0.2, feather=2, crf=18,
                       preset="ultrafast", start_frame=10, end_frame=40,
                       temporal_window=25)
    render_video(STATE["pan"], str(out), boxes, None, cfg,
                 str(LAMA) if LAMA.exists() else None)

    before, after = [], []
    with FrameReader(STATE["pan"], mk.W, mk.H) as r:
        srcs = list(r.frames())
    with FrameReader(str(out), mk.W, mk.H) as r:
        outs = list(r.frames())

    for k, of in enumerate(outs):
        i = 10 + k
        if i >= len(meta["gt"]):
            break
        box = tuple(meta["gt"][i]["box"])
        gt = _pan_gt_frame(i)
        before.append(_region_error(srcs[i], gt, box, 0.2))
        after.append(_region_error(of, gt, box, 0.2))

    b, a = float(np.mean(before)), float(np.mean(after))
    # The pan fixture is high-frequency noise, the worst case for temporal fill, and
    # x264/opencv thread scheduling moves the result by +/-5, so keep the bar clearly
    # outside that noise band while still catching a render that does nothing (~0%).
    assert a < b * 0.75, f"rendered video barely improved: {b:.1f} -> {a:.1f}"
    print(f"  quality : mean err vs true background {b:.1f} -> {a:.1f} "
          f"({100 * (1 - a / b):.0f}% reduction)")
    shutil.rmtree(out.parent, ignore_errors=True)


def test_auto_mode_does_not_call_the_network_needlessly():
    """Regression guard for a 40x slowdown.

    The feathered mask edge always leaves a thin uncovered rim. An earlier version
    treated that rim as an unfilled hole, so `auto` invoked LaMa on every single
    frame — 0.15 fps on two cores — even when temporal recovery had already handled
    the whole watermark. Only the mask core counts as a hole now.
    """
    if not LAMA.exists():
        print("  auto    : SKIPPED (model not downloaded)")
        return
    setup_module()
    meta = STATE["pan_meta"]
    boxes = [g["box"] for g in meta["gt"]]
    out = Path(tempfile.mkdtemp(prefix="wmauto-")) / "a.mp4"
    cfg = RenderConfig(mode="auto", margin=0.2, feather=2, crf=22, preset="ultrafast",
                       start_frame=10, end_frame=35, temporal_window=25)
    conf = [0.8] * len(boxes)
    stats = render_video(STATE["pan"], str(out), boxes, conf, cfg, str(LAMA))

    n = stats.frames
    assert n == 25, f"expected 25 frames, got {n}"
    frac = stats.lama_used / n
    assert frac < 0.2, (
        f"neural fallback fired on {stats.lama_used}/{n} frames ({frac:.0%}) — "
        f"temporal was {stats.temporal_used}")
    assert stats.seconds < 30, f"{stats.seconds:.0f}s for {n} frames is far too slow"
    print(f"  auto    : {n} frames in {stats.seconds:.1f}s "
          f"({n / stats.seconds:.1f} fps) — temporal {stats.temporal_used}, "
          f"neural {stats.lama_used}, diffusion {stats.classic_used}")
    shutil.rmtree(out.parent, ignore_errors=True)


def test_tracker_also_works_on_the_panning_clip():
    setup_module()
    meta, info = STATE["pan_meta"], STATE["pan_info"]
    res = track_watermark(STATE["pan"], tuple(meta["gt"][0]["box"]), 0.0,
                          info.width, info.height, info.fps, moving=True)
    errs = []
    for i, g in enumerate(meta["gt"]):
        if i >= len(res.boxes):
            break
        bx, by, bw, bh = res.boxes[i]
        errs.append(float(np.hypot(bx + bw / 2 - g["center"][0], by + bh / 2 - g["center"][1])))
    med, p95 = float(np.median(errs)), float(np.percentile(errs, 95))
    assert med < 8 and p95 < 25, f"pan clip: median {med:.1f}px, p95 {p95:.1f}px"
    assert res.lost_frames == 0, f"lost {res.lost_frames} frames"
    print(f"  tracker : pan clip median {med:.1f}px, p95 {p95:.1f}px, "
          f"match {res.mean_confidence:.2f}, lost 0")


def test_preview_frame_helper():
    setup_module()
    meta = STATE["meta"]
    box = tuple(meta["gt"][12]["box"])
    cfg = RenderConfig(mode="temporal", margin=0.2, temporal_window=7)
    before, after, used = preview_frame(STATE["video"], 12 / mk.FPS, box, cfg,
                                        str(LAMA) if LAMA.exists() else None)
    assert before.shape == after.shape
    assert not np.array_equal(before, after), "preview changed nothing"
    print(f"  preview : engine={used}, {before.shape[1]}x{before.shape[0]}")


# --------------------------------------------------------------------------- #
# 5. HTTP API
# --------------------------------------------------------------------------- #

def test_api_smoke():
    setup_module()
    import app as appmod

    client = appmod.app.test_client()

    h = client.get("/health")
    assert h.status_code == 200 and h.get_json()["ok"]

    idx = client.get("/")
    assert idx.status_code == 200 and b"Watermark Remover" in idx.data

    mf = client.get("/manifest.webmanifest")
    assert mf.status_code == 200

    with open(STATE["video"], "rb") as fh:
        up = client.post("/api/upload", data={"file": (fh, "clip.mp4")},
                         content_type="multipart/form-data")
    assert up.status_code == 200, up.data
    mid = up.get_json()["id"]

    fr = client.get(f"/api/frame/{mid}?t=0.4&w=320")
    assert fr.status_code == 200 and fr.data[:2] == b"\xff\xd8"

    md = client.get(f"/api/media/{mid}")
    assert md.status_code == 200

    meta = STATE["meta"]
    box = meta["gt"][0]["box"]
    tr = client.post("/api/track", json={
        "id": mid, "box": box, "ref_w": mk.W, "t": 0.0, "moving": True})
    assert tr.status_code == 200, tr.data
    tid = tr.get_json()["job"]

    import time
    for _ in range(240):
        st = client.get(f"/api/job/{tid}").get_json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert st["state"] == "done", st.get("error")
    boxes = st["result"]["boxes"]
    assert len(boxes) >= 45, f"track returned {len(boxes)} boxes"

    rd = client.post("/api/render", json={
        "id": mid, "boxes": boxes, "ref_w": mk.W, "mode": "temporal",
        "start_frame": 5, "end_frame": 20, "crf": 24, "preset": "ultrafast"})
    assert rd.status_code == 200, rd.data
    rid = rd.get_json()["job"]
    for _ in range(400):
        st = client.get(f"/api/job/{rid}").get_json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert st["state"] == "done", st.get("error")
    res = st["result"]
    assert res["frames"] == 15, res
    assert res["size_mb"] > 0, res

    dl = client.get(res["url"])
    assert dl.status_code == 200 and len(dl.data) > 1000

    bad = client.post("/api/upload", data={"file": (__file__, "notes.txt")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400, "non-video upload should be rejected"

    print(f"  api     : upload/frame/media/track/render/download OK "
          f"({len(boxes)} boxes, {res['frames']} frames, {res['size_mb']} MB)")


# --------------------------------------------------------------------------- #

def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    print(f"\nRunning {len(tests)} tests\n" + "=" * 66)
    for fn in tests:
        name = fn.__name__[5:]
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL  {name}\n{traceback.format_exc()}")
        print("-" * 66)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
