# Watermark Remover

A local, offline web app for removing watermarks from **videos you own**. It runs
entirely on your own machine — your video is never uploaded anywhere, there is no
account, and there are no outbound network calls in the request path.

It is built for the hard case: a watermark that **moves** around the frame.

---

## Quick start

**macOS / Linux**
```bash
./run.sh
```

**Windows**
```
run.bat
```

Then open <http://127.0.0.1:8000>.

Manual setup, if you prefer:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_model.py     # 88 MB, only needed for the Neural mode
python app.py                     # --host 0.0.0.0 to reach it from your phone
```

### Using it from your phone

Run the server with `--host 0.0.0.0`, find your computer's LAN address
(`ipconfig getifaddr en0` on macOS, `ipconfig` on Windows), and open
`http://<that-address>:8000` on your phone. The interface is mobile-first and is an
installable PWA — in Chrome or Safari choose **Add to Home Screen** and it launches
full-screen like a native app.

Note that processing happens on the *computer*, not the phone. A phone CPU would take
minutes per frame.

---

## How it works

1. **Pick a video.** Anything FFmpeg can read: mp4, mov, mkv, webm, avi.
2. **Box the watermark.** Scrub to a frame where it is clearly visible and drag a
   rectangle over it. The blue outline shows the *safety margin* — the area that will
   actually be repaired. Watermarks usually bleed a few pixels past their letters, so
   give it some room.
3. **Track.** The app locates the watermark in every frame and you can press play to
   watch the box follow it.
4. **Pick a fill mode** and preview a single frame before committing.
5. **Render.** You get an MP4 with the original audio track intact.

### Only the text is ever erased

The app does **not** repaint the rectangle around your watermark. It isolates the thin
glyph strokes (high-pass + threshold + a temporal-stability vote) and heals just those
pixels, leaving the subject behind the text — a face, a mouth, a product — bit-for-bit
untouched. Where a stroke crosses a mouth or an eye, the fill reconstructs that small
gap from the immediately surrounding skin, so the feature stays. On a real face with
text across the mouth, the old box-based approach made the mouth error *worse* than
leaving the watermark (29.7 vs 21.7); tight text-only removal drops it to ~5 while
erasing the text. If clean text can't be isolated (very busy backgrounds) it falls back
to the safety-margin box so you never get a half-removed mark.

### Fill modes

| Mode | What it does | When to use it |
|---|---|---|
| **Fast · text only** *(default)* | Erases just the glyph strokes and heals the thin gaps | Almost always — keeps faces, ~10× faster |
| **AI · realistic** (LaMa) | Neural re-renders what's under each stroke | Text over a mouth/eye or busy background; slow on CPU |
| **Auto** | Recovers pixels from other frames; AI-fills only what's missing | Moving watermark on varied footage |
| **Temporal** | Rebuilds the hidden area from neighbouring frames only | Moving logo on steady footage |
| **Blur / pixelate** | Just obscures it | When you don't need it to look natural |

---

## Why the default is not "just use the AI"

This is the single most useful thing to know about the tool, and it came out of
measuring rather than assuming.

When a watermark is **semi-transparent** — as most are — the pixels underneath still
contain part of the real background. Throwing those pixels away and asking a network
to invent a plausible replacement can be *worse* than doing less. Measured on the
test clip, on a semi-transparent mark:

- recover pixels from other frames: error **2.3**
- replace them with a synthesis: error **11.7**

So `Auto` recovers first and only synthesises what recovery could not reach. The
neural model still earns its place — on an **opaque** mark over a detailed scene it
cut the error from **62.5 → 15.0** — and it is the only option when the watermark
never moves and there is nothing to recover from.

Two implementation details worth knowing:

- **Tracking matches on local contrast, not raw pixels.** A watermark's raw
  appearance is dominated by whatever colour sits behind it, which changes every
  frame. Correlating the raw patch found the right spot 2 times in 10; correlating a
  high-pass version of the frame found it 25 times in 25.
- **The network sees a crop, not the whole frame.** LaMa's ONNX graph has a fixed
  512×512 input. Squashing an entire 1080p frame into that blurs the repair, so the
  app crops a square around the watermark and runs on that — the reconstruction
  happens at full effective resolution.

---

## Speed

Ballpark on this 2-core CPU sandbox (640×360):

- **Fast · text only: ~28–60 frames/sec** — a 30 s clip renders in about a second
- Temporal / Auto: **~3.5 frames/sec**
- AI · realistic (LaMa): **~0.15 frames/sec** — about 6 seconds per frame

The Fast default is the one that makes 10-minute videos practical: it only heals a few
pixels of text per frame instead of repainting a whole rectangle, so it runs
near-realtime even on weak hardware. The AI mode is the slow one and is offered for the
shots where you want the most faithful reconstruction — it parallelises across cores
and is dramatically faster on a GPU or many-core machine. Drop **Output resolution** to
50% for a fast draft, check the result, then render full size.

---

## Verifying it

```bash
python tests/test_pipeline.py       # or: pytest tests/ -q
```

18 tests. They build two synthetic clips with a moving watermark over known
backgrounds, so "did it work?" is measured as error against the *true* background
rather than eyeballed:

```
tracker    : travel 225px, median err 0.5px, p95 6.4px, lost 0
temporal   : err 23.9 -> 2.8 vs true background, coverage 100%
lama       : controlled gradient err 130.6 -> 2.7, outside-mask byte-identical
lama       : opaque mark on detailed scene, core err 62.5 -> 15.0
render     : 30 frames, audio preserved, mean err 24.1 -> 10.5
auto       : 25 frames in 7.1s (3.5 fps) — temporal 25, neural 0
api        : upload/frame/media/track/render/download OK
```

The suite has caught real bugs, including a feathered-mask edge that made `Auto`
invoke the network on every frame (0.15 fps instead of 3.5) and a tracker that locked
onto background texture and drifted 95px off target.

---

## Layout

```
app.py                  Flask server and API
engine/
  ff.py                 FFmpeg: probe, decode, encode, mux audio
  tracker.py            follow a moving watermark
  inpaint.py            LaMa / temporal / diffusion fill
  textmask.py           tight glyph-only mask (erase text, keep the subject)
  render.py             the pipeline
  jobs.py               background jobs + progress
static/                 mobile-first UI (PWA)
scripts/fetch_model.py  downloads + checksum-verifies the LaMa model
tests/                  test suite and clip generator
models/lama.onnx        downloaded separately, Apache-2.0
```

Uploads and outputs go to `data/` and are deleted after 6 hours. Delete the folder to
clear everything immediately.

FFmpeg comes from your `PATH`, or from the `imageio-ffmpeg` wheel if you do not have
it installed. Override with `WMREMOVER_FFMPEG=/path/to/ffmpeg`.

---

## Deploy it so it's live with upload

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/udhayakumar24/Watermark-remover)

GitHub Pages can only show a static page — it cannot run the Python engine, so it has no
upload box. To get a *working* public app (with upload) you need a host that runs Python.
A `Dockerfile` and a `render.yaml` are included:

1. Push this repo to GitHub (done).
2. On [render.com](https://render.com): **New → Blueprint**, choose the repo — it reads
   `render.yaml` and builds the `Dockerfile` (FFmpeg + OpenCV + LaMa baked in).
3. When the deploy finishes, Render gives you a public `https://…onrender.com` URL where
   you can upload videos and render.

The same `Dockerfile` works on any VPS or on Railway/Fly.io. For fully-private use, skip
the cloud and just run `./run.sh` on your own machine.

---

## Legal

Use this only on videos you own or have permission to edit. Removing a watermark from
someone else's work — to reuse it, to pass it off as your own, or to strip an
attribution — infringes their copyright and, in many places, anti-circumvention law
as well. Legitimate uses include your own screen recordings, recovering your own
footage behind a preview overlay, and cleaning up material you hold the rights to.

The LaMa model is [Apache-2.0](https://huggingface.co/opencv/inpainting_lama) and is
downloaded separately so this repository stays small.
