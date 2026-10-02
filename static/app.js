/* Watermark Remover — front end. All logic talks to the local server only. */
(() => {
"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  drop: $("drop"), fileInput: $("fileInput"), uploadMeta: $("uploadMeta"),
  step1: $("step1"), step2: $("step2"), step3: $("step3"), step4: $("step4"),
  video: $("video"), overlay: $("overlay"), stage: $("stage"), stageHint: $("stageHint"),
  playBtn: $("playBtn"), playIco: $("playIco"), scrub: $("scrub"), timeLbl: $("timeLbl"),
  moveSeg: $("moveSeg"), margin: $("margin"), marginVal: $("marginVal"),
  trackBtn: $("trackBtn"), clearBox: $("clearBox"), trackInfo: $("trackInfo"),
  modes: $("modes"), previewBtn: $("previewBtn"), previewBox: $("previewBox"),
  pvBefore: $("pvBefore"), pvAfter: $("pvAfter"), pvEngine: $("pvEngine"),
  renderBtn: $("renderBtn"), progressWrap: $("progressWrap"), barFill: $("barFill"),
  progText: $("progText"), cancelBtn: $("cancelBtn"), resultBox: $("resultBox"),
  outVideo: $("outVideo"), downloadBtn: $("downloadBtn"), againBtn: $("againBtn"),
  renderStats: $("renderStats"), toast: $("toast"), engineBadge: $("engineBadge"),
  installBtn: $("installBtn"), ffPath: $("ffPath"),
};

const S = {
  mid: null, info: null, box: null, track: null, moving: true, mode: "classic",
  natW: 0, natH: 0, fps: 25, dragging: false, dragStart: null, jobs: {},
};

/* ------------------------------------------------------------------ utils */

let toastTimer = null;
function toast(msg, isErr = false) {
  el.toast.textContent = msg;
  el.toast.className = "toast show" + (isErr ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.className = "toast"; }, 3600);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const txt = await r.text();
  let data;
  try { data = JSON.parse(txt); } catch { data = { error: txt.slice(0, 300) }; }
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Poll a background job until it finishes; `onTick` receives each status. */
function pollJob(jobId, onTick) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      let st;
      try { st = await api(`/api/job/${jobId}`); }
      catch (e) { return reject(e); }
      if (onTick) onTick(st);
      if (st.state === "done") return resolve(st.result);
      if (st.state === "error") return reject(new Error(st.error || "Job failed"));
      if (st.state === "cancelled") return reject(new Error("Cancelled"));
      setTimeout(tick, 450);
    };
    tick();
  });
}

function busy(btn, on, label) {
  if (on) { btn.dataset.label = btn.textContent; btn.disabled = true; if (label) btn.textContent = label; }
  else { btn.disabled = false; if (btn.dataset.label) btn.textContent = btn.dataset.label; }
}

/* ------------------------------------------------------------------ boot */

(async function boot() {
  try {
    const h = await api("/health");
    el.ffPath.textContent = h.ffmpeg.split(/[\\/]/).pop();
    if (h.model.present) {
      el.engineBadge.textContent = `AI ready · ${h.model.size_mb} MB`;
      el.engineBadge.className = "badge badge-ok";
    } else {
      el.engineBadge.textContent = "AI model missing";
      el.engineBadge.className = "badge badge-off";
    }
  } catch { el.engineBadge.textContent = "server offline"; }

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
})();

let deferredPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredPrompt = e;
  el.installBtn.classList.remove("hidden");
});
el.installBtn.addEventListener("click", async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  el.installBtn.classList.add("hidden");
});

/* -------------------------------------------------------------- step 1 */

el.drop.addEventListener("click", () => el.fileInput.click());
el.drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
});
["dragenter", "dragover"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
el.drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) upload(f);
});
el.fileInput.addEventListener("change", () => {
  const f = el.fileInput.files?.[0];
  if (f) upload(f);
});

async function upload(file) {
  if (file.size > 8 * 1024 * 1024 * 1024) return toast("That file is over 8 GB.", true);
  const fd = new FormData();
  fd.append("file", file);

  el.drop.querySelector(".drop-title").textContent = "Reading…";
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    S.mid = res.id;
    S.info = res.info;
    S.fps = res.info.fps || 25;
    S.box = null; S.track = null;

    const mb = (res.info.size_bytes / 1048576).toFixed(1);
    el.uploadMeta.innerHTML =
      `<b>${escapeHtml(res.name)}</b><br>` +
      `${res.info.width}×${res.info.height} · ${res.info.fps.toFixed(2)} fps · ` +
      `${res.info.duration.toFixed(1)}s · ~${res.info.nb_frames} frames · ${mb} MB` +
      (res.info.has_audio ? " · audio ✓" : " · no audio");
    el.uploadMeta.classList.remove("hidden");

    el.video.src = `/api/media/${res.id}`;
    el.video.load();
    el.step2.classList.remove("hidden");
    el.step3.classList.add("hidden");
    el.step4.classList.add("hidden");
    el.resultBox.classList.add("hidden");
    el.progressWrap.classList.add("hidden");
    el.trackInfo.classList.add("hidden");
    el.previewBox.classList.add("hidden");
    el.trackBtn.disabled = true;
    setTimeout(() => el.step2.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
  } catch (e) {
    toast(e.message, true);
  } finally {
    el.drop.querySelector(".drop-title").textContent = "Tap to choose another video";
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* -------------------------------------------------------------- step 2 */

el.video.addEventListener("loadedmetadata", () => {
  S.natW = el.video.videoWidth;
  S.natH = el.video.videoHeight;
  if (S.info && S.natW) S.fps = S.info.fps || (S.natW ? S.fps : 25);
  el.scrub.value = 0;
  sizeCanvas();
  el.playBtn.disabled = false;
  draw();
  el.stageHint.style.opacity = "1";
});
el.video.addEventListener("timeupdate", () => {
  if (el.video.duration) el.scrub.value = (el.video.currentTime / el.video.duration) * 1000;
  el.timeLbl.textContent = el.video.currentTime.toFixed(1) + "s";
  if (el.video.paused) draw();
});
el.video.addEventListener("play", () => setPlayIcon(true));
el.video.addEventListener("pause", () => { setPlayIcon(false); draw(); });
el.scrub.addEventListener("input", () => {
  if (!el.video.duration) return;
  el.video.pause();
  el.video.currentTime = (el.scrub.value / 1000) * el.video.duration;
});
el.playBtn.addEventListener("click", () => {
  if (el.video.paused) el.video.play(); else el.video.pause();
});
function setPlayIcon(playing) {
  el.playIco.innerHTML = playing
    ? '<path d="M7 5h3.4v14H7zM13.6 5H17v14h-3.4z" fill="currentColor"/>'
    : '<path d="M8 5.5v13l11-6.5Z" fill="currentColor"/>';
}

function sizeCanvas() {
  const r = el.video.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  el.overlay.width = Math.max(2, Math.round(r.width * dpr));
  el.overlay.height = Math.max(2, Math.round(r.height * dpr));
  el.overlay._dpr = dpr;
  draw();
}
window.addEventListener("resize", sizeCanvas);
if (window.ResizeObserver) new ResizeObserver(sizeCanvas).observe(el.stage);

/** natural-pixel box -> canvas-pixel box */
function toCanvas(b) {
  const sx = el.overlay.width / S.natW;
  const sy = el.overlay.height / S.natH;
  return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
}
function toNatural(p) {
  return [
    Math.round(p[0] * S.natW / el.overlay.width),
    Math.round(p[1] * S.natH / el.overlay.height),
  ];
}

function currentTrackBox() {
  if (!S.track || !S.track.boxes.length) return null;
  const idx = Math.min(S.track.boxes.length - 1,
    Math.max(0, Math.floor(el.video.currentTime * S.fps)));
  return S.track.boxes[idx];
}

function draw() {
  const c = el.overlay;
  const g = c.getContext("2d");
  if (!c.width || !c.height) return;
  g.clearRect(0, 0, c.width, c.height);
  const dpr = c._dpr || 1;

  const box = S.track ? currentTrackBox() : S.box;
  if (!box) return;

  const [x, y, w, h] = toCanvas(box);
  const m = parseFloat(el.margin.value) / 100;
  const mx = w * m, my = h * m;

  // dim everything outside the repair area
  g.fillStyle = "rgba(0,0,0,.42)";
  g.beginPath();
  g.rect(0, 0, c.width, c.height);
  g.rect(Math.max(0, x - mx), Math.max(0, y - my), w + mx * 2, h + my * 2);
  g.fill("evenodd");

  g.strokeStyle = "rgba(255,255,255,.85)";
  g.lineWidth = 1.6 * dpr;
  g.setLineDash([5 * dpr, 4 * dpr]);
  g.strokeRect(x, y, w, h);

  g.strokeStyle = "rgba(91,140,255,.95)";
  g.lineWidth = 2 * dpr;
  g.setLineDash([]);
  g.strokeRect(x - mx, y - my, w + mx * 2, h + my * 2);

  // corner handles
  g.fillStyle = "#5b8cff";
  const hs = 4.5 * dpr;
  [[x - mx, y - my], [x + w + mx, y - my], [x - mx, y + h + my], [x + w + mx, y + h + my]]
    .forEach(([cx, cy]) => g.fillRect(cx - hs, cy - hs, hs * 2, hs * 2));

  if (S.track) {
    const ci = Math.min(S.track.confidence.length - 1,
      Math.max(0, Math.floor(el.video.currentTime * S.fps)));
    const conf = S.track.confidence[ci] ?? 1;
    g.font = `${11 * dpr}px -apple-system,system-ui,sans-serif`;
    const label = `match ${(conf * 100).toFixed(0)}%`;
    const tw = g.measureText(label).width + 12 * dpr;
    g.fillStyle = "rgba(0,0,0,.7)";
    g.fillRect(x - mx, y - my - 20 * dpr, tw, 17 * dpr);
    g.fillStyle = conf >= 0.28 ? "#3ddc97" : "#ff5c7a";
    g.fillText(label, x - mx + 6 * dpr, y - my - 7 * dpr);
  }
}

/* ---- pointer drawing ---- */
function ptrPos(e) {
  const r = el.overlay.getBoundingClientRect();
  return [
    (e.clientX - r.left) * (el.overlay.width / r.width),
    (e.clientY - r.top) * (el.overlay.height / r.height),
  ];
}
el.overlay.addEventListener("pointerdown", (e) => {
  if (!S.natW) return;
  e.preventDefault();
  el.overlay.setPointerCapture(e.pointerId);
  el.video.pause();
  S.dragging = true;
  S.dragStart = ptrPos(e);
  S.track = null;
  el.trackInfo.classList.add("hidden");
  el.step3.classList.add("hidden");
  el.step4.classList.add("hidden");
  el.stageHint.style.opacity = "0";
  draw();
});
el.overlay.addEventListener("pointermove", (e) => {
  if (!S.dragging) return;
  const p = ptrPos(e);
  const a = S.dragStart;
  const x0 = Math.min(a[0], p[0]), y0 = Math.min(a[1], p[1]);
  const w = Math.abs(p[0] - a[0]), h = Math.abs(p[1] - a[1]);
  const [nx, ny] = toNatural([x0, y0]);
  const [nx2, ny2] = toNatural([x0 + w, y0 + h]);
  S.box = [nx, ny, Math.max(4, nx2 - nx), Math.max(4, ny2 - ny)];
  draw();
});
["pointerup", "pointercancel"].forEach((ev) =>
  el.overlay.addEventListener(ev, () => {
    if (!S.dragging) return;
    S.dragging = false;
    if (S.box && S.box[2] > 6 && S.box[3] > 6) {
      el.trackBtn.disabled = false;
      el.trackBtn.classList.add("primary");
    } else {
      S.box = null;
      el.trackBtn.disabled = true;
      el.stageHint.style.opacity = "1";
    }
    draw();
  }));

el.clearBox.addEventListener("click", () => {
  S.box = null; S.track = null;
  el.trackBtn.disabled = true;
  el.trackInfo.classList.add("hidden");
  el.step3.classList.add("hidden");
  el.step4.classList.add("hidden");
  el.stageHint.style.opacity = "1";
  draw();
});

el.moveSeg.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  [...el.moveSeg.children].forEach((x) => x.classList.toggle("on", x === b));
  S.moving = b.dataset.v === "1";
});

el.margin.addEventListener("input", () => {
  el.marginVal.textContent = el.margin.value + "%";
  draw();
});

/* ---- tracking ---- */
el.trackBtn.addEventListener("click", async () => {
  if (!S.box) return;
  busy(el.trackBtn, true, "Tracking…");
  el.trackInfo.classList.add("hidden");
  try {
    const { job } = await post("/api/track", {
      id: S.mid, box: S.box, ref_w: S.natW,
      t: el.video.currentTime, moving: S.moving,
    });
    const res = await pollJob(job, (st) => {
      el.trackBtn.textContent = `Tracking ${Math.round((st.progress || 0) * 100)}%`;
    });
    S.track = res;

    const span = res.max_displacement.toFixed(0);
    let html = `<b>Tracked ${res.frames} frames.</b> Mean match ` +
      `${(res.mean_confidence * 100).toFixed(0)}% · moved ${span}px · ` +
      `${res.lost_frames} frame(s) lost.`;
    if (res.notes.length) html += "<br>" + res.notes.map(escapeHtml).join("<br>");
    html += "<br><i>Press play to watch the box follow the watermark. Redraw it if it drifts.</i>";
    el.trackInfo.innerHTML = html;
    el.trackInfo.classList.remove("hidden");

    el.step3.classList.remove("hidden");
    el.step4.classList.remove("hidden");
    el.resultBox.classList.add("hidden");
    el.progressWrap.classList.add("hidden");
    draw();
    setTimeout(() => el.step3.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(el.trackBtn, false);
    el.trackBtn.textContent = "Re-track";
  }
});

/* playback overlay animation */
(function loop() {
  if (!el.video.paused && S.track) draw();
  requestAnimationFrame(loop);
})();

/* -------------------------------------------------------------- step 3 */

el.modes.addEventListener("click", (e) => {
  const b = e.target.closest(".mode");
  if (!b) return;
  [...el.modes.children].forEach((x) => x.classList.toggle("on", x === b));
  S.mode = b.dataset.mode;
});

function params() {
  return {
    mode: S.mode,
    margin: parseFloat(el.margin.value) / 100,
    feather: 3,
    crf: parseInt($("crf").value, 10) || 18,
    preset: $("preset").value,
    scale: parseFloat($("scaleSel").value) || 1,
    keep_audio: $("keepAudio").checked,
    temporal_window: parseInt($("twin").value, 10) || 25,
    temporal_align: $("talign").checked,
  };
}

el.previewBtn.addEventListener("click", async () => {
  if (!S.box) return toast("Draw a box first.", true);
  busy(el.previewBtn, true, "Rendering preview…");
  try {
    // The watermark may have moved since it was drawn — preview the tracked box at
    // the current frame, not the original rectangle.
    const b = (S.track && currentTrackBox()) || S.box;
    const { job } = await post("/api/preview", {
      id: S.mid, box: b, ref_w: S.natW, t: el.video.currentTime, ...params(),
    });
    const res = await pollJob(job);
    el.pvBefore.src = "data:image/jpeg;base64," + res.before;
    el.pvAfter.src = "data:image/jpeg;base64," + res.after;
    el.pvEngine.textContent = `Engine used: ${res.engine} · ${res.w}×${res.h}`;
    el.previewBox.classList.remove("hidden");
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(el.previewBtn, false);
  }
});

/* -------------------------------------------------------------- step 4 */

el.renderBtn.addEventListener("click", async () => {
  if (!S.track) return toast("Track the watermark first.", true);
  busy(el.renderBtn, true, "Rendering…");
  el.progressWrap.classList.remove("hidden");
  el.resultBox.classList.add("hidden");
  el.barFill.style.width = "0%";
  el.progText.textContent = "Preparing…";
  S.jobs.render = null;

  try {
    const { job } = await post("/api/render", {
      id: S.mid, boxes: S.track.boxes, confidence: S.track.confidence,
      ref_w: S.natW, name: (S.info && S.info.name) || "clean", ...params(),
    });
    S.jobs.render = job;

    const res = await pollJob(job, (st) => {
      const pct = Math.round((st.progress || 0) * 100);
      el.barFill.style.width = pct + "%";
      const bits = [`${pct}%`];
      if (st.current) bits.push(`${st.current}/${st.total || "?"} frames`);
      if (st.fps) bits.push(`${st.fps.toFixed(1)} fps`);
      if (st.eta) bits.push(`~${Math.round(st.eta)}s left`);
      if (st.stage) bits.push(st.stage);
      el.progText.textContent = bits.join(" · ");
    });

    el.barFill.style.width = "100%";
    el.progText.textContent = "Done.";
    el.outVideo.src = res.url;
    el.downloadBtn.href = res.url;
    el.downloadBtn.download = res.filename;
    const parts = [`<b>${res.size_mb} MB</b>`, `${res.frames} frames`, `${res.seconds}s`];
    if (res.temporal_frames) parts.push(`temporal ${res.temporal_frames}`);
    if (res.lama_frames) parts.push(`neural ${res.lama_frames}`);
    if (res.classic_frames) parts.push(`diffusion ${res.classic_frames}`);
    el.renderStats.innerHTML = parts.join(" · ") +
      (res.notes.length ? "<br>" + res.notes.map(escapeHtml).join("<br>") : "");
    el.resultBox.classList.remove("hidden");
    setTimeout(() => el.resultBox.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
  } catch (e) {
    el.progText.textContent = e.message;
    toast(e.message, true);
  } finally {
    busy(el.renderBtn, false);
  }
});

el.cancelBtn.addEventListener("click", async () => {
  if (!S.jobs.render) return;
  try { await post(`/api/cancel/${S.jobs.render}`, {}); toast("Cancelling…"); } catch {}
});

el.againBtn.addEventListener("click", () => {
  location.reload();
});

})();
