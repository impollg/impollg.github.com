// Lo que ibas a hacer — app principal.
import { FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import {
  JOINTS, J, NJ, PD, SD, W, F, DT, BONES,
  rootScale, toSample, samplePixels, pixelsToSample, lerpArr, poseError,
  buildQuery, searchArchive, futureFromArchive, futureExtrap,
} from './core.js';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const ACCENT = '#a8463f';

// ───────────────────────── Parámetros ─────────────────────────
const DEFAULTS = {
  facing: 'user', mirror: true, fit: 'cover', forceLandscape: true,
  numPoses: 1, requireLegs: true, maskThr: 0.5,
  kInit: 3, kMin: 2, kMax: 7, kCooldown: 600,
  obeyThr: 0.15, devThr: 0.45, horizon: 0.8, recompute: 200, blendMs: 150,
  matchMax: 0.6, velW: 3, rootVelW: 2, minArchiveSec: 120,
  persistMin: 5, persistMax: 8, lostMs: 250, minSessionSec: 3, maxSamples: 30000,
  ghostMinA: 0.2, ghostMaxA: 0.45, trail: 0.85, blurScale: 6, blurPx: 4, capsule: 0.34,
  flashMs: 150, accent: false, debug: false, recordSeed: false,
};
const SCHEMA = [
  ['Cámara y pantalla'],
  ['facing', 'Cámara', { options: [['user', 'Frontal'], ['environment', 'Trasera']] }],
  ['mirror', 'Espejo', 'bool'],
  ['fit', 'Encuadre', { options: [['cover', 'Llenar pantalla'], ['contain', 'Mostrar todo el cuadro']] }],
  ['forceLandscape', 'Pedir orientación horizontal', 'bool'],
  ['numPoses', 'Personas detectadas (usa la más grande)', [1, 3, 1]],
  ['requireLegs', 'Exigir rodillas visibles (cuerpo entero)', 'bool'],
  ['maskThr', 'Umbral de silueta', [0.2, 0.8, 0.05]],
  ['Predicción'],
  ['kInit', 'Fantasmas al entrar (K inicial)', [1, 7, 1]],
  ['kMin', 'K mínimo', [1, 7, 1]],
  ['kMax', 'K máximo', [1, 7, 1]],
  ['kCooldown', 'Pausa entre cambios de K (ms)', [0, 2000, 50]],
  ['obeyThr', 'Umbral de obediencia (torsos)', [0.05, 0.5, 0.01]],
  ['devThr', 'Umbral de desvío (torsos)', [0.15, 1.2, 0.01]],
  ['horizon', 'Horizonte del fantasma (s)', [0.3, 1.5, 0.1]],
  ['recompute', 'Recalcular cada (ms)', [100, 600, 20]],
  ['blendMs', 'Transición entre predicciones (ms)', [0, 400, 10]],
  ['matchMax', 'Distancia máxima de coincidencia', [0.1, 2, 0.05]],
  ['velW', 'Peso de la velocidad', [0, 8, 0.25]],
  ['rootVelW', 'Peso del desplazamiento', [0, 8, 0.25]],
  ['minArchiveSec', 'Archivo mínimo para usarlo (s)', [0, 600, 10]],
  ['Persistencia y archivo'],
  ['persistMin', 'Persistencia mínima (s)', [0, 15, 0.5]],
  ['persistMax', 'Persistencia máxima (s)', [0, 15, 0.5]],
  ['lostMs', 'Tiempo sin cuerpo para darlo por ido (ms)', [100, 1500, 50]],
  ['minSessionSec', 'Sesión mínima para archivar (s)', [1, 20, 0.5]],
  ['maxSamples', 'Tope del archivo (muestras)', [3000, 60000, 1000]],
  ['recordSeed', 'Grabar sesiones como semilla', 'bool'],
  ['Imagen'],
  ['ghostMinA', 'Opacidad mínima fantasma', [0, 1, 0.01]],
  ['ghostMaxA', 'Opacidad máxima fantasma', [0, 1, 0.01]],
  ['trail', 'Estela (permanencia por cuadro)', [0, 0.98, 0.01]],
  ['blurScale', 'Desenfoque por reducción (×)', [1, 12, 1]],
  ['blurPx', 'Desenfoque adicional (px, si el navegador lo soporta)', [0, 20, 1]],
  ['capsule', 'Grosor de las cápsulas', [0.1, 0.8, 0.01]],
  ['flashMs', 'Destello de fusión (ms)', [0, 600, 10]],
  ['accent', 'Acento rojo en el fantasma más probable', 'bool'],
  ['debug', 'Capa de depuración', 'bool'],
];
const LS_KEY = 'lqiah.params.v1';
let P = loadParams();
function loadParams() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(LS_KEY) || '{}') }; } catch { return { ...DEFAULTS }; }
}
function saveParams() { try { localStorage.setItem(LS_KEY, JSON.stringify(P)); } catch {} }

// ───────────────────────── DOM y lienzos ─────────────────────────
const $ = (s) => document.querySelector(s);
const video = $('#cam'), canvas = $('#stage'), ctx = canvas.getContext('2d');
const startEl = $('#start'), startErr = $('#startErr'), rotateEl = $('#rotate'), panel = $('#panel');
const maskCanvas = document.createElement('canvas'), mctx = maskCanvas.getContext('2d');
const ghostCanvas = document.createElement('canvas'), gctx = ghostCanvas.getContext('2d');
const tmpCanvas = document.createElement('canvas'), tctx = tmpCanvas.getContext('2d');
let maskImg = null, maskReady = false;
let VW = 640, VH = 480, gw = 1, gh = 1, dpr = 1, fitScale = 1;
const filterOK = (() => { try { const c = document.createElement('canvas').getContext('2d'); c.filter = 'blur(2px)'; return c.filter === 'blur(2px)'; } catch { return false; } })();

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  canvas.style.width = innerWidth + 'px'; canvas.style.height = innerHeight + 'px';
  canvas.width = Math.round(innerWidth * dpr); canvas.height = Math.round(innerHeight * dpr);
  rotateEl.style.display = started && P.forceLandscape && innerHeight > innerWidth ? 'flex' : 'none';
}
function sizeOffscreen() {
  const bs = Math.max(1, P.blurScale);
  gw = Math.max(1, Math.ceil(VW / bs)); gh = Math.max(1, Math.ceil(VH / bs));
  ghostCanvas.width = tmpCanvas.width = gw; ghostCanvas.height = tmpCanvas.height = gh;
}
function applyFit(c) {
  const cw = canvas.width, ch = canvas.height;
  const sc = P.fit === 'contain' ? Math.min(cw / VW, ch / VH) : Math.max(cw / VW, ch / VH);
  const ox = (cw - VW * sc) / 2, oy = (ch - VH * sc) / 2;
  fitScale = sc;
  if (P.mirror) c.setTransform(-sc, 0, 0, sc, cw - ox, oy); else c.setTransform(sc, 0, 0, sc, ox, oy);
}
addEventListener('resize', resize);

// ───────────────────────── Estado ─────────────────────────
// Estados: idle (reposo) → enter (entrada) → play (interacción) → persist (persistencia) → archivo → idle
let started = false, state = 'idle';
let stream = null, landmarker = null, delegate = '-';
let session = null, seg = [], segT = [], lastDet = null, nextGrid = 0;
let curPose = null, curRoot = null, curPoseT = -1e9, lastSeen = -1e9;
let K = P.kInit, lastKChange = 0, lastPredict = 0, checks = [];
let ghosts = [], gid = 0;
let persistStart = 0, persistEnd = 0;
let lastErrMin = NaN, searchMs = 0;
let frameDt = 16, lastFrame = 0, rFrames = 0, dFrames = 0, rFps = 0, dFps = 0, lastFpsT = 0, lastResultT = 0;

// ───────────────────────── Archivo (IndexedDB) ─────────────────────────
let db = null, records = [], flat = new Float32Array(0), archiveTotal = 0, archiveVer = 0, A = null;
function openDB() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('lo-que-ibas-a-hacer', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('sessions', { keyPath: 'id' });
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
function dbReq(mode, fn) {
  if (!db) return Promise.resolve(null);
  return new Promise((res) => {
    try { const tx = db.transaction('sessions', mode); const r = fn(tx.objectStore('sessions')); tx.oncomplete = () => res(r && r.result); tx.onerror = () => res(null); }
    catch { res(null); }
  });
}
const dbAll = () => dbReq('readonly', (s) => s.getAll());
const dbPut = (rec) => dbReq('readwrite', (s) => s.put(rec));
const dbDel = (id) => dbReq('readwrite', (s) => s.delete(id));

async function loadArchive() {
  try { db = await openDB(); records = (await dbAll()) || []; } catch { db = null; records = []; }
  records = records.filter((r) => r && r.data && r.n >= W + F);
  if (!records.some((r) => r.seed)) {
    try {
      const res = await fetch('seed.json', { cache: 'no-cache' });
      if (res.ok) addRecords(parseExport(await res.json(), true));
    } catch {}
  }
  rebuild();
}
function addRecords(recs) {
  for (const r of recs) { records = records.filter((x) => x.id !== r.id); records.push(r); dbPut(r); }
  enforceCap(); rebuild();
}
function enforceCap() {
  let total = records.reduce((a, r) => a + r.n, 0);
  const visits = records.filter((r) => !r.seed).sort((a, b) => a.created - b.created);
  while (total > P.maxSamples && visits.length) {
    const r = visits.shift(); total -= r.n; records.splice(records.indexOf(r), 1); dbDel(r.id);
  }
}
function rebuild() {
  archiveTotal = records.reduce((a, r) => a + r.n, 0);
  flat = new Float32Array(archiveTotal * SD);
  const starts = new Int32Array(records.length), ns = new Int32Array(records.length), groups = new Int32Array(records.length);
  const gmap = new Map(); let off = 0;
  records.forEach((r, i) => {
    flat.set(r.data, off * SD); starts[i] = off; ns[i] = r.n;
    let gi = gmap.get(r.group); if (gi === undefined) { gi = gmap.size; gmap.set(r.group, gi); }
    groups[i] = gi; off += r.n;
  });
  archiveVer++;
  A = { flat, starts, ns, groups };
  if (worker) worker.postMessage({ type: 'archive', ver: archiveVer, A: { flat: flat.slice(), starts, ns, groups } });
}
function parseExport(obj, forceSeed) {
  if (!obj || !Array.isArray(obj.sessions)) throw new Error('Formato de archivo no reconocido');
  return obj.sessions
    .filter((s) => s && s.n >= W + F && Array.isArray(s.data) && s.data.length === s.n * SD)
    .map((s) => ({
      id: String(s.id || Math.random().toString(36).slice(2)), group: String(s.group || s.id),
      seed: forceSeed ? true : !!s.seed, created: s.created || Date.now(), n: s.n, data: Float32Array.from(s.data),
    }));
}
function exportArchive(onlySeed) {
  const list = records.filter((r) => !onlySeed || r.seed);
  const out = {
    format: 'lo-que-ibas-a-hacer', version: 1, sd: SD, hz: 10, joints: JOINTS, exported: new Date().toISOString(),
    sessions: list.map((r) => ({ id: r.id, group: r.group, seed: !!r.seed, created: r.created, n: r.n, data: Array.from(r.data, (x) => Math.round(x * 1e4) / 1e4) })),
  };
  const blob = new Blob([JSON.stringify(out)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = onlySeed ? 'seed.json' : `archivo-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ───────────────────────── Búsqueda (Web Worker con respaldo) ─────────────────────────
let worker = null, qid = 0, pendingLive = 0;
const pending = new Map();
try {
  const src = `${searchArchive.toString()}
let A = null, ver = 0;
onmessage = (e) => {
  const m = e.data;
  if (m.type === 'archive') { A = m.A; ver = m.ver; return; }
  const t = performance.now();
  const r = A ? searchArchive(A, m.q, m.k, m.maxDist, m.velW, m.rootVelW) : [];
  postMessage({ id: m.id, ver, r, ms: performance.now() - t });
};`;
  worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  worker.onmessage = (e) => onSearch(e.data);
  worker.onerror = () => { worker = null; };
} catch { worker = null; }

function query(q, k, c) {
  const id = ++qid; c.sent = performance.now(); pending.set(id, c);
  if (c.mode === 'live') pendingLive = id;
  const msg = { type: 'query', id, q, k, maxDist: P.matchMax, velW: P.velW, rootVelW: P.rootVelW };
  if (worker) worker.postMessage(msg);
  else setTimeout(() => {
    const t = performance.now();
    const r = A ? searchArchive(A, q, k, P.matchMax, P.velW, P.rootVelW) : [];
    onSearch({ id, ver: archiveVer, r, ms: performance.now() - t });
  }, 0);
}
function onSearch({ id, ver, r, ms }) {
  const c = pending.get(id); if (!c) return;
  pending.delete(id); searchMs = ms;
  if (c.mode === 'live' && pendingLive === id) pendingLive = 0;
  if (ver !== archiveVer) r = [];
  const now = performance.now();
  if (c.mode === 'live') applyLive(r, c, now); else applyGhost(r, c, now);
}
function sweepPending(now) {
  for (const [id, c] of pending) if (now - c.sent > 1500) { pending.delete(id); if (c.ghost) c.ghost.requery = false; if (pendingLive === id) pendingLive = 0; }
}

// ───────────────────────── Cámara y modelo ─────────────────────────
async function openCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: P.facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
  });
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => (video.onloadedmetadata = r));
  updateVideoSize();
}
function updateVideoSize() {
  if (video.videoWidth && (video.videoWidth !== VW || video.videoHeight !== VH)) {
    VW = video.videoWidth; VH = video.videoHeight; sizeOffscreen(); maskReady = false;
  }
}
async function loadModel() {
  const fs = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (d) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate: d }, runningMode: 'VIDEO',
    numPoses: P.numPoses, outputSegmentationMasks: true,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { landmarker = await PoseLandmarker.createFromOptions(fs, opts('GPU')); delegate = 'GPU'; }
  catch { landmarker = await PoseLandmarker.createFromOptions(fs, opts('CPU')); delegate = 'CPU'; }
}

// ───────────────────────── Detección ─────────────────────────
const useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
let lastVideoTime = -1, lastTs = 0;
function startDetect() {
  const tick = () => {
    try { detect(); } catch (e) { console.warn(e); }
    if (useRVFC) video.requestVideoFrameCallback(tick); else requestAnimationFrame(tick);
  };
  tick();
}
function detect() {
  if (!landmarker || video.readyState < 2) return;
  if (!useRVFC) { if (video.currentTime === lastVideoTime) return; lastVideoTime = video.currentTime; }
  updateVideoSize();
  let ts = performance.now(); if (ts <= lastTs) ts = lastTs + 0.01; lastTs = ts;
  landmarker.detectForVideo(video, ts, (res) => onResult(res, ts)); // las máscaras se liberan solas al volver
  dFrames++; lastResultT = ts;
}
function onResult(res, t) {
  const L = res.landmarks; if (!L || !L.length) return;
  let idx = 0;
  if (L.length > 1) {
    let best = -1;
    L.forEach((lm, i) => {
      let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
      for (const l of lm) { if (l.x < x0) x0 = l.x; if (l.x > x1) x1 = l.x; if (l.y < y0) y0 = l.y; if (l.y > y1) y1 = l.y; }
      const a = (x1 - x0) * (y1 - y0); if (a > best) { best = a; idx = i; }
    });
  }
  const lm = L[idx];
  const p = new Float32Array(PD), v = new Float32Array(NJ);
  for (let i = 0; i < NJ; i++) { const l = lm[JOINTS[i]]; p[2 * i] = l.x * VW; p[2 * i + 1] = l.y * VH; v[i] = l.visibility ?? 1; }
  const need = [J.ls, J.rs, J.lh, J.rh]; if (P.requireLegs) need.push(J.lk, J.rk);
  if (!need.every((i) => v[i] > 0.5)) return;
  const m = res.segmentationMasks && res.segmentationMasks[idx];
  if (m) copyMask(m);
  lastSeen = t;
  onBody(t, p, v);
}
function copyMask(m) {
  const w = m.width, h = m.height;
  if (maskCanvas.width !== w || maskCanvas.height !== h || !maskImg) {
    maskCanvas.width = w; maskCanvas.height = h; maskImg = mctx.createImageData(w, h);
    maskImg.data.fill(255);
  }
  const f = m.getAsFloat32Array(), d = maskImg.data;
  const lo = P.maskThr - 0.12, k = 255 / 0.24;
  for (let i = 0, j = 3; i < f.length; i++, j += 4) d[j] = (f[i] - lo) * k; // Uint8ClampedArray recorta
  mctx.putImageData(maskImg, 0, 0);
  maskReady = true;
}

// ───────────────────────── Sesión y remuestreo a 10 Hz ─────────────────────────
function onBody(t, p, v) {
  if (state === 'idle') beginSession();
  else if (state === 'persist') resumeSession();
  const s = toSample(p, v, lastDet ? lastDet.s : null);
  curPose = samplePixels(s); curRoot = [s[26], s[27], s[28]]; curPoseT = t;
  resample(t, s);
  if (state === 'enter' && seg.length >= W) state = 'play';
}
function resample(t, s) {
  if (!lastDet) { pushSample(t, s); nextGrid = t + DT; lastDet = { t, s }; return; }
  const t0 = lastDet.t, s0 = lastDet.s;
  while (nextGrid <= t) { pushSample(nextGrid, lerpArr(s0, s, (nextGrid - t0) / (t - t0))); nextGrid += DT; }
  lastDet = { t, s };
}
function pushSample(t, s) {
  seg.push(s); segT.push(t);
  if (seg.length > 6000) { session.segments.push(seg.splice(0, 3000)); segT.splice(0, 3000); }
}
function beginSession() {
  session = { group: 'v' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5), segments: [] };
  seg = []; segT = []; lastDet = null; K = P.kInit; ghosts = []; checks = []; lastPredict = 0; state = 'enter';
}
function resumeSession() {
  if (seg.length) session.segments.push(seg);
  seg = []; segT = []; lastDet = null; checks = []; lastPredict = 0; state = 'enter';
  for (const g of ghosts) g.requery = false;
}
function enterPersist(now) {
  state = 'persist'; persistStart = now;
  persistEnd = now + (P.persistMin + Math.random() * Math.max(0, P.persistMax - P.persistMin)) * 1000;
  checks = []; pendingLive = 0;
}
function endSession() {
  state = 'idle';
  const segs = session ? [...session.segments] : [];
  if (seg.length) segs.push(seg);
  const group = session ? session.group : 'v' + Date.now().toString(36);
  ghosts = []; checks = []; seg = []; segT = []; lastDet = null; session = null;
  const recs = [];
  segs.forEach((sg, i) => {
    if (sg.length < Math.max(W + F, P.minSessionSec * 10)) return;
    const data = new Float32Array(sg.length * SD);
    sg.forEach((s, j) => data.set(s, j * SD));
    recs.push({ id: `${group}-${i}`, group, seed: !!P.recordSeed, created: Date.now(), n: sg.length, data });
  });
  if (recs.length) addRecords(recs);
}

// ───────────────────────── Predicción ─────────────────────────
function predict() {
  if (seg.length < 1 || pendingLive) return;
  const last = seg[seg.length - 1], prev = seg.length > 1 ? seg[seg.length - 2] : null;
  const c = { mode: 'live', t0: segT[segT.length - 1], cur: [last[26], last[27], last[28]], last, prev, k: K };
  if (seg.length >= W && archiveTotal >= P.minArchiveSec * 10) query(buildQuery(seg.slice(-W), P.velW, P.rootVelW), K, c);
  else applyLive([], c, performance.now());
}
function applyLive(r, c, now) {
  if (state !== 'enter' && state !== 'play') return;
  const futs = [];
  for (const m of r) futs.push({ key: 'g' + m.g, frames: futureFromArchive(flat, m.e, c.cur, c.last), sim: 1 - m.dist / P.matchMax });
  for (let v = 0; futs.length < c.k; v++) futs.push({ key: 'x' + v, frames: futureExtrap(c.last, c.prev, v, c.cur), sim: 0.4 });
  futs.length = Math.min(futs.length, c.k);
  assignGhosts(futs, c.t0, now);
  const hi = Math.max(1, Math.min(F, Math.round(P.horizon * 10)));
  checks.push({
    due: c.t0 + hi * DT,
    items: futs.map((f) => ({ key: f.key, pose: f.frames[hi], moved: poseError(f.frames[hi], f.frames[0], c.cur[2]) })),
  });
  if (checks.length > 40) checks.splice(0, checks.length - 40);
}
function newGhost(f, t0, now) {
  return { id: ++gid, key: f.key, frames: f.frames, t0, offset: P.horizon * 1000, sim: f.sim, alpha: 0, state: 'live', from: null, blendStart: 0, cur: null, requery: false, born: now };
}
function assignGhosts(futs, t0, now) {
  const live = ghosts.filter((g) => g.state === 'live');
  const used = new Set(), keys = new Set(futs.map((f) => f.key));
  for (const f of futs) {
    let g = live.find((x) => !used.has(x) && x.key === f.key) || live.find((x) => !used.has(x) && !keys.has(x.key));
    if (g) {
      g.from = g.cur; g.blendStart = now; g.frames = f.frames; g.t0 = t0; g.offset = P.horizon * 1000; g.key = f.key; g.sim = f.sim;
    } else { g = newGhost(f, t0, now); ghosts.push(g); }
    used.add(g);
  }
  for (const g of live) if (!used.has(g)) g.state = 'dying';
}
function evalChecks(now) {
  while (checks.length && checks[0].due <= now) {
    const c = checks.shift();
    if (state !== 'play' || !curPose || now - curPoseT > 300 || now - c.due > 500) continue;
    let minE = Infinity, minKey = null, allDev = true;
    for (const it of c.items) {
      const e = poseError(it.pose, curPose, curRoot[2]);
      if (e <= P.devThr) allDev = false;
      if (it.moved > P.obeyThr && e < minE) { minE = e; minKey = it.key; } // obedecer exige que el fantasma anunciara un movimiento
    }
    lastErrMin = minE;
    if (now - lastKChange < P.kCooldown) continue;
    if (minE < P.obeyThr) {
      const g = ghosts.find((x) => x.state === 'live' && x.key === minKey);
      if (g) { g.state = 'fusing'; g.fuseStart = now; K = Math.max(P.kMin, K - 1); lastKChange = now; }
    } else if (allDev) { K = Math.min(P.kMax, K + 1); lastKChange = now; }
  }
}
// Persistencia: cada fantasma termina su continuación y vuelve a consultar el archivo con su propia ventana.
function persistStep(now) {
  for (const g of ghosts) {
    if (g.state !== 'live' || g.requery) continue;
    if ((now - g.t0 + g.offset) / DT < F) continue;
    const win = []; for (let j = F - W + 1; j <= F; j++) win.push(pixelsToSample(g.frames[j]));
    const last = win[W - 1], prev = win[W - 2];
    const c = { mode: 'ghost', ghost: g, last, prev, cur: [last[26], last[27], last[28]] };
    g.requery = true;
    if (archiveTotal >= P.minArchiveSec * 10) query(buildQuery(win, P.velW, P.rootVelW), 1, c);
    else applyGhost([], c, now);
  }
}
function applyGhost(r, c, now) {
  const g = c.ghost; g.requery = false;
  if (state !== 'persist' || !ghosts.includes(g)) return;
  g.frames = r.length ? futureFromArchive(flat, r[0].e, c.cur, c.last) : futureExtrap(c.last, c.prev, 0, c.cur, 0.8);
  g.from = g.cur; g.blendStart = now; g.t0 = now; g.offset = 0;
}

// ───────────────────────── Render ─────────────────────────
function ghostPose(g, now) {
  const idx = Math.min(F, Math.max(0, (now - g.t0 + g.offset) / DT));
  const i0 = Math.floor(idx), i1 = Math.min(F, i0 + 1);
  let pose = lerpArr(g.frames[i0], g.frames[i1], idx - i0);
  if (g.from && P.blendMs > 0 && now - g.blendStart < P.blendMs) {
    const b = (now - g.blendStart) / P.blendMs;
    pose = lerpArr(g.from, pose, b * b * (3 - 2 * b));
  }
  return pose;
}
function updateGhost(g, now) {
  if (g.state === 'fusing') {
    const el = now - g.fuseStart;
    if (curPose && g.cur) g.cur = lerpArr(g.cur, curPose, Math.min(1, frameDt / 120));
    if (el < P.flashMs) return 1;
    const f = 1 - (el - P.flashMs) / 300;
    if (f <= 0) { g.dead = true; return 0; }
    return f;
  }
  g.cur = ghostPose(g, now);
  const target = P.ghostMinA + (P.ghostMaxA - P.ghostMinA) * Math.max(0, Math.min(1, g.sim));
  if (g.state === 'dying') {
    g.alpha -= frameDt / 200 * Math.max(0.05, target);
    if (g.alpha <= 0) { g.dead = true; return 0; }
  } else if (g.alpha < target) g.alpha = Math.min(target, g.alpha + frameDt / 250 * target);
  else g.alpha = Math.max(target, g.alpha - frameDt / 400 * target);
  return g.alpha;
}
function drawBody(c, p, color) {
  const s = rootScale(p)[2];
  c.fillStyle = color; c.strokeStyle = color; c.lineCap = 'round'; c.lineJoin = 'round';
  c.beginPath();
  c.moveTo(p[2 * J.ls], p[2 * J.ls + 1]); c.lineTo(p[2 * J.rs], p[2 * J.rs + 1]);
  c.lineTo(p[2 * J.rh], p[2 * J.rh + 1]); c.lineTo(p[2 * J.lh], p[2 * J.lh + 1]);
  c.closePath(); c.fill();
  c.lineWidth = s * P.capsule;
  c.beginPath();
  for (const [a, b] of BONES) { c.moveTo(p[2 * a], p[2 * a + 1]); c.lineTo(p[2 * b], p[2 * b + 1]); }
  c.moveTo((p[2 * J.ls] + p[2 * J.rs]) / 2, (p[2 * J.ls + 1] + p[2 * J.rs + 1]) / 2); c.lineTo(p[0], p[1]);
  c.stroke();
  c.beginPath(); c.arc(p[0], p[1], s * 0.3, 0, Math.PI * 2); c.fill();
}
function drawGhost(p, a, color) {
  // Se dibuja opaco en un lienzo auxiliar y se compone con la opacidad, para que las cápsulas no se sumen entre sí.
  tctx.setTransform(1, 0, 0, 1, 0, 0); tctx.clearRect(0, 0, gw, gh);
  tctx.setTransform(gw / VW, 0, 0, gh / VH, 0, 0);
  drawBody(tctx, p, color);
  gctx.globalAlpha = Math.min(1, a); gctx.drawImage(tmpCanvas, 0, 0); gctx.globalAlpha = 1;
}
function draw(now) {
  const cw = canvas.width, ch = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; ctx.filter = 'none';
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cw, ch);

  // Estela: acumulación de cuadros con desvanecimiento.
  gctx.setTransform(1, 0, 0, 1, 0, 0);
  gctx.globalCompositeOperation = 'destination-out';
  gctx.fillStyle = `rgba(0,0,0,${1 - P.trail})`; gctx.fillRect(0, 0, gw, gh);
  gctx.globalCompositeOperation = 'source-over';

  const fade = state === 'persist' ? Math.max(0, 1 - (now - persistStart) / Math.max(1, persistEnd - persistStart)) : 1;
  let top = null;
  if (P.accent) for (const g of ghosts) if (g.state === 'live' && (!top || g.sim > top.sim)) top = g;
  for (let i = ghosts.length - 1; i >= 0; i--) {
    const g = ghosts[i];
    const a = updateGhost(g, now);
    if (g.dead) { ghosts.splice(i, 1); continue; }
    if (!g.cur || a * fade <= 0.003) continue;
    drawGhost(g.cur, a * fade, g === top ? ACCENT : '#fff');
  }

  applyFit(ctx);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  if (filterOK && P.blurPx > 0) ctx.filter = `blur(${P.blurPx * dpr}px)`;
  ctx.drawImage(ghostCanvas, 0, 0, VW, VH);
  ctx.filter = 'none';

  // Silueta: lo único nítido. Desaparece de golpe cuando se pierde el cuerpo.
  if ((state === 'enter' || state === 'play') && now - lastSeen < P.lostMs && maskReady) ctx.drawImage(maskCanvas, 0, 0, VW, VH);

  if (P.debug) drawDebug(now);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (state === 'idle') {
    ctx.fillStyle = '#555'; ctx.font = `${Math.round(13 * dpr)}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    ctx.textBaseline = 'alphabetic'; ctx.fillText('Lo que ibas a hacer', 18 * dpr, ch - 18 * dpr);
  }
}
function drawDebug(now) {
  applyFit(ctx);
  ctx.lineWidth = 2 / fitScale;
  if (curPose && now - curPoseT < 500) {
    ctx.strokeStyle = '#e33'; ctx.beginPath();
    for (const [a, b] of BONES) { ctx.moveTo(curPose[2 * a], curPose[2 * a + 1]); ctx.lineTo(curPose[2 * b], curPose[2 * b + 1]); }
    ctx.stroke();
  }
  ctx.fillStyle = '#3c9';
  for (const g of ghosts) if (g.cur) for (let c = 0; c < PD; c += 2) ctx.fillRect(g.cur[c] - 2 / fitScale, g.cur[c + 1] - 2 / fitScale, 4 / fitScale, 4 / fitScale);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const seeds = records.filter((r) => r.seed).reduce((a, r) => a + r.n, 0);
  const lines = [
    `estado ${state} · K ${K} · fantasmas ${ghosts.length}`,
    `render ${rFps} fps · pose ${dFps} fps · ${delegate} · video ${VW}×${VH}`,
    `error ${isFinite(lastErrMin) ? lastErrMin.toFixed(2) : '–'} · obediencia < ${P.obeyThr} · desvío > ${P.devThr}`,
    `archivo ${(archiveTotal / 600).toFixed(1)} min (${(seeds / 600).toFixed(1)} semilla) · ${records.length} tramos · búsqueda ${searchMs.toFixed(1)} ms${worker ? '' : ' (sin worker)'}`,
    archiveTotal < P.minArchiveSec * 10 ? 'archivo insuficiente: predice por extrapolación' : 'predice desde el archivo',
  ];
  const fs = Math.round(12 * dpr);
  ctx.font = `${fs}px ui-monospace, Menlo, monospace`; ctx.textBaseline = 'top';
  ctx.fillStyle = 'rgba(0,0,0,.6)'; ctx.fillRect(8 * dpr, 8 * dpr, 560 * dpr, (lines.length * 1.5 + 0.8) * fs);
  ctx.fillStyle = '#9c9';
  lines.forEach((l, i) => ctx.fillText(l, 14 * dpr, 8 * dpr + (0.4 + i * 1.5) * fs));
}

function frame(now) {
  frameDt = lastFrame ? Math.min(100, now - lastFrame) : 16; lastFrame = now; rFrames++;
  if (now - lastFpsT >= 1000) {
    rFps = Math.round(rFrames * 1000 / (now - lastFpsT)); dFps = Math.round(dFrames * 1000 / (now - lastFpsT));
    rFrames = dFrames = 0; lastFpsT = now; sweepPending(now); watchdog(now);
    if (!panel.hidden) updateStats();
  }
  if ((state === 'enter' || state === 'play') && now - lastSeen >= P.lostMs) enterPersist(now);
  if (state === 'enter' || state === 'play') {
    if (now - lastPredict >= P.recompute) { lastPredict = now; predict(); }
    evalChecks(now);
  } else if (state === 'persist') {
    if (now >= persistEnd) endSession(); else persistStep(now);
  }
  draw(now);
}
function loop(now) {
  requestAnimationFrame(loop);
  try { frame(now); } catch (e) { console.warn(e); }
}
// Para 8 horas seguidas: si la cámara se corta o se pausa, se recupera sola.
let reopening = false;
function watchdog(now) {
  if (!started || reopening) return;
  const track = stream && stream.getVideoTracks()[0];
  if (!track || track.readyState === 'ended') { reopening = true; openCamera().catch(() => {}).finally(() => (reopening = false)); return; }
  if (video.paused) video.play().catch(() => {});
}

// ───────────────────────── Pantalla ─────────────────────────
let wakeLock = null;
async function requestWake() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {} }
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && started) { requestWake(); if (video.paused) video.play().catch(() => {}); } });
function goFullscreen() {
  const el = document.documentElement;
  const f = el.requestFullscreen || el.webkitRequestFullscreen;
  if (!f || document.fullscreenElement || document.webkitFullscreenElement) return;
  try {
    const r = f.call(el, { navigationUI: 'hide' });
    if (r && r.then) r.then(() => screen.orientation?.lock?.('landscape').catch(() => {})).catch(() => {});
  } catch {}
}

// ───────────────────────── Inicio ─────────────────────────
let starting = false;
async function start() {
  if (started || starting) return;
  starting = true; startErr.textContent = '';
  startEl.querySelector('.hint').textContent = 'cargando…';
  goFullscreen(); // tiene que ir antes de cualquier await: requiere el gesto del usuario
  try {
    if (!window.isSecureContext) throw new Error('La cámara exige HTTPS. Abrí la app desde su dirección https://.');
    await openCamera();
    await loadModel();
    await loadArchive();
  } catch (e) {
    starting = false;
    startEl.querySelector('.hint').textContent = 'tocar para reintentar';
    startErr.textContent = e && e.name === 'NotAllowedError' ? 'Sin permiso para usar la cámara.' : String(e && e.message || e);
    return;
  }
  started = true; starting = false;
  startEl.hidden = true; startEl.style.display = 'none';
  resize(); requestWake(); startDetect(); requestAnimationFrame(loop);
}
startEl.addEventListener('click', start);

// ───────────────────────── Panel oculto (triple toque arriba a la derecha) ─────────────────────────
let taps = [];
$('#hotspot').addEventListener('pointerdown', (e) => {
  e.stopPropagation();
  const t = performance.now(); taps = taps.filter((x) => t - x < 700); taps.push(t);
  if (taps.length >= 3) { taps = []; openPanel(); }
});
function buildForm() {
  const form = $('#pform'); form.innerHTML = '';
  for (const [key, label, spec] of SCHEMA) {
    if (!label) { const h = document.createElement('h3'); h.textContent = key; form.appendChild(h); continue; }
    const row = document.createElement('div'); row.className = 'row';
    const lab = document.createElement('label'); lab.textContent = label; row.appendChild(lab);
    let input;
    if (spec === 'bool') {
      input = document.createElement('input'); input.type = 'checkbox'; input.checked = !!P[key];
      input.addEventListener('change', () => setParam(key, input.checked));
      row.appendChild(input);
    } else if (spec.options) {
      input = document.createElement('select');
      for (const [v, t] of spec.options) { const o = document.createElement('option'); o.value = v; o.textContent = t; input.appendChild(o); }
      input.value = P[key];
      input.addEventListener('change', () => setParam(key, input.value));
      row.appendChild(input);
    } else {
      const [min, max, step] = spec;
      input = document.createElement('input'); input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = P[key];
      const out = document.createElement('output'); out.textContent = P[key];
      input.addEventListener('input', () => { const v = Number(input.value); out.textContent = v; setParam(key, v); });
      row.appendChild(input); row.appendChild(out);
    }
    form.appendChild(row);
  }
}
function setParam(key, v) {
  P[key] = v; saveParams();
  if (key === 'facing' && started) openCamera().catch((e) => alert('No se pudo cambiar de cámara: ' + e.message));
  if (key === 'numPoses' && landmarker) landmarker.setOptions({ numPoses: v }).catch?.(() => {});
  if (key === 'blurScale') sizeOffscreen();
  if (key === 'maxSamples') { enforceCap(); rebuild(); }
  if (key === 'forceLandscape') resize();
  if (key === 'kInit' || key === 'kMin' || key === 'kMax') K = Math.max(P.kMin, Math.min(P.kMax, K));
}
function updateStats() {
  const seeds = records.filter((r) => r.seed), visits = records.filter((r) => !r.seed);
  const min = (rs) => (rs.reduce((a, r) => a + r.n, 0) / 600).toFixed(1);
  $('#pstats').textContent =
    `Archivo: ${min(records)} min en ${records.length} tramos (semilla ${min(seeds)} min · visitas ${min(visits)} min)\n` +
    `Tope: ${(P.maxSamples / 600).toFixed(0)} min · Pose: ${delegate} · Render ${rFps} fps · Pose ${dFps} fps\n` +
    `Estado: ${started ? state : 'sin iniciar'} · ${navigator.serviceWorker?.controller ? 'disponible sin conexión' : 'sin caché offline todavía'}`;
}
function openPanel() { buildForm(); updateStats(); panel.hidden = false; }
let importAsSeed = false;
panel.addEventListener('click', async (e) => {
  const act = e.target.dataset && e.target.dataset.act; if (!act) return;
  if (act === 'close') panel.hidden = true;
  if (act === 'fullscreen') goFullscreen();
  if (act === 'export') exportArchive(false);
  if (act === 'exportSeed') exportArchive(true);
  if (act === 'import' || act === 'importSeed') { importAsSeed = act === 'importSeed'; $('#importFile').click(); }
  if (act === 'clearVisits' && confirm('¿Borrar todas las visitas? La semilla se conserva.')) {
    for (const r of records.filter((r) => !r.seed)) dbDel(r.id);
    records = records.filter((r) => r.seed); rebuild(); updateStats();
  }
  if (act === 'clearAll' && confirm('¿Borrar todo el archivo, incluida la semilla?')) {
    for (const r of records) dbDel(r.id);
    records = []; rebuild(); updateStats();
  }
  if (act === 'reset' && confirm('¿Volver todos los parámetros a sus valores iniciales?')) {
    const facing = P.facing; P = { ...DEFAULTS }; saveParams(); sizeOffscreen(); buildForm();
    if (started && facing !== P.facing) openCamera().catch(() => {});
  }
});
$('#importFile').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0]; e.target.value = '';
  if (!file) return;
  try {
    const recs = parseExport(JSON.parse(await file.text()), importAsSeed);
    addRecords(recs); updateStats();
    alert(`Importados ${recs.length} tramos (${(recs.reduce((a, r) => a + r.n, 0) / 600).toFixed(1)} min).`);
  } catch (err) { alert('No se pudo importar: ' + err.message); }
});

// ───────────────────────── PWA ─────────────────────────
if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
resize();
