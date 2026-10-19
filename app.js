// Lo que ibas a hacer — v2. El espectador se ve a sí mismo; sus futuros son su propia imagen,
// deformada hacia movimientos que otros hicieron antes.
import {
  NJ, PD, SD, W, F, DT, MI, NB,
  rootScale, toSample, samplePixels, pixelsToSample, lerpArr, poseError,
  buildQuery, searchArchive, futureFromArchive, futureExtrap, extend, boneTransforms,
} from './core.js?v=21';
import { Renderer } from './render.js?v=21';

const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODELS = {
  lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
};
// Huesos del esqueleto dibujado en la capa de depuración (33 puntos).
const DEBUG_BONES = [[11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]];

// Estilos de imagen: cuánto de la foto (fill), cuánto trazo de boceto (edge), cuánto contorno (rim).
const STYLES = {
  foto:    { fill: 1.0,  edge: 0.35, rim: 0.15, gamma: 1.1, contrast: 1.35 },
  boceto:  { fill: 0.18, edge: 1.4,  rim: 0.45, gamma: 1.0, contrast: 1.2 },
  sombra:  { fill: 0.06, edge: 0.35, rim: 1.1,  gamma: 1.0, contrast: 1.0 },
  negativo:{ fill: 0.0,  edge: 1.8,  rim: 0.2,  gamma: 1.0, contrast: 1.0 },
};

// ───────────────────────── Parámetros ─────────────────────────
const DEFAULTS = {
  facing: 'user', mirror: true, fit: 'contain', forceLandscape: true, model: 'full', delegate: 'auto',
  numPoses: 1, requireLegs: false, smoothCut: 1.6, smoothBeta: 0.02,
  selfStyle: 'foto', selfBright: 0.95, ghostStyle: 'boceto', ghostBright: 1.15,
  ghostMinA: 0.25, ghostMaxA: 0.6, trail: 0.88, blur: 1.6, soft: 0.55, ghostRes: 0.5,
  dissolve: 0.12, grain: 0.035, sigma: 0.28, flashMs: 160, accent: false,
  kInit: 3, kMin: 2, kMax: 7, kCooldown: 600,
  obeyThr: 0.15, devThr: 0.45, horizon: 0.8, recompute: 220, blendMs: 160,
  matchMax: 0.6, velW: 3, rootVelW: 2, minArchiveSec: 120,
  persistMin: 5, persistMax: 8, lostMs: 300, minSessionSec: 3, maxSamples: 30000,
  debug: false, recordSeed: false,
};
const STYLE_OPTS = [['foto', 'Foto'], ['boceto', 'Boceto'], ['sombra', 'Sombra'], ['negativo', 'Trazo']];
const SCHEMA = [
  ['Cámara y pantalla'],
  ['facing', 'Cámara', { options: [['user', 'Frontal'], ['environment', 'Trasera']] }],
  ['mirror', 'Espejo', 'bool'],
  ['fit', 'Encuadre', { options: [['cover', 'Llenar pantalla'], ['contain', 'Mostrar todo el cuadro']] }],
  ['forceLandscape', 'Pedir orientación horizontal', 'bool'],
  ['model', 'Modelo de pose (al reiniciar)', { options: [['full', 'Completo (más preciso)'], ['lite', 'Liviano (más rápido)']] }],
  ['delegate', 'Procesador (al reiniciar)', { options: [['auto', 'Automático'], ['GPU', 'GPU'], ['CPU', 'CPU']] }],
  ['numPoses', 'Personas detectadas (usa la más grande)', [1, 3, 1]],
  ['requireLegs', 'Exigir rodillas visibles (cuerpo entero)', 'bool'],
  ['smoothCut', 'Suavizado de puntos (menos = más suave)', [0.2, 6, 0.1]],
  ['smoothBeta', 'Respuesta a movimientos rápidos', [0, 0.1, 0.002]],
  ['Imagen'],
  ['selfStyle', 'Espectador', { options: STYLE_OPTS }],
  ['selfBright', 'Brillo del espectador', [0, 2, 0.05]],
  ['ghostStyle', 'Fantasmas', { options: STYLE_OPTS }],
  ['ghostBright', 'Brillo de los fantasmas', [0, 3, 0.05]],
  ['ghostMinA', 'Opacidad mínima fantasma', [0, 1, 0.01]],
  ['ghostMaxA', 'Opacidad máxima fantasma', [0, 1, 0.01]],
  ['trail', 'Estela (permanencia por cuadro)', [0, 0.98, 0.01]],
  ['blur', 'Desenfoque', [0, 6, 0.1]],
  ['soft', 'Mezcla nítido / desenfocado', [0, 1, 0.05]],
  ['ghostRes', 'Resolución de fantasmas', [0.25, 1, 0.05]],
  ['dissolve', 'Polvo (disolución)', [0, 0.9, 0.05]],
  ['grain', 'Grano', [0, 0.15, 0.005]],
  ['sigma', 'Rigidez de la deformación', [0.1, 0.8, 0.01]],
  ['flashMs', 'Destello de fusión (ms)', [0, 600, 10]],
  ['accent', 'Acento rojo en el fantasma más probable', 'bool'],
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
  ['debug', 'Capa de depuración', 'bool'],
];
const LS_KEY = 'lqiah2.params';
let P = loadParams();
function loadParams() { try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(LS_KEY) || '{}') }; } catch { return { ...DEFAULTS }; } }
function saveParams() { try { localStorage.setItem(LS_KEY, JSON.stringify(P)); } catch {} }

// ───────────────────────── DOM ─────────────────────────
const $ = (s) => document.querySelector(s);
// Si el index.html es de una versión anterior y no trae el lienzo de interfaz, se crea acá.
const ui = $('#ui') || (() => {
  const c = document.createElement('canvas'); c.id = 'ui';
  c.style.cssText = 'position:fixed;left:0;top:0;display:block;pointer-events:none;z-index:2';
  $('#stage').after(c); return c;
})();
const video = $('#cam'), canvas = $('#stage'), uctx = ui.getContext('2d');
const startEl = $('#start'), startErr = $('#startErr'), rotateEl = $('#rotate'), panel = $('#panel');
let renderer = null, maskU8 = null;
let VW = 640, VH = 480, dpr = 1;

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  for (const c of [canvas, ui]) {
    c.style.width = innerWidth + 'px'; c.style.height = innerHeight + 'px';
    c.width = Math.round(innerWidth * dpr); c.height = Math.round(innerHeight * dpr);
  }
  if (renderer) renderer.resize(canvas.width, canvas.height, P.ghostRes);
  rotateEl.style.display = started && P.forceLandscape && innerHeight > innerWidth ? 'flex' : 'none';
}
addEventListener('resize', resize);
// Cuadro de video → pantalla (cover/contain, espejo). Devuelve [kx, ky, bx, by] en espacio de recorte y la escala en px.
function viewParams() {
  const cw = canvas.width, ch = canvas.height;
  const sc = P.fit === 'contain' ? Math.min(cw / VW, ch / VH) : Math.max(cw / VW, ch / VH);
  const ox = (cw - VW * sc) / 2, oy = (ch - VH * sc) / 2;
  const kx = (P.mirror ? -1 : 1) * sc * 2 / cw, bx = P.mirror ? (cw - ox) * 2 / cw - 1 : ox * 2 / cw - 1;
  return { view: [kx, -sc * 2 / ch, bx, 1 - oy * 2 / ch], sc, ox, oy, cw, ch };
}
function toScreen(v, x, y) { return [P.mirror ? v.cw - v.ox - x * v.sc : v.ox + x * v.sc, v.oy + y * v.sc]; }

// ───────────────────────── Estado ─────────────────────────
let started = false, state = 'idle';
let stream = null, landmarker = null, delegate = '-';
let session = null, seg = [], segT = [], lastDet = null, nextGrid = 0;
let curPose = null, curRoot = null, curPoseT = -1e9, lastSeen = -1e9, partialSeen = -1e9;
let texPose = null, texVis = null; // pose (extendida) y visibilidad del cuadro que está en la textura
let K = P.kInit, lastKChange = 0, lastPredict = 0, checks = [];
let ghosts = [], gid = 0;
let persistStart = 0, persistEnd = 0;
let lastErrMin = NaN, searchMs = 0;
let frameDt = 16, lastFrame = 0, rFrames = 0, dFrames = 0, rFps = 0, dFps = 0, lastFpsT = 0;

// Filtro One-Euro: quita el temblor de los puntos sin agregar retraso a los movimientos rápidos.
class OneEuro {
  constructor() { this.x = null; this.dx = null; this.t = 0; }
  reset() { this.x = null; }
  filter(v, t) {
    if (!this.x) { this.x = Float32Array.from(v); this.dx = new Float32Array(v.length); this.t = t; return Float32Array.from(v); }
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const al = (c) => { const r = 2 * Math.PI * c * dt; return r / (r + 1); };
    const ad = al(1), out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) {
      this.dx[i] += ad * ((v[i] - this.x[i]) / dt - this.dx[i]);
      this.x[i] += al(P.smoothCut + P.smoothBeta * Math.abs(this.dx[i])) * (v[i] - this.x[i]);
      out[i] = this.x[i];
    }
    return out;
  }
}
const euro = new OneEuro();

// ───────────────────────── Archivo (IndexedDB) ─────────────────────────
let db = null, records = [], flat = new Float32Array(0), archiveTotal = 0, archiveVer = 0, A = null;
function openDB() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('lo-que-ibas-a-hacer-2', 1);
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
  records = records.filter((r) => r && r.data && r.n >= W + F && r.data.length === r.n * SD);
  if (!records.some((r) => r.seed)) {
    try { const res = await fetch('seed.json', { cache: 'no-cache' }); if (res.ok) addRecords(parseExport(await res.json(), true)); } catch {}
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
  while (total > P.maxSamples && visits.length) { const r = visits.shift(); total -= r.n; records.splice(records.indexOf(r), 1); dbDel(r.id); }
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
  if (obj.sd && obj.sd !== SD) throw new Error('El archivo es de otra versión de la obra (formato de ' + obj.sd + ' valores).');
  return obj.sessions
    .filter((s) => s && s.n >= W + F && Array.isArray(s.data) && s.data.length === s.n * SD)
    .map((s) => ({ id: String(s.id || Math.random().toString(36).slice(2)), group: String(s.group || s.id), seed: forceSeed ? true : !!s.seed, created: s.created || Date.now(), n: s.n, data: Float32Array.from(s.data) }));
}
function exportArchive(onlySeed) {
  const list = records.filter((r) => !onlySeed || r.seed);
  const out = { format: 'lo-que-ibas-a-hacer', version: 2, sd: SD, hz: 10, joints: 33, exported: new Date().toISOString(),
    sessions: list.map((r) => ({ id: r.id, group: r.group, seed: !!r.seed, created: r.created, n: r.n, data: Array.from(r.data, (x) => Math.round(x * 1e4) / 1e4) })) };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], { type: 'application/json' }));
  a.download = onlySeed ? 'seed.json' : `archivo-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ───────────────────────── Búsqueda (Web Worker) ─────────────────────────
let worker = null, qid = 0, pendingLive = 0;
const pending = new Map();
try {
  const src = `${searchArchive.toString()}
const MI = new Int32Array([${Array.from(MI).join(',')}]);
let A = null, ver = 0;
onmessage = (e) => {
  const m = e.data;
  if (m.type === 'archive') { A = m.A; ver = m.ver; return; }
  const t = performance.now();
  const r = A ? searchArchive(A, m.q, m.k, m.maxDist, m.velW, m.rootVelW, MI) : [];
  postMessage({ id: m.id, ver, r, ms: performance.now() - t });
};`;
  worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  worker.onmessage = (e) => onSearch(e.data);
  worker.onerror = () => { worker = null; };
} catch { worker = null; }
function query(q, k, c) {
  const id = ++qid; c.sent = performance.now(); pending.set(id, c);
  if (c.mode === 'live') pendingLive = id;
  if (worker) worker.postMessage({ type: 'query', id, q, k, maxDist: P.matchMax, velW: P.velW, rootVelW: P.rootVelW });
  else setTimeout(() => {
    const t = performance.now();
    const r = A ? searchArchive(A, q, k, P.matchMax, P.velW, P.rootVelW, MI) : [];
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
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: P.facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } });
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => (video.onloadedmetadata = r));
  updateVideoSize();
}
function updateVideoSize() {
  if (video.videoWidth && (video.videoWidth !== VW || video.videoHeight !== VH)) { VW = video.videoWidth; VH = video.videoHeight; }
}
function withTimeout(p, ms) { return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('tardó demasiado')), ms))]); }
function status(t) { const h = startEl.querySelector('.hint'); if (h) h.textContent = t; }
async function loadModel() {
  let mp;
  try { mp = await import(MP_URL); } catch { throw new Error('No se pudo descargar MediaPipe. Revisá la conexión a internet y volvé a intentar.'); }
  const { FilesetResolver, PoseLandmarker } = mp;
  const fs = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (d) => ({
    baseOptions: { modelAssetPath: MODELS[P.model] || MODELS.full, delegate: d }, runningMode: 'VIDEO',
    numPoses: P.numPoses, outputSegmentationMasks: true,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  mpMake = (d) => withTimeout(PoseLandmarker.createFromOptions(fs, opts(d)), d === 'GPU' ? 25000 : 35000);
  if (P.delegate !== 'CPU') {
    try { landmarker = await mpMake('GPU'); delegate = 'GPU'; return; } catch {}
    status('cargando modelo (modo CPU)…');
  }
  try { landmarker = await mpMake('CPU'); delegate = 'CPU'; }
  catch (e) { throw new Error('No se pudo cargar el modelo de pose: ' + (e && e.message || e)); }
}
// Algunos navegadores no pueden leer la máscara desde la GPU (sale vacía). Si pasa, se cambia a CPU solo.
let mpMake = null, emptyMasks = 0, switching = false;
async function fallbackToCPU() {
  if (switching || !mpMake || delegate !== 'GPU' || P.delegate === 'GPU') return;
  switching = true;
  try { const lm = await mpMake('CPU'); const old = landmarker; landmarker = lm; delegate = 'CPU'; try { old.close(); } catch {} }
  catch (e) { console.warn(e); }
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
  landmarker.detectForVideo(video, ts, (res) => onResult(res, ts));
  dFrames++;
}
function onResult(res, t) {
  const L = res.landmarks; if (!L || !L.length) return;
  let idx = 0;
  if (L.length > 1) {
    let best = -1;
    L.forEach((lm, i) => {
      let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
      for (const l of lm) { x0 = Math.min(x0, l.x); x1 = Math.max(x1, l.x); y0 = Math.min(y0, l.y); y1 = Math.max(y1, l.y); }
      const a = (x1 - x0) * (y1 - y0); if (a > best) { best = a; idx = i; }
    });
  }
  const lm = L[idx];
  const raw = new Float32Array(PD), v = new Float32Array(NJ);
  for (let i = 0; i < NJ; i++) { raw[2 * i] = lm[i].x * VW; raw[2 * i + 1] = lm[i].y * VH; v[i] = lm[i].visibility ?? 1; }
  const need = [11, 12, 23, 24]; if (P.requireLegs) need.push(25, 26);
  if (!need.every((i) => v[i] > 0.5)) { partialSeen = t; return; }
  const m = res.segmentationMasks && res.segmentationMasks[idx];
  if (!m) return;
  // Máscara a 8 bits y subida junto con el cuadro de cámara correspondiente.
  const f = m.getAsFloat32Array();
  if (!maskU8 || maskU8.length !== f.length) maskU8 = new Uint8Array(f.length);
  let mmax = 0;
  for (let i = 0; i < f.length; i++) { const x = f[i] * 255; maskU8[i] = x; if (x > mmax) mmax = x; }
  if (mmax < 8) { if (++emptyMasks > 12) fallbackToCPU(); return; }
  emptyMasks = 0;
  if (state === 'idle') { euro.reset(); beginSession(); }
  else if (state === 'persist') resumeSession();
  const p = euro.filter(raw, t);
  renderer.upload(video, maskU8, m.width, m.height);
  lastSeen = t;
  onBody(t, p, v);
  // La pose de la textura usa los puntos sin suavizar: tiene que calzar exacto con la imagen del cuadro.
  const rv = new Float32Array(PD);
  for (let i = 0; i < NJ; i++) { const ok = v[i] >= 0.5; rv[2 * i] = ok ? raw[2 * i] : curPose[2 * i]; rv[2 * i + 1] = ok ? raw[2 * i + 1] : curPose[2 * i + 1]; }
  texPose = extend(rv); texVis = v;
}

// ───────────────────────── Sesión y remuestreo a 10 Hz ─────────────────────────
function onBody(t, p, v) {
  const s = toSample(p, v, lastDet ? lastDet.s : null);
  curPose = samplePixels(s); curRoot = [s[PD], s[PD + 1], s[PD + 2]]; curPoseT = t;
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
  const c = { mode: 'live', t0: segT[segT.length - 1], cur: [last[PD], last[PD + 1], last[PD + 2]], last, prev, k: K };
  if (seg.length >= W && archiveTotal >= P.minArchiveSec * 10) query(buildQuery(seg.slice(-W), P.velW, P.rootVelW), K, c);
  else applyLive([], c, performance.now());
}
function applyLive(r, c, now) {
  if (state !== 'enter' && state !== 'play') return;
  const futs = [];
  for (const m of r) futs.push({ key: 'g' + m.g, frames: futureFromArchive(flat, m.e, c.cur, c.last), sim: 1 - m.dist / P.matchMax });
  for (let v = 0; futs.length < c.k; v++) futs.push({ key: 'x' + v, frames: futureExtrap(c.last, c.prev, v, c.cur), sim: 0.45 });
  futs.length = Math.min(futs.length, c.k);
  assignGhosts(futs, c.t0, now);
  const hi = Math.max(1, Math.min(F, Math.round(P.horizon * 10)));
  checks.push({ due: c.t0 + hi * DT, items: futs.map((f) => ({ key: f.key, pose: f.frames[hi], moved: poseError(f.frames[hi], f.frames[0], c.cur[2]) })) });
  if (checks.length > 40) checks.splice(0, checks.length - 40);
}
function assignGhosts(futs, t0, now) {
  const live = ghosts.filter((g) => g.state === 'live');
  const used = new Set(), keys = new Set(futs.map((f) => f.key));
  for (const f of futs) {
    let g = live.find((x) => !used.has(x) && x.key === f.key) || live.find((x) => !used.has(x) && !keys.has(x.key));
    if (g) { g.from = g.cur; g.blendStart = now; g.frames = f.frames; g.t0 = t0; g.offset = P.horizon * 1000; g.key = f.key; g.sim = f.sim; }
    else { g = { id: ++gid, key: f.key, frames: f.frames, t0, offset: P.horizon * 1000, sim: f.sim, alpha: 0, state: 'live', from: null, blendStart: 0, cur: null, requery: false }; ghosts.push(g); }
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
      if (it.moved > P.obeyThr && e < minE) { minE = e; minKey = it.key; }
    }
    lastErrMin = minE;
    if (now - lastKChange < P.kCooldown) continue;
    if (minE < P.obeyThr) {
      const g = ghosts.find((x) => x.state === 'live' && x.key === minKey);
      if (g) { g.state = 'fusing'; g.fuseStart = now; K = Math.max(P.kMin, K - 1); lastKChange = now; }
    } else if (allDev) { K = Math.min(P.kMax, K + 1); lastKChange = now; }
  }
}
function persistStep(now) {
  for (const g of ghosts) {
    if (g.state !== 'live' || g.requery) continue;
    if ((now - g.t0 + g.offset) / DT < F) continue;
    const win = []; for (let j = F - W + 1; j <= F; j++) win.push(pixelsToSample(g.frames[j]));
    const last = win[W - 1], prev = win[W - 2];
    const c = { mode: 'ghost', ghost: g, last, prev, cur: [last[PD], last[PD + 1], last[PD + 2]] };
    g.requery = true;
    if (archiveTotal >= P.minArchiveSec * 10) query(buildQuery(win, P.velW, P.rootVelW), 1, c);
    else applyGhost([], c, now);
  }
}
function applyGhost(r, c, now) {
  const g = c.ghost; g.requery = false;
  if (state !== 'persist' || !ghosts.includes(g)) return;
  g.frames = r.length ? futureFromArchive(flat, r[0].e, c.cur, c.last) : futureExtrap(c.last, c.prev, 1 + (g.id % 6), c.cur, 0.8);
  g.from = g.cur; g.blendStart = now; g.t0 = now; g.offset = 0;
}

// ───────────────────────── Fantasmas: pose y opacidad ─────────────────────────
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
    if (curPose && g.cur) g.cur = lerpArr(g.cur, curPose, Math.min(1, frameDt / 110));
    if (el < P.flashMs) return 1;
    const f = 1 - (el - P.flashMs) / 320;
    if (f <= 0) { g.dead = true; return 0; }
    return f;
  }
  g.cur = ghostPose(g, now);
  const target = P.ghostMinA + (P.ghostMaxA - P.ghostMinA) * Math.max(0, Math.min(1, g.sim));
  if (g.state === 'dying') { g.alpha -= frameDt / 220 * Math.max(0.05, target); if (g.alpha <= 0) { g.dead = true; return 0; } }
  else if (g.alpha < target) g.alpha = Math.min(target, g.alpha + frameDt / 300 * target);
  else g.alpha = Math.max(target, g.alpha - frameDt / 400 * target);
  return g.alpha;
}
// Pose destino del fantasma, con dos resguardos: los puntos que la cámara no ve (fuera de cuadro o
// tapados) siguen al torso en vez de inventar movimiento, y ningún punto se aleja más de 2,5 torsos.
function ghostTarget(pose) {
  const t = extend(pose), r = texPose;
  const ax = r[68], ay = r[69], bx = r[66], by = r[67], cx = t[68], cy = t[69], dx = t[66], dy = t[67];
  const v0x = bx - ax, v0y = by - ay, v1x = dx - cx, v1y = dy - cy, l0 = v0x * v0x + v0y * v0y || 1;
  let c = (v0x * v1x + v0y * v1y) / l0, s = (v0x * v1y - v0y * v1x) / l0;
  const sc = Math.hypot(c, s) || 1, cl = Math.min(1.4, Math.max(0.7, sc)); c *= cl / sc; s *= cl / sc;
  const torso = Math.sqrt(l0), lim = 1.6 * torso;
  // El cuerpo entero no se desplaza más de 2 torsos del lugar real.
  const mx = cx - ax, my = cy - ay, md = Math.hypot(mx, my), mlim = 2 * torso;
  if (md > mlim) { const k = mlim / md - 1; for (let j = 0; j < 35; j++) { t[2 * j] += mx * k; t[2 * j + 1] += my * k; } }
  const cx2 = t[68], cy2 = t[69];
  for (let j = 0; j < 35; j++) {
    const rx = r[2 * j] - ax, ry = r[2 * j + 1] - ay;
    const fx = cx2 + c * rx - s * ry, fy = cy2 + s * rx + c * ry; // el punto llevado por el torso
    const hidden = j < NJ && texVis && texVis[j] < 0.5;
    if (hidden) { t[2 * j] = fx; t[2 * j + 1] = fy; continue; }
    const ox = t[2 * j] - fx, oy = t[2 * j + 1] - fy, d = Math.hypot(ox, oy);
    if (d > lim) { t[2 * j] = fx + ox * lim / d; t[2 * j + 1] = fy + oy * lim / d; }
  }
  return t;
}
function boxFor(pose) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let c = 0; c < PD; c += 2) { if (texVis && texVis[c / 2] < 0.5) continue; x0 = Math.min(x0, pose[c]); x1 = Math.max(x1, pose[c]); y0 = Math.min(y0, pose[c + 1]); y1 = Math.max(y1, pose[c + 1]); }
  const m = rootScale(pose)[2] * 0.85;
  if (!isFinite(x0)) return [0, 0, VW, VH];
  return [Math.max(0, x0 - m), Math.max(0, y0 - m * 1.1), Math.min(VW, x1 + m), Math.min(VH, y1 + m)];
}

// ───────────────────────── Render ─────────────────────────
const WHITE = [0.93, 0.95, 1.0], RED = [0.85, 0.32, 0.28];
function styleOf(name, bright, dissolve) {
  const s = STYLES[name] || STYLES.foto;
  return { fill: s.fill * bright, edge: s.edge * bright, rim: s.rim * bright, gamma: s.gamma, contrast: s.contrast, grain: P.grain, dissolve };
}
function draw(now) {
  const v = viewParams();
  const fade = state === 'persist' ? Math.max(0, 1 - (now - persistStart) / Math.max(1, persistEnd - persistStart)) : 1;
  let top = null;
  if (P.accent) for (const g of ghosts) if (g.state === 'live' && (!top || g.sim > top.sim)) top = g;
  const list = [];
  const restBox = texPose ? boxFor(texPose) : null;
  for (let i = ghosts.length - 1; i >= 0; i--) {
    const g = ghosts[i];
    const a = updateGhost(g, now);
    if (g.dead) { ghosts.splice(i, 1); continue; }
    if (!g.cur || !texPose || a * fade <= 0.004) continue;
    if (!g.tf) g.tf = { bone: new Float32Array(4 * NB), mx: new Float32Array(3 * NB), my: new Float32Array(3 * NB) };
    boneTransforms(texPose, ghostTarget(g.cur), g.tf);
    const fusing = g.state === 'fusing' && now - g.fuseStart < P.flashMs;
    list.push({ box: restBox, bones: g.tf, alpha: a * fade, flash: fusing ? 0.6 : 0, tint: g === top ? RED : WHITE });
  }
  const present = (state === 'enter' || state === 'play') && now - lastSeen < P.lostMs;
  renderer.render({
    view: v.view, video: [VW, VH], time: now / 1000, sigma: (texPose ? rootScale(texPose)[2] : 100) * P.sigma,
    trail: Math.pow(P.trail, frameDt / 16.7), blur: P.blur, soft: P.soft, gain: 1, grain: P.grain,
    ghosts: list, ghostStyle: styleOf(P.ghostStyle, P.ghostBright, P.dissolve),
    self: present ? { alpha: 1, tint: WHITE } : null, selfStyle: styleOf(P.selfStyle, P.selfBright, 0),
  });
  drawUI(now, v);
}
function drawUI(now, v) {
  uctx.setTransform(1, 0, 0, 1, 0, 0); uctx.clearRect(0, 0, ui.width, ui.height);
  if (state === 'idle') {
    uctx.font = `${Math.round(13 * dpr)}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    uctx.textBaseline = 'alphabetic'; uctx.fillStyle = '#555';
    uctx.fillText('Lo que ibas a hacer', 18 * dpr, ui.height - 18 * dpr);
    if (now - partialSeen < 600) {
      uctx.fillStyle = '#3a3a3a'; uctx.textAlign = 'right';
      uctx.fillText(P.requireLegs ? 'alejate: tiene que verse el cuerpo entero' : 'alejate un poco', ui.width - 18 * dpr, ui.height - 18 * dpr);
      uctx.textAlign = 'left';
    }
  }
  if (!P.debug) return;
  uctx.lineWidth = 1.5 * dpr;
  const line = (p, col) => {
    uctx.strokeStyle = col; uctx.beginPath();
    for (const [a, b] of DEBUG_BONES) { const [x0, y0] = toScreen(v, p[2 * a], p[2 * a + 1]), [x1, y1] = toScreen(v, p[2 * b], p[2 * b + 1]); uctx.moveTo(x0, y0); uctx.lineTo(x1, y1); }
    uctx.stroke();
  };
  if (curPose && now - curPoseT < 500) line(curPose, '#e44');
  for (const g of ghosts) if (g.cur) line(g.cur, 'rgba(80,220,160,.6)');
  const seeds = records.filter((r) => r.seed).reduce((a, r) => a + r.n, 0);
  const lines = [
    `estado ${state} · K ${K} · fantasmas ${ghosts.length}`,
    `render ${rFps} fps · pose ${dFps} fps · ${delegate} · modelo ${P.model} · video ${VW}×${VH}`,
    `error ${isFinite(lastErrMin) ? lastErrMin.toFixed(2) : '–'} · obediencia < ${P.obeyThr} · desvío > ${P.devThr}`,
    `archivo ${(archiveTotal / 600).toFixed(1)} min (${(seeds / 600).toFixed(1)} semilla) · ${records.length} tramos · búsqueda ${searchMs.toFixed(1)} ms`,
    archiveTotal < P.minArchiveSec * 10 ? 'archivo insuficiente: predice por extrapolación' : 'predice desde el archivo',
  ];
  const fs = Math.round(12 * dpr);
  uctx.font = `${fs}px ui-monospace, Menlo, monospace`; uctx.textBaseline = 'top';
  uctx.fillStyle = 'rgba(0,0,0,.6)'; uctx.fillRect(8 * dpr, 8 * dpr, 600 * dpr, (lines.length * 1.5 + 0.8) * fs);
  uctx.fillStyle = '#9c9';
  lines.forEach((l, i) => uctx.fillText(l, 14 * dpr, 8 * dpr + (0.4 + i * 1.5) * fs));
}

function frame(now) {
  frameDt = lastFrame ? Math.min(100, now - lastFrame) : 16; lastFrame = now; rFrames++;
  if (now - lastFpsT >= 1000) {
    rFps = Math.round(rFrames * 1000 / (now - lastFpsT)); dFps = Math.round(dFrames * 1000 / (now - lastFpsT));
    rFrames = dFrames = 0; lastFpsT = now; sweepPending(now); watchdog();
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
function loop(now) { requestAnimationFrame(loop); try { frame(now); } catch (e) { console.warn(e); } }
let reopening = false;
function watchdog() {
  if (!started || reopening) return;
  const track = stream && stream.getVideoTracks()[0];
  if (!track || track.readyState === 'ended') { reopening = true; openCamera().catch(() => {}).finally(() => (reopening = false)); return; }
  if (video.paused) video.play().catch(() => {});
}

// ───────────────────────── Pantalla ─────────────────────────
async function requestWake() { try { await navigator.wakeLock?.request('screen'); } catch {} }
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && started) { requestWake(); if (video.paused) video.play().catch(() => {}); } });
function goFullscreen() {
  const el = document.documentElement, f = el.requestFullscreen || el.webkitRequestFullscreen;
  if (!f || document.fullscreenElement || document.webkitFullscreenElement) return;
  try { const r = f.call(el, { navigationUI: 'hide' }); if (r && r.then) r.then(() => screen.orientation?.lock?.('landscape').catch(() => {})).catch(() => {}); } catch {}
}
canvas.addEventListener('webglcontextlost', (e) => e.preventDefault());
canvas.addEventListener('webglcontextrestored', () => { try { renderer = new Renderer(canvas); resize(); } catch (e) { console.warn(e); } });

// ───────────────────────── Inicio ─────────────────────────
let starting = false;
async function start() {
  if (started || starting) return;
  starting = true; startErr.textContent = '';
  status('abriendo cámara…');
  goFullscreen();
  try {
    if (!window.isSecureContext) throw new Error('La cámara exige HTTPS. Abrí la app desde su dirección https://.');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw new Error('Este navegador no da acceso a la cámara. Abrí la dirección en Safari (iPhone) o Chrome (Android), no dentro de otra app.');
    if (!renderer) renderer = new Renderer(canvas);
    await openCamera();
    status('cargando modelo…');
    await loadModel();
    status('cargando archivo…');
    await loadArchive();
  } catch (e) {
    starting = false;
    status('tocar para reintentar');
    const n = e && e.name;
    startErr.textContent =
      n === 'NotAllowedError' ? 'Sin permiso para usar la cámara. Habilitalo en los ajustes del navegador para este sitio y volvé a tocar.' :
      n === 'NotFoundError' || n === 'OverconstrainedError' ? 'No se encontró la cámara pedida. Probá cambiar a la otra cámara en el panel.' :
      n === 'NotReadableError' ? 'La cámara está siendo usada por otra app. Cerrala y volvé a tocar.' :
      String(e && e.message || e);
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
      input.addEventListener('change', () => setParam(key, input.checked)); row.appendChild(input);
    } else if (spec.options) {
      input = document.createElement('select');
      for (const [v, t] of spec.options) { const o = document.createElement('option'); o.value = v; o.textContent = t; input.appendChild(o); }
      input.value = P[key]; input.addEventListener('change', () => setParam(key, input.value)); row.appendChild(input);
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
  if (key === 'ghostRes' || key === 'forceLandscape') resize();
  if (key === 'maxSamples') { enforceCap(); rebuild(); }
  if (key === 'kInit' || key === 'kMin' || key === 'kMax') K = Math.max(P.kMin, Math.min(P.kMax, K));
  if (key === 'model' && started) alert('El modelo nuevo se usa al recargar la página.');
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
panel.addEventListener('click', (e) => {
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
  if (act === 'clearAll' && confirm('¿Borrar todo el archivo, incluida la semilla?')) { for (const r of records) dbDel(r.id); records = []; rebuild(); updateStats(); }
  if (act === 'reset' && confirm('¿Volver todos los parámetros a sus valores iniciales?')) {
    const facing = P.facing; P = { ...DEFAULTS }; saveParams(); resize(); buildForm();
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

if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
resize();
window.__lqiahReady = true;
window.__lqiahDebug = () => ({ maskMean: maskU8 ? Math.round(maskU8.reduce((a, b) => a + b, 0) / maskU8.length) : -1, maskMax: maskU8 ? maskU8.reduce((a, b) => (b > a ? b : a), 0) : -1, glErr: renderer ? renderer.gl.getError() : -1, state, K, ghosts: ghosts.length, rFps, dFps, delegate, partialSeen: Math.round(partialSeen), lastSeen: Math.round(lastSeen), now: Math.round(performance.now()) });
