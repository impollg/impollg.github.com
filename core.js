// core.js — lógica pura (sin DOM): representación de pose, búsqueda en el archivo y predicción.
// Se usa desde app.js y desde los tests en Node. searchArchive se copia al Web Worker con toString(),
// así que no puede depender de nada fuera de su propio cuerpo.

// 13 puntos de MediaPipe: nariz, hombros, codos, muñecas, caderas, rodillas, tobillos.
export const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const J = { nose: 0, ls: 1, rs: 2, le: 3, re: 4, lw: 5, rw: 6, lh: 7, rh: 8, lk: 9, rk: 10, la: 11, ra: 12 };
export const NJ = 13;      // puntos
export const PD = 26;      // dimensiones de pose (13 × 2)
export const SD = 29;      // dimensiones por muestra: 26 normalizadas + raízX, raízY, escala (px)
export const W = 6;        // ventana de consulta: 0,6 s a 10 Hz
export const F = 15;       // continuación: 1,5 s a 10 Hz
export const DT = 100;     // ms entre muestras
export const QD = W * PD + PD + 2; // dimensiones de la consulta

export const BONES = [
  [J.ls, J.le], [J.le, J.lw], [J.rs, J.re], [J.re, J.rw],
  [J.lh, J.lk], [J.lk, J.la], [J.rh, J.rk], [J.rk, J.ra],
  [J.ls, J.rs], [J.lh, J.rh], [J.ls, J.lh], [J.rs, J.rh],
];

// Raíz = punto medio de caderas; escala = largo del torso (medio hombros → medio caderas), en px.
export function rootScale(p) {
  const hx = (p[2 * J.lh] + p[2 * J.rh]) / 2, hy = (p[2 * J.lh + 1] + p[2 * J.rh + 1]) / 2;
  const sx = (p[2 * J.ls] + p[2 * J.rs]) / 2, sy = (p[2 * J.ls + 1] + p[2 * J.rs + 1]) / 2;
  return [hx, hy, Math.max(1e-3, Math.hypot(sx - hx, sy - hy))];
}

// Pose en px → muestra normalizada. Puntos con visibilidad < 0,5 se toman de la muestra anterior.
export function toSample(p, vis, prev) {
  const [rx, ry, s] = rootScale(p);
  const out = new Float32Array(SD);
  for (let i = 0; i < NJ; i++) {
    const c = 2 * i;
    if (vis && vis[i] < 0.5 && prev) { out[c] = prev[c]; out[c + 1] = prev[c + 1]; }
    else { out[c] = (p[c] - rx) / s; out[c + 1] = (p[c + 1] - ry) / s; }
  }
  out[26] = rx; out[27] = ry; out[28] = s;
  return out;
}

export function samplePixels(s) {
  const p = new Float32Array(PD);
  for (let c = 0; c < PD; c += 2) { p[c] = s[26] + s[c] * s[28]; p[c + 1] = s[27] + s[c + 1] * s[28]; }
  return p;
}

export function pixelsToSample(p) { return toSample(p, null, null); }

export function lerpArr(a, b, t) {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] + (b[i] - a[i]) * t;
  return o;
}

// Error medio por punto, en unidades de torso.
export function poseError(a, b, s) {
  let e = 0;
  for (let c = 0; c < PD; c += 2) e += Math.hypot(a[c] - b[c], a[c + 1] - b[c + 1]);
  return e / NJ / s;
}

// Consulta: 6 poses normalizadas + velocidad de la última pose + velocidad de la raíz (en torsos / muestra).
export function buildQuery(win, velW, rootVelW) {
  const q = new Float32Array(QD);
  for (let w = 0; w < W; w++) q.set(win[w].subarray(0, PD), w * PD);
  const last = win[W - 1], prev = win[W - 2];
  for (let c = 0; c < PD; c++) q[W * PD + c] = (last[c] - prev[c]) * velW;
  const s = last[28] || 1;
  q[QD - 2] = (last[26] - prev[26]) / s * rootVelW;
  q[QD - 1] = (last[27] - prev[27]) / s * rootVelW;
  return q;
}

// Fuerza bruta: mejor ventana por grupo (sesión), K grupos distintos, ordenados por distancia.
// A = { flat: Float32Array, starts: Int32Array, ns: Int32Array, groups: Int32Array }
export function searchArchive(A, q, k, maxDist, velW, rootVelW) {
  const SD = 29, PD = 26, W = 6, F = 15, NJ = 13;
  const flat = A.flat, starts = A.starts, ns = A.ns, groups = A.groups;
  const norm = W * NJ + NJ + 1;
  const best = new Map();
  for (let si = 0; si < starts.length; si++) {
    const st = starts[si], n = ns[si], g = groups[si];
    const eMax = st + n - 1 - F;
    const prior = best.get(g);
    let bd = prior ? prior.d : Infinity, be = -1;
    for (let e = st + W - 1; e <= eMax; e++) {
      let d = 0, qi = 0, bail = false;
      for (let w = 0; w < W; w++) {
        const b = (e - W + 1 + w) * SD;
        for (let c = 0; c < PD; c++) { const x = flat[b + c] - q[qi++]; d += x * x; }
        if (d >= bd) { bail = true; break; }
      }
      if (bail) continue;
      const b1 = e * SD, b0 = b1 - SD;
      for (let c = 0; c < PD; c++) { const x = (flat[b1 + c] - flat[b0 + c]) * velW - q[qi++]; d += x * x; }
      const s = flat[b1 + 28] || 1;
      const vx = (flat[b1 + 26] - flat[b0 + 26]) / s * rootVelW - q[qi++];
      const vy = (flat[b1 + 27] - flat[b0 + 27]) / s * rootVelW - q[qi++];
      d += vx * vx + vy * vy;
      if (d < bd) { bd = d; be = e; }
    }
    if (be >= 0) best.set(g, { d: bd, e: be, g });
  }
  const out = [];
  for (const v of best.values()) {
    const dist = Math.sqrt(v.d / norm);
    if (dist <= maxDist) out.push({ e: v.e, g: v.g, dist });
  }
  out.sort((a, b) => a.dist - b.dist);
  return out.slice(0, k);
}

// Continuación de la ventana que termina en e, reubicada sobre la posición y escala actuales.
// La diferencia de forma inicial con el cuerpo actual se desvanece a lo largo del futuro,
// para que el fantasma salga del cuerpo y se vuelva otro.
export function futureFromArchive(flat, e, cur, curNorm) {
  const b0 = e * SD, r0x = flat[b0 + 26], r0y = flat[b0 + 27], s0 = flat[b0 + 28] || 1;
  const [cx, cy, cs] = cur;
  const frames = [];
  for (let j = 0; j <= F; j++) {
    const b = (e + j) * SD;
    const w = curNorm ? (1 - j / F) ** 2 : 0;
    const rx = cx + (flat[b + 26] - r0x) / s0 * cs;
    const ry = cy + (flat[b + 27] - r0y) / s0 * cs;
    const p = new Float32Array(PD);
    for (let c = 0; c < PD; c++) {
      const n = flat[b + c] + (w ? (curNorm[c] - flat[b0 + c]) * w : 0);
      p[c] = (c & 1 ? ry : rx) + n * cs;
    }
    frames.push(p);
  }
  return frames;
}

// Respaldo: extrapolación de velocidad con amortiguación. La variante 0 es la extrapolación pura;
// las demás varían la ganancia y agregan una deriva suave para que siempre haya futuros visibles.
const GAINS = [1, 0.45, 1.7, -0.5, 1.25, 0.2, 2.2];
const LIMBS = new Set([J.le, J.lw, J.re, J.rw, J.lk, J.la, J.rk, J.ra]);
export function futureExtrap(last, prev, variant, cur, damp = 0.85) {
  const g = GAINS[variant % GAINS.length];
  const [cx, cy, cs] = cur;
  const vel = new Float32Array(PD);
  let rvx = 0, rvy = 0;
  if (prev) {
    for (let c = 0; c < PD; c++) vel[c] = last[c] - prev[c];
    const s = last[28] || 1;
    rvx = (last[26] - prev[26]) / s; rvy = (last[27] - prev[27]) / s;
  }
  const amp = variant === 0 ? 0 : 1;
  const ph = variant * 2.39996; // ángulo áureo: reparte las variantes alrededor del cuerpo
  const driftX = Math.cos(ph) * 0.35 * amp, driftY = Math.sin(ph) * 0.08 * amp;
  const frames = [];
  for (let j = 0; j <= F; j++) {
    const cum = damp * (1 - Math.pow(damp, j)) / (1 - damp);
    const ramp = j / F;
    const rx = cx + (rvx * g * cum + driftX * ramp) * cs;
    const ry = cy + (rvy * g * cum + driftY * ramp) * cs;
    const p = new Float32Array(PD);
    for (let i = 0; i < NJ; i++) {
      const c = 2 * i;
      let nx = last[c] + vel[c] * g * cum, ny = last[c + 1] + vel[c + 1] * g * cum;
      if (amp && LIMBS.has(i)) { const k = i * 1.7 + ph; nx += Math.sin(k) * 0.45 * ramp; ny += Math.cos(k * 1.3) * 0.3 * ramp; }
      p[c] = rx + nx * cs; p[c + 1] = ry + ny * cs;
    }
    frames.push(p);
  }
  return frames;
}
