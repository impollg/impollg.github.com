// core.js — lógica pura (sin DOM): pose de 33 puntos, búsqueda en el archivo y predicción.
// searchArchive se copia al Web Worker con toString(): no puede depender de nada fuera de su cuerpo.

export const NJ = 33;      // los 33 puntos de MediaPipe (cara, manos, pies incluidos)
export const PD = 66;      // dimensiones de pose
export const SD = 69;      // por muestra: 66 normalizadas + raízX, raízY, escala (px)
export const W = 6;        // ventana de consulta: 0,6 s a 10 Hz
export const F = 15;       // continuación: 1,5 s a 10 Hz
export const DT = 100;     // ms entre muestras

// Puntos que pesan en la comparación de movimiento (la cara tiembla y aporta poco).
export const MATCH = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32];
export const MI = Int32Array.from(MATCH.flatMap((j) => [2 * j, 2 * j + 1]));
export const MQ = MI.length;
export const QD = W * MQ + MQ + 2;

// Raíz = punto medio de caderas; escala = largo del torso, en px.
export function rootScale(p) {
  const hx = (p[46] + p[48]) / 2, hy = (p[47] + p[49]) / 2;
  const sx = (p[22] + p[24]) / 2, sy = (p[23] + p[25]) / 2;
  return [hx, hy, Math.max(1e-3, Math.hypot(sx - hx, sy - hy))];
}

export function toSample(p, vis, prev) {
  const [rx, ry, s] = rootScale(p);
  const out = new Float32Array(SD);
  for (let i = 0; i < NJ; i++) {
    const c = 2 * i;
    if (vis && vis[i] < 0.5 && prev) { out[c] = prev[c]; out[c + 1] = prev[c + 1]; }
    else { out[c] = (p[c] - rx) / s; out[c + 1] = (p[c + 1] - ry) / s; }
  }
  out[PD] = rx; out[PD + 1] = ry; out[PD + 2] = s;
  return out;
}

export function samplePixels(s) {
  const p = new Float32Array(PD), rx = s[PD], ry = s[PD + 1], sc = s[PD + 2];
  for (let c = 0; c < PD; c += 2) { p[c] = rx + s[c] * sc; p[c + 1] = ry + s[c + 1] * sc; }
  return p;
}
export function pixelsToSample(p) { return toSample(p, null, null); }

export function lerpArr(a, b, t) {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] + (b[i] - a[i]) * t;
  return o;
}

// Error medio por punto (sobre los puntos de comparación), en torsos.
export function poseError(a, b, s) {
  let e = 0;
  for (const j of MATCH) e += Math.hypot(a[2 * j] - b[2 * j], a[2 * j + 1] - b[2 * j + 1]);
  return e / MATCH.length / s;
}

export function buildQuery(win, velW, rootVelW) {
  const q = new Float32Array(QD);
  let k = 0;
  for (let w = 0; w < W; w++) for (let i = 0; i < MQ; i++) q[k++] = win[w][MI[i]];
  const last = win[W - 1], prev = win[W - 2];
  for (let i = 0; i < MQ; i++) q[k++] = (last[MI[i]] - prev[MI[i]]) * velW;
  const s = last[PD + 2] || 1;
  q[k++] = (last[PD] - prev[PD]) / s * rootVelW;
  q[k++] = (last[PD + 1] - prev[PD + 1]) / s * rootVelW;
  return q;
}

// Fuerza bruta: mejor ventana por grupo (sesión); K grupos distintos ordenados por distancia.
export function searchArchive(A, q, k, maxDist, velW, rootVelW, mi) {
  const SD = 69, PD = 66, W = 6, F = 15, MQ = mi.length;
  const flat = A.flat, starts = A.starts, ns = A.ns, groups = A.groups;
  const norm = (W * MQ + MQ) / 2 + 1;
  const best = new Map();
  for (let si = 0; si < starts.length; si++) {
    const st = starts[si], n = ns[si], g = groups[si];
    const prior = best.get(g);
    let bd = prior ? prior.d : Infinity, be = -1;
    for (let e = st + W - 1; e <= st + n - 1 - F; e++) {
      let d = 0, qi = 0, bail = false;
      for (let w = 0; w < W; w++) {
        const b = (e - W + 1 + w) * SD;
        for (let i = 0; i < MQ; i++) { const x = flat[b + mi[i]] - q[qi++]; d += x * x; }
        if (d >= bd) { bail = true; break; }
      }
      if (bail) continue;
      const b1 = e * SD, b0 = b1 - SD;
      for (let i = 0; i < MQ; i++) { const x = (flat[b1 + mi[i]] - flat[b0 + mi[i]]) * velW - q[qi++]; d += x * x; }
      const s = flat[b1 + PD + 2] || 1;
      const vx = (flat[b1 + PD] - flat[b0 + PD]) / s * rootVelW - q[qi++];
      const vy = (flat[b1 + PD + 1] - flat[b0 + PD + 1]) / s * rootVelW - q[qi++];
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

// Continuación del archivo reubicada sobre el cuerpo actual; la diferencia inicial de forma se desvanece.
export function futureFromArchive(flat, e, cur, curNorm) {
  const b0 = e * SD, r0x = flat[b0 + PD], r0y = flat[b0 + PD + 1], s0 = flat[b0 + PD + 2] || 1;
  const [cx, cy, cs] = cur;
  const frames = [];
  for (let j = 0; j <= F; j++) {
    const b = (e + j) * SD;
    const w = curNorm ? (1 - j / F) ** 2 : 0;
    const rx = cx + (flat[b + PD] - r0x) / s0 * cs, ry = cy + (flat[b + PD + 1] - r0y) / s0 * cs;
    const p = new Float32Array(PD);
    for (let c = 0; c < PD; c++) {
      const n = flat[b + c] + (w ? (curNorm[c] - flat[b0 + c]) * w : 0);
      p[c] = (c & 1 ? ry : rx) + n * cs;
    }
    frames.push(p);
  }
  return frames;
}

// Respaldo: extrapolación con amortiguación. Las variantes rotan brazos, piernas y torso como cadenas
// articuladas (hombro → codo → muñeca → dedos), así los futuros siguen siendo cuerpos posibles.
const GAINS = [1, 0.6, 1.4, 0.3, 1.2, 0.8, 1.6];
const CHAIN_AMP = [0.6, 0.6, 0.18, 0.18]; // brazos más libres que piernas
const CHAINS = [[11, [13, 15, 17, 19, 21]], [12, [14, 16, 18, 20, 22]], [23, [25, 27, 29, 31]], [24, [26, 28, 30, 32]]];
function rot(p, i, ox, oy, a) {
  const x = p[2 * i] - ox, y = p[2 * i + 1] - oy, c = Math.cos(a), s = Math.sin(a);
  p[2 * i] = ox + x * c - y * s; p[2 * i + 1] = oy + x * s + y * c;
}
export function futureExtrap(last, prev, variant, cur, damp = 0.85) {
  const g = GAINS[variant % GAINS.length];
  const [cx, cy, cs] = cur;
  const vel = new Float32Array(PD);
  let rvx = 0, rvy = 0;
  if (prev) {
    for (let c = 0; c < PD; c++) vel[c] = last[c] - prev[c];
    const s = last[PD + 2] || 1;
    rvx = (last[PD] - prev[PD]) / s; rvy = (last[PD + 1] - prev[PD + 1]) / s;
  }
  const amp = variant === 0 ? 0 : 1;
  const ph = variant * 2.39996;
  const driftX = Math.cos(ph) * 0.3 * amp, driftY = Math.sin(ph) * 0.04 * amp;
  const lean = Math.sin(ph * 1.3) * 0.1 * amp;
  const ang = CHAINS.map((_, k) => Math.sin(ph * (k + 1) * 1.7 + k) * CHAIN_AMP[k] * amp);
  const frames = [];
  for (let j = 0; j <= F; j++) {
    const cum = damp * (1 - Math.pow(damp, j)) / (1 - damp);
    const ramp = j / F, ease = ramp * ramp * (3 - 2 * ramp);
    const n = new Float32Array(PD);
    for (let c = 0; c < PD; c++) n[c] = last[c] + vel[c] * g * cum;
    if (amp) {
      for (let i = 0; i <= 22; i++) rot(n, i, 0, 0, lean * ease);
      CHAINS.forEach(([root, pts], k) => { for (const i of pts) rot(n, i, n[2 * root], n[2 * root + 1], ang[k] * ease); });
    }
    const rx = cx + (rvx * g * cum + driftX * ease) * cs, ry = cy + (rvy * g * cum + driftY * ease) * cs;
    const p = new Float32Array(PD);
    for (let c = 0; c < PD; c += 2) { p[c] = rx + n[c] * cs; p[c + 1] = ry + n[c + 1] * cs; }
    frames.push(p);
  }
  return frames;
}

// ── Esqueleto para deformar la imagen ──
// Puntos virtuales: 33 = medio de hombros, 34 = medio de caderas.
export const BONES = [
  [33, 0], [7, 8], [34, 33], [11, 12], [23, 24],
  [11, 13], [13, 15], [15, 19], [12, 14], [14, 16], [16, 20],
  [23, 25], [25, 27], [27, 31], [24, 26], [26, 28], [28, 32],
];
export const NB = BONES.length;
export function extend(p) {
  const e = new Float32Array(70);
  e.set(p);
  e[66] = (p[22] + p[24]) / 2; e[67] = (p[23] + p[25]) / 2;
  e[68] = (p[46] + p[48]) / 2; e[69] = (p[47] + p[49]) / 2;
  return e;
}
// Transformación de semejanza por hueso (rotación + escala acotada + traslación) de la pose de la
// imagen (rest) a la pose del fantasma (target). out: { bone: 4·NB, mx: 3·NB, my: 3·NB }
export function boneTransforms(rest, target, out) {
  for (let i = 0; i < NB; i++) {
    const [a, b] = BONES[i];
    const a0x = rest[2 * a], a0y = rest[2 * a + 1], b0x = rest[2 * b], b0y = rest[2 * b + 1];
    const a1x = target[2 * a], a1y = target[2 * a + 1], b1x = target[2 * b], b1y = target[2 * b + 1];
    const v0x = b0x - a0x, v0y = b0y - a0y, v1x = b1x - a1x, v1y = b1y - a1y;
    const l0 = v0x * v0x + v0y * v0y;
    let c = 1, s = 0;
    if (l0 > 1e-6) {
      c = (v0x * v1x + v0y * v1y) / l0; s = (v0x * v1y - v0y * v1x) / l0;
      const sc = Math.hypot(c, s), cl = Math.min(1.25, Math.max(0.8, sc));
      if (sc > 1e-6) { c *= cl / sc; s *= cl / sc; }
    }
    out.bone.set([a0x, a0y, b0x, b0y], 4 * i);
    out.mx.set([c, -s, a1x - (c * a0x - s * a0y)], 3 * i);
    out.my.set([s, c, a1y - (s * a0x + c * a0y)], 3 * i);
  }
  return out;
}
