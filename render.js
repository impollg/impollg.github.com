// render.js — WebGL2. El espectador es su propia imagen recortada (foto, boceto o sombra).
// Cada fantasma es esa misma imagen deformada hacia la pose predicha: una malla fina sobre el cuerpo,
// donde cada vértice sigue a los huesos cercanos (skinning 2D). Los fantasmas se acumulan en un
// búfer con estela, desenfoque y grano.
import { NB } from './core.js';

const VS_MESH = `#version 300 es
precision highp float;
#define NB ${NB}
in vec2 a_uv;
uniform vec4 u_box; uniform vec2 u_video; uniform vec4 u_view;
uniform int u_nb; uniform vec4 u_bone[NB]; uniform vec3 u_mx[NB]; uniform vec3 u_my[NB]; uniform float u_sigma;
out vec2 v_tex;
out float v_keep;
// Skinning 2D: cada punto sigue a los huesos cercanos, con peso inverso a la distancia (potencia 4).
vec3 warp(vec2 p) {
  float ws = 0.0; vec2 acc = vec2(0.0); float acc2 = 0.0; vec3 P = vec3(p, 1.0);
  for (int i = 0; i < NB; i++) {
    if (i >= u_nb) break;
    vec2 A = u_bone[i].xy, ab = u_bone[i].zw - A;
    float t = clamp(dot(p - A, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
    float d = length(p - (A + ab * t)) / u_sigma;
    float d2 = d * d;
    float w = 1.0 / (d2 * d2 + 1e-3);
    vec2 qi = vec2(dot(u_mx[i], P), dot(u_my[i], P));
    acc += w * qi; acc2 += w * dot(qi, qi); ws += w;
  }
  vec2 q = acc / ws;
  return vec3(q, sqrt(max(acc2 / ws - dot(q, q), 0.0)));
}
void main() {
  vec2 p = mix(u_box.xy, u_box.zw, a_uv);
  v_tex = p / u_video;
  vec2 q = p;
  v_keep = 1.0;
  if (u_nb > 0) {
    vec3 w0 = warp(p);
    q = w0.xy;
    // Estiramiento local: si la malla se estira, se pliega o se desgarra, esa zona se desvanece.
    float h = 3.0;
    vec2 jx = (warp(p + vec2(h, 0.0)).xy - q) / h, jy = (warp(p + vec2(0.0, h)).xy - q) / h;
    float stretch = max(length(jx), length(jy));
    float det = jx.x * jy.y - jx.y * jy.x;
    v_keep = (1.0 - smoothstep(1.8, 3.2, stretch)) * smoothstep(0.08, 0.3, det)
           * (1.0 - smoothstep(0.5, 1.8, w0.z / u_sigma));
  }
  gl_Position = vec4(q * u_view.xy + u_view.zw, 0.0, 1.0);
}`;

const FS_MESH = `#version 300 es
precision highp float;
in vec2 v_tex; in float v_keep; out vec4 o;
uniform sampler2D u_cam; uniform sampler2D u_mask; uniform vec2 u_texel;
uniform float u_alpha, u_fill, u_edge, u_rim, u_flash, u_time, u_grain, u_dissolve, u_gamma, u_contrast;
uniform vec3 u_tint;
float lum(vec2 t) { return dot(texture(u_cam, t).rgb, vec3(0.299, 0.587, 0.114)); }
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
  if (v_tex.x < 0.0 || v_tex.y < 0.0 || v_tex.x > 1.0 || v_tex.y > 1.0) discard;
  float m = smoothstep(0.3, 0.7, texture(u_mask, v_tex).r);
  if (m < 0.004) discard;
  vec2 e = u_texel;
  float a00 = lum(v_tex + e * vec2(-1.0, -1.0)), a10 = lum(v_tex + e * vec2(0.0, -1.0)), a20 = lum(v_tex + e * vec2(1.0, -1.0));
  float a01 = lum(v_tex + e * vec2(-1.0, 0.0)), l = lum(v_tex), a21 = lum(v_tex + e * vec2(1.0, 0.0));
  float a02 = lum(v_tex + e * vec2(-1.0, 1.0)), a12 = lum(v_tex + e * vec2(0.0, 1.0)), a22 = lum(v_tex + e * vec2(1.0, 1.0));
  float gx = (a20 + 2.0 * a21 + a22) - (a00 + 2.0 * a01 + a02);
  float gy = (a02 + 2.0 * a12 + a22) - (a00 + 2.0 * a10 + a20);
  float edge = smoothstep(0.06, 0.5, length(vec2(gx, gy)));
  vec2 e2 = e * 2.5;
  float mx = texture(u_mask, v_tex + vec2(e2.x, 0.0)).r - texture(u_mask, v_tex - vec2(e2.x, 0.0)).r;
  float my = texture(u_mask, v_tex + vec2(0.0, e2.y)).r - texture(u_mask, v_tex - vec2(0.0, e2.y)).r;
  float rim = clamp(length(vec2(mx, my)) * 1.4, 0.0, 1.0);
  float tone = pow(clamp((l - 0.5) * u_contrast + 0.5, 0.0, 1.0), u_gamma);
  float g = hash(gl_FragCoord.xy + fract(u_time * 0.37) * 311.0) - 0.5;
  float v = tone * u_fill + edge * u_edge + rim * u_rim + g * u_grain;
  float a = m * u_alpha * v_keep;
  if (u_dissolve > 0.0) a *= smoothstep(u_dissolve - 0.2, u_dissolve + 0.2, hash(floor(v_tex / e)));
  vec3 c = u_tint * max(v, 0.0) + vec3(u_flash);
  o = vec4(c * a, a);
}`;

const VS_QUAD = `#version 300 es
in vec2 a_pos; out vec2 v_uv;
void main() { v_uv = a_pos * 0.5 + 0.5; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FS_DECAY = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o; uniform sampler2D u_src; uniform float u_decay;
void main() { o = max(texture(u_src, v_uv) * u_decay - vec4(1.5 / 255.0), vec4(0.0)); }`;

const FS_COMP = `#version 300 es
precision highp float;
in vec2 v_uv; out vec4 o;
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_blur, u_gain, u_time, u_grain, u_soft;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
  vec3 sharp = texture(u_src, v_uv).rgb;
  vec3 acc = sharp * 0.16; float W = 0.16;
  for (int i = 0; i < 16; i++) {
    float ang = float(i) * 0.3926991 + (i % 2 == 0 ? 0.0 : 0.19);
    float r = (i % 2 == 0) ? 1.0 : 2.3;
    float w = (i % 2 == 0) ? 0.075 : 0.035;
    acc += texture(u_src, v_uv + vec2(cos(ang), sin(ang)) * r * u_blur * u_texel).rgb * w; W += w;
  }
  vec3 c = mix(sharp, acc / W, u_soft) * u_gain;
  c += (hash(gl_FragCoord.xy + fract(u_time * 0.53) * 199.0) - 0.5) * u_grain;
  o = vec4(max(c, 0.0), 1.0);
}`;

function compile(gl, vs, fs) {
  const sh = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader: ' + gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'a_uv'); gl.bindAttribLocation(p, 0, 'a_pos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Programa: ' + gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); const name = info.name.replace(/\[0\]$/, ''); u[name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false, premultipliedAlpha: true, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('Este navegador no tiene WebGL2. Actualizá el sistema o probá con Chrome/Safari recientes.');
    this.gl = gl; this.canvas = canvas;
    this.mesh = compile(gl, VS_MESH, FS_MESH);
    this.decay = compile(gl, VS_QUAD, FS_DECAY);
    this.comp = compile(gl, VS_QUAD, FS_COMP);
    this.floatFB = !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));

    // Malla fina: 72 × 96 celdas sobre el cuerpo.
    const NX = 72, NY = 96, uv = new Float32Array((NX + 1) * (NY + 1) * 2), idx = new Uint16Array(NX * NY * 6);
    let k = 0; for (let y = 0; y <= NY; y++) for (let x = 0; x <= NX; x++) { uv[k++] = x / NX; uv[k++] = y / NY; }
    k = 0; for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
      const i = y * (NX + 1) + x; idx.set([i, i + 1, i + NX + 1, i + 1, i + NX + 2, i + NX + 1], k); k += 6;
    }
    this.meshCount = idx.length;
    this.meshVao = gl.createVertexArray(); gl.bindVertexArray(this.meshVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    this.quadVao = gl.createVertexArray(); gl.bindVertexArray(this.quadVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    const tex = () => {
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    this.camTex = tex(); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    this.maskTex = tex(); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
    this.tex = tex; this.fb = [null, null]; this.gw = 0; this.gh = 0; this.texW = 1; this.texH = 1;
  }
  resize(w, h, ghostScale) {
    const gl = this.gl;
    const gw = Math.max(2, Math.round(w * ghostScale)), gh = Math.max(2, Math.round(h * ghostScale));
    if (gw === this.gw && gh === this.gh) return;
    this.gw = gw; this.gh = gh;
    for (let i = 0; i < 2; i++) {
      if (this.fb[i]) { gl.deleteFramebuffer(this.fb[i].f); gl.deleteTexture(this.fb[i].t); }
      const t = this.tex();
      if (this.floatFB) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, gw, gh, 0, gl.RGBA, gl.HALF_FLOAT, null);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gw, gh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      this.fb[i] = { f, t };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  // Cámara y máscara del mismo cuadro: así la imagen y la silueta coinciden.
  upload(video, mask, mw, mh) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    this.texW = video.videoWidth || this.texW; this.texH = video.videoHeight || this.texH;
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, mw, mh, 0, gl.RED, gl.UNSIGNED_BYTE, mask);
  }
  _drawMesh(m, f, style, alpha, flash, tint) {
    const gl = this.gl, u = this.mesh.u;
    gl.uniform4f(u.u_box, m.box[0], m.box[1], m.box[2], m.box[3]);
    gl.uniform1i(u.u_nb, m.bones ? NB : 0);
    if (m.bones) { gl.uniform4fv(u.u_bone, m.bones.bone); gl.uniform3fv(u.u_mx, m.bones.mx); gl.uniform3fv(u.u_my, m.bones.my); }
    gl.uniform1f(u.u_alpha, alpha); gl.uniform1f(u.u_flash, flash || 0);
    gl.uniform3f(u.u_tint, tint[0], tint[1], tint[2]);
    gl.uniform1f(u.u_fill, style.fill); gl.uniform1f(u.u_edge, style.edge); gl.uniform1f(u.u_rim, style.rim);
    gl.uniform1f(u.u_gamma, style.gamma); gl.uniform1f(u.u_contrast, style.contrast);
    gl.uniform1f(u.u_grain, style.grain); gl.uniform1f(u.u_dissolve, style.dissolve);
    gl.drawElements(gl.TRIANGLES, this.meshCount, gl.UNSIGNED_SHORT, 0);
  }
  render(f) {
    const gl = this.gl, W = this.canvas.width, H = this.canvas.height;
    // 1 · Búfer de fantasmas con estela.
    const [src, dst] = this.fb;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.f); gl.viewport(0, 0, this.gw, this.gh);
    gl.disable(gl.BLEND);
    gl.useProgram(this.decay.p); gl.bindVertexArray(this.quadVao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src.t);
    gl.uniform1i(this.decay.u.u_src, 0); gl.uniform1f(this.decay.u.u_decay, f.trail);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const mu = this.mesh.u;
    gl.useProgram(this.mesh.p); gl.bindVertexArray(this.meshVao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.uniform1i(mu.u_cam, 0); gl.uniform1i(mu.u_mask, 1);
    gl.uniform2f(mu.u_video, f.video[0], f.video[1]);
    gl.uniform2f(mu.u_texel, 1 / this.texW, 1 / this.texH);
    gl.uniform4f(mu.u_view, f.view[0], f.view[1], f.view[2], f.view[3]);
    gl.uniform1f(mu.u_sigma, f.sigma); gl.uniform1f(mu.u_time, f.time);
    if (f.ghosts.length) {
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_COLOR);
      for (const g of f.ghosts) this._drawMesh(g, f, f.ghostStyle, g.alpha, g.flash, g.tint);
      gl.disable(gl.BLEND);
    }
    this.fb = [dst, src];

    // 2 · Composición: fantasmas desenfocados y luego el cuerpo nítido.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
    const cu = this.comp.u;
    gl.useProgram(this.comp.p); gl.bindVertexArray(this.quadVao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dst.t);
    gl.uniform1i(cu.u_src, 0); gl.uniform2f(cu.u_texel, 1 / this.gw, 1 / this.gh);
    gl.uniform1f(cu.u_blur, f.blur); gl.uniform1f(cu.u_gain, f.gain); gl.uniform1f(cu.u_soft, f.soft);
    gl.uniform1f(cu.u_time, f.time); gl.uniform1f(cu.u_grain, f.grain);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    if (f.self) {
      gl.useProgram(this.mesh.p); gl.bindVertexArray(this.meshVao);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.camTex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      this._drawMesh({ box: [0, 0, f.video[0], f.video[1]], bones: null }, f, f.selfStyle, f.self.alpha, 0, f.self.tint);
      gl.disable(gl.BLEND);
    }
    gl.bindVertexArray(null);
  }
}
