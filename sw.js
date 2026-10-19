// Service worker: la app y el modelo quedan en caché para funcionar sin conexión.
// Cambiar VERSION al publicar una versión nueva.
const VERSION = 'lqiah2-v1';
const SHELL = ['./', './index.html', './app.js?v=3', './core.js', './render.js', './manifest.webmanifest', './icon.svg'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const sameOrigin = new URL(req.url).origin === location.origin;
  const put = (res) => { if (res && (res.ok || res.type === 'opaque')) { const cp = res.clone(); caches.open(VERSION).then((c) => c.put(req, cp)); } return res; };
  if (sameOrigin) {
    // Propios: red primero (para tomar cambios), caché si no hay conexión.
    e.respondWith(fetch(req).then(put).catch(() => caches.match(req, { ignoreSearch: true }).then((r) => r || caches.match('./index.html'))));
  } else {
    // MediaPipe (librería, wasm y modelo): caché primero.
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then(put)));
  }
});
