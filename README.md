# Lo que ibas a hacer

App web estática: cámara → MediaPipe (pose + segmentación) → predicción por similitud de movimiento contra el archivo de visitantes → fantasmas. Todo corre en el navegador; no hay servidor.

## Archivos

- `index.html` — página y estilos
- `app.js` — cámara, detección, estados, render, panel, archivo (IndexedDB)
- `core.js` — lógica pura: normalización, búsqueda (corre en un Web Worker), futuros, extrapolación
- `sw.js`, `manifest.webmanifest`, `icon.svg` — PWA y funcionamiento sin conexión
- `test/core.test.mjs` — pruebas de la lógica (`node test/core.test.mjs`)
- `seed.json` — **no incluido todavía**: el archivo semilla (ver abajo)

## Publicar (HTTPS es obligatorio para la cámara)

**Netlify Drop (lo más rápido):** desde una compu, entrar a app.netlify.com/drop y arrastrar la carpeta. Da una dirección https al instante.

**GitHub Pages:** subir la carpeta a un repositorio → Settings → Pages → rama `main`, carpeta raíz.

Primera vez: abrir la dirección con internet, tocar para iniciar y dejar que cargue (descarga el modelo, ~6 MB). Después funciona en modo avión. Al publicar cambios, subir `VERSION` en `sw.js`.

## Uso

- Tocar la pantalla para iniciar (pide cámara, pantalla completa y mantiene la pantalla encendida).
- **Panel oculto:** triple toque en la esquina superior derecha. Todos los parámetros, capa de depuración, exportar/importar, borrar visitas.
- En iPhone no se puede bloquear la orientación: si el teléfono está vertical aparece "Girá el teléfono".

## Archivo semilla

Hasta que el archivo tenga 2 minutos de movimiento, la app predice por extrapolación (los fantasmas aparecen igual). Para grabar la semilla:

1. Panel → activar "Grabar sesiones como semilla".
2. Pasan 15–20 personas, de a una, moviéndose variado (cada sesión ≥ 3 s; salir del cuadro cierra la sesión).
3. Desactivar la opción → "Exportar semilla" → guardar como `seed.json` en esta carpeta y volver a publicar. En dispositivos nuevos se carga sola; en otros, "Importar como semilla".

La semilla nunca se borra al llenarse el archivo (tope 30.000 muestras ≈ 50 min).

## Probar en el dispositivo de referencia

Con la capa de depuración activada se ven fps de render y de pose, K, error de predicción, tamaño del archivo y tiempo de búsqueda. Puntos a validar contra los criterios de aceptación: ≥ 24 fps de render y ≥ 15 de pose; latencia < 150 ms; detección a 2,5–3,5 m; prueba de 8 horas (la app recupera sola la cámara si se corta).

Ajustes probables tras las primeras pruebas: `obeyThr` / `devThr` (si funde o multiplica demasiado fácil), `blurScale` / `blurPx` (aspecto de los fantasmas; en Safari iOS el desenfoque sale de la reducción, porque `ctx.filter` puede no estar), `matchMax` (cuándo el archivo le gana a la extrapolación).
