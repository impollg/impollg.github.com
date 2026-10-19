# Lo que ibas a hacer — v2

El espectador se ve a sí mismo: su imagen real, recortada del fondo. Sus futuros son esa misma imagen, dibujada como boceto y deformada hacia movimientos que otras personas hicieron antes. No hay figuras: los fantasmas son él.

## Qué cambió respecto de la v1

- **33 puntos del cuerpo** (cara, manos y pies incluidos) en lugar de 13, con suavizado One-Euro para que no tiemblen.
- **Modelo de pose completo** por defecto (más preciso); en el panel se puede pasar al liviano si el teléfono va lento.
- **Render en WebGL2.** Cada fantasma es una malla fina (72 × 96) sobre la imagen del espectador. Cada punto de la malla sigue a los huesos cercanos, así el cuerpo se dobla como cuerpo. Donde la deformación desgarraría la imagen, esa zona se desvanece.
- **Estilos de imagen** para el espectador y para los fantasmas: Foto, Boceto, Sombra y Trazo. Por defecto: espectador en Foto, fantasmas en Boceto.
- Estela, desenfoque, polvo y grano ajustables.
- Al irse la persona, los fantasmas siguen con su última imagen.
- Los futuros de respaldo (cuando el archivo es chico) mueven brazos, piernas y torso como cadenas articuladas, sin deformar el cuerpo.

El archivo de la v2 tiene otro formato (33 puntos): no se mezcla con el de la v1.

## Archivos

`index.html`, `app.js`, `core.js`, `render.js`, `sw.js`, `manifest.webmanifest`, `icon.svg`. Prueba de la lógica: `node test/core.test.mjs`.

## Publicar en GitHub Pages

1. Crear un repositorio público nuevo (por ejemplo `lo-que-ibas-a-hacer-2`).
2. "uploading an existing file" → subir todos los archivos menos la carpeta `test`.
3. Settings → Pages → "Deploy from a branch" → `main`, `/ (root)` → Save.
4. Abrir `https://tu-usuario.github.io/lo-que-ibas-a-hacer-2/` en Safari o Chrome del teléfono.

## Uso

- Tocar para iniciar. El teléfono horizontal, apoyado, a 2–3 m.
- Panel oculto: triple toque en la esquina superior derecha.
- Si el teléfono no puede leer la silueta por GPU, la app pasa sola a CPU (más lenta). Se puede forzar en el panel ("Procesador").

## Parámetros que más cambian el aspecto

- **Fantasmas / Espectador** (estilo), **Brillo**, **Opacidad máxima fantasma**.
- **Estela**: cuánto rastro deja cada fantasma.
- **Desenfoque** y **Mezcla nítido / desenfocado**: de humo a dibujo.
- **Rigidez de la deformación**: más baja, el cuerpo se dobla más localmente; más alta, se mueve más en bloque.
- **Polvo** y **Grano**.

## Archivo semilla

Igual que en la v1: activar "Grabar sesiones como semilla", que pasen 15–20 personas, "Exportar semilla", guardar como `seed.json` junto a `index.html` y volver a subir.
