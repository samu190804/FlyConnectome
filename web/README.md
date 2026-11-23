# Web (Astro + React + Tailwind CSS)

Web estática en español: introducción biológica, visor 3D del circuito con la animación de la señal, resultados comparativos, límites y referencias. Todos los comandos, desde esta carpeta (`web/`). Requiere **Node 22.23 o superior** (Astro 7).

```bash
npm install        # instala dependencias
npm run dev        # servidor local con recarga: http://localhost:4321
npm run build      # genera dist/ (lo que se despliega)
npm run preview    # sirve dist/ para comprobarlo antes de desplegar
```

## Datos
La web usa tres JSON de `public/data/`, generados desde Python (desde la raíz del repositorio):
```bash
uv run python -m flyconn.export
```
- `graph.json`: neuronas (capa, tipo, lado, signo, posición 3D) y conexiones.
- `activations.json`: actividad de 10 olores de prueba con la red del conectoma B.
- `results.json`: curvas y comparaciones de los modelos (se importa al compilar, así que hay que volver a ejecutar `npm run build` tras cambiarlo).

## Estructura
- `src/pages/index.astro`: la página (secciones estáticas, sin JavaScript).
- `src/layouts/Base.astro`: cabecera, navegación y pie.
- `src/components/BrainMap.astro`: esquema del cerebro y lista de regiones (estático).
- `src/components/ViewerApp.jsx`: isla React con el visor 3D y el panel de etapas (solo en el navegador).
- `src/components/Results.jsx` y `LineChart.jsx`: isla React con tarjetas, gráficas y tabla de comparaciones (se hidrata al hacer scroll).
- `src/lib/viewerEngine.js`: motor del visor (3d-force-graph sobre three.js), independiente de React. Dibuja las neuronas como puntos con un halo muy suave (un shader propio, una sola malla), mueve la cámara entre etapas, enciende las neuronas en una onda con destello y anillo, ilumina las conexiones por las que viaja la señal y muestra una etiqueta al pasar el ratón.
- `src/lib/content.js`: textos biológicos, etapas, referencias y estilos de series.
- `src/styles/global.css`: Tailwind CSS 4, colores y tipografías del tema.

## Despliegue en Netlify
`netlify.toml` define el build (`npm run build`, carpeta `dist`) y Node 22. En Netlify, importa el repositorio y fija **Base directory = `web`**. Los datos de FlyWire se usan bajo CC BY-NC 4.0: el sitio debe ser **no comercial** y mantener la atribución (ya está en el pie y en las referencias).

## Notas
- Aspecto: estilo editorial sobrio. Fondo gris mineral, titulares con serifa (Fraunces), texto en Inter y un único color de acento (cobalto); los colores de las capas son apagados. Todo definido en `src/styles/global.css`. El visor 3D es una pieza oscura dentro de la página. Respeta `prefers-reduced-motion`.
- El resplandor se hace en el propio shader de los puntos. Probé un efecto de postprocesado (bloom), pero aclaraba todo el fondo, así que se descartó.
- Las posiciones son las reales de cada neurona (en µm, centradas; el eje vertical se invierte para que «arriba» sea arriba). Las líneas muestran solo las entradas más fuertes de cada neurona (control en «Opciones de visualización»).
- El esquema de regiones del cerebro es una ilustración simplificada, sin escala.
- Revisa las cifras y textos con las fuentes citadas antes de publicar.
