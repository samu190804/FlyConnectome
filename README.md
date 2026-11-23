# Conectoma de la mosca como arquitectura de red neuronal

Experimento: usar el cableado real de la vía olfativa de *Drosophila melanogaster* (conectoma FlyWire) como arquitectura de una red neuronal en PyTorch, entrenarla para clasificar olores con pocos ejemplos y compararla con redes genéricas.

> **Es un experimento de arquitectura inspirado en el conectoma, no una simulación fiel de la mosca.** Usa neuronas de tasa (no de disparo) y simplifica la dopamina y la plasticidad.

## Pregunta
¿Aporta algo el cableado concreto del conectoma frente a (a) una red dispersa aleatoria con la misma densidad y (b) un MLP denso con parámetros parecidos? Todas las redes se implementan en PyTorch; lo que se compara son **arquitecturas**. Un resultado negativo es válido.

## Estado

| Paso | Estado |
|---|---|
| 1. Exploración de datos | Hecho ([notebook](notebooks/01_exploracion.ipynb)) |
| 2. Subcircuito procesado (Parquet + tests) | Hecho |
| 3. Datos de olores (Hallem y Carlson 2006) | Hecho |
| 4. Modelos | Hecho |
| 5. Experimentos | Hecho (15 semillas) |
| 6. Exportación a JSON | Hecho |
| 7. Visualizador web y despliegue | Pendiente |

## Circuito
Vía olfativa: receptores olfativos (ORN) → neuronas de proyección (PN) y neuronas locales inhibidoras (LN) en el lóbulo antenal → células de Kenyon (KC) del cuerpo fungiforme → neuronas de salida (MBON). Ambos hemisferios en una sola red.

Datos: FlyWire FAFB v783 (Codex), conexiones "Filtered". Subcircuito procesado:

| Capa | Neuronas |
|---|---|
| ORN (53 tipos de receptor) | 2.281 |
| PN | 639 |
| LN | 422 |
| KC | 4.750 |
| MBON | 83 |
| **Total** | **8.175 neuronas, 129.600 conexiones** |

Conexiones incluidas en esta versión: ORN→PN, ORN→LN, LN→ORN, LN→PN, LN→LN, PN→LN, PN→KC, KC→MBON. Se excluyen las recurrentes ORN→ORN, PN→PN, KC→KC y las que salen de las MBON (posible variante futura).

### Decisiones y simplificaciones
- **Neuronas huérfanas descartadas (493):** se eliminan las neuronas a las que no llega señal desde las ORN siguiendo las conexiones permitidas (46 PN, 7 LN, 427 KC, 13 MBON). Suelen ser reconstrucciones incompletas o neuronas que reciben señal por vías que no incluimos. Las ORN se conservan todas.
- **Signo (excitador/inhibidor)** a partir del neurotransmisor predicho (Eckstein, Bates et al., 2024), con correcciones porque la predicción falla o falta en parte de las ORN y LN:
  - ORN y KC: siempre excitadoras (son colinérgicas).
  - PN y MBON: ACH excita; GABA y glutamato inhiben. MBON sin predicción: excita.
  - LN: ACH excita; GABA y glutamato inhiben; sin predicción o serotonina/octopamina: inhiben.
- **KC por hemisferio:** unas 2.600 (no 2.000). **ORN:** 2.281, en 53 tipos.
- **Posición 3D:** mediana de los puntos marcados de cada neurona, en micras.
- El `side` de las ORN es el lado de entrada del nervio; sus conexiones cruzan entre hemisferios porque proyectan a ambos lóbulos antenales.

## Datos de olores y tarea
- **Olores:** Tabla S1 de Hallem y Carlson (2006): 110 olores y 24 receptores. Se usan 23 receptores, que se casan con tipos de ORN de FlyWire por su glomérulo (Couto et al., 2005). Solo 1.121 de las 2.281 ORN (23 de 53 tipos) tienen dato de respuesta; el resto recibe solo actividad de fondo.
- **10 clases químicas:** ácidos, acetatos, otros ésteres, alcoholes, aldehídos, cetonas, terpenos, aromáticos, lactonas y nitrógeno/azufre. Simplificaciones: los ésteres se dividieron en dos clases y las aminas se agruparon con los compuestos de azufre.
- **Partición por olores:** los olores de prueba nunca aparecen en entrenamiento (77 de entrenamiento y 33 de prueba). Cada ejemplo es una respuesta con ruido gaussiano.
- Con un clasificador trivial (comparar con el promedio de cada clase) la precisión es de 0,29 con 1 ejemplo por clase y 0,47 con 20 (azar 0,10).

## Resultados
Precisión en prueba con **olores no vistos** (media ± desviación sobre 15 semillas; cada semilla cambia la partición de olores, el ruido, el cableado barajado y la inicialización). Azar: 0,10. Ruido de prueba: 10 espigas/s. Datos en `results/all.csv`, resumen en `results/summary.csv`, gráficas en `results/learning_curve.png` y `results/noise_robustness.png`.

| Modelo | 1 ej./clase | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Conectoma A | 0,22 ± 0,09 | 0,32 ± 0,06 | 0,40 ± 0,07 | 0,48 ± 0,06 | 0,49 ± 0,07 |
| Conectoma B | 0,26 ± 0,09 | 0,37 ± 0,08 | 0,45 ± 0,08 | 0,54 ± 0,07 | 0,54 ± 0,08 |
| Aleatorio A | 0,30 ± 0,09 | 0,40 ± 0,06 | 0,47 ± 0,07 | 0,52 ± 0,06 | 0,52 ± 0,06 |
| Aleatorio B | 0,29 ± 0,07 | 0,37 ± 0,08 | 0,48 ± 0,06 | 0,54 ± 0,08 | 0,56 ± 0,07 |
| MLP denso | 0,34 ± 0,10 | 0,44 ± 0,06 | 0,52 ± 0,07 | 0,56 ± 0,07 | 0,57 ± 0,07 |

**Lectura (resultado mayormente negativo):**
- **Conectoma B frente a aleatorio B: empate.** La diferencia emparejada por semilla está entre −0,03 y +0,005, dentro del ruido. El cableado real no aporta precisión cuando se entrena toda la red.
- **Conectoma A frente a aleatorio A: el conectoma queda por debajo** (de −0,08 con 1 ejemplo a −0,03 con 20). Con pesos fijos, el cableado real funciona peor que uno aleatorio.
- **Conectoma B frente a MLP denso:** el MLP gana entre 2 y 8 puntos, con más claridad con pocos ejemplos.
- **Entrenar el interior (B frente a A)** da unos 5 puntos.
- **Robustez al ruido** (entrenados con 20 ejemplos, ruido de prueba 40): el conectoma A cae de 0,49 a 0,47, mientras que el aleatorio A cae de 0,54 a 0,37. Es lo único favorable al conectoma; el MLP y el conectoma B quedan igual (≈0,46).
- Las diferencias de unos pocos puntos están dentro del ruido. No se ha corregido por comparaciones múltiples.

**Limitaciones que condicionan la lectura:** la mitad de las ORN (30 de 53 tipos) no tiene datos de Hallem y recibe solo actividad de fondo; el conjunto de prueba son 33 olores; el barajado conserva el nº de conexiones por bloque y los signos, pero no el de cada neurona; y varias decisiones de diseño son propias (capa de lectura, normalización de pesos, 3 pasos de tiempo, top-k global, tasa de aprendizaje elegida con un piloto en semillas distintas).

## Modelos que se comparan
Las tres arquitecturas reciben lo mismo (la respuesta de los receptores a un olor) y clasifican en las mismas 10 clases químicas. Solo cambia cómo están conectadas las neuronas por dentro.

| Modelo | Cableado | Parámetros entrenables |
|---|---|---|
| **Conectoma, variante A** | Real de FlyWire, pesos fijos proporcionales al nº de sinapsis. Solo se entrena KC→MBON y la capa de lectura (como en la mosca real, donde se aprende en esa etapa) | ~26.000 |
| **Conectoma, variante B** | Real de FlyWire, todas las conexiones reales entrenables (respetando máscara y signos) | ~130.000 |
| **Aleatorio, variantes A y B** | Mismas neuronas, signos y nº de conexiones por bloque, pero entre neuronas elegidas al azar | Igual que el conectoma correspondiente |
| **MLP denso** | Todas las neuronas conectadas con todas, una capa oculta de 57 neuronas | ~130.000 |

El aleatorio tiene los mismos parámetros que el conectoma: solo cambia a qué neurona se conecta cada una. Por eso mide lo que aporta el cableado concreto, y no la capacidad. En total hay 5 modelos entrenables: conectoma A, conectoma B, aleatorio A, aleatorio B y MLP.

### Preguntas que responde el experimento
1. **Conectoma B frente a aleatorio B:** ¿importa qué neurona se conecta con cuál (con todas las conexiones entrenables)?
2. **Conectoma A frente a aleatorio A:** lo mismo, sin entrenar la estructura interna.
3. **Conectoma B frente a MLP denso:** ¿ayuda una estructura biológica frente a una red genérica del mismo tamaño?
4. **Variante A frente a B:** ¿hace falta entrenar el interior de la red, o basta con aprender la lectura?

### Cómo está implementado (PyTorch)
- `MaskedLinear`: capa lineal que solo tiene peso donde hay sinapsis real. El signo (excita/frena) viene de la neurona de origen y no cambia al entrenar; solo se aprende la magnitud.
- `ConnectomeNet`: ORN → lóbulo antenal (3 pasos de tiempo, con inhibición de las neuronas locales) → PN → KC (solo el 5 % más activo, top-k) → MBON → capa lineal hacia las 10 clases (esta capa no existe en la mosca y la usan igual las tres redes). El control aleatorio es la misma clase con el cableado barajado.
- `DenseMLP`: capa oculta de 57 neuronas, elegida para igualar los parámetros de la variante B.
- Decisiones propias: pesos iniciales normalizados por neurona, 3 pasos de tiempo, top-k global entre los dos hemisferios, y el barajado no conserva el nº de conexiones de cada neurona.

## Estructura
```
configs/      configuración de experimentos
data/raw/     descargas de FlyWire (no se sube a git)
data/processed/  subcircuito en Parquet
src/flyconn/  código Python (ver src/flyconn/README.md)
notebooks/    exploración
tests/        pytest
web/          visualizador (ver web/README.md)
results/      resultados de experimentos
```

## Reproducir
Instrucciones por lenguaje: [Python](src/flyconn/README.md) · [Web](web/README.md).

Datos: descarga de [FlyWire Codex](https://codex.flywire.ai/api/download) (requiere aceptar sus condiciones) los archivos `classification.csv.gz`, `connections_princeton.csv.gz`, `neurons.csv.gz`, `coordinates.csv.gz` y `consolidated_cell_types.csv.gz` en `data/raw/`.

## Datos, licencia y citas
Los datos de FlyWire se usan bajo **CC BY-NC 4.0** (solo uso no comercial, con atribución) y los Terms of Service de FlyWire. Citas y detalles en [CITATIONS.md](CITATIONS.md). Cita obligatoria: Dorkenwald et al., 2024, *Nature*, y Schlegel et al., 2024, *Nature*. Codex: http://dx.doi.org/10.13140/RG.2.2.35928.67844. Respuestas de receptores: Hallem y Carlson, 2006, *Cell*. Mapa receptor→glomérulo: Couto, Alenius y Dickson, 2005, *Current Biology*.
