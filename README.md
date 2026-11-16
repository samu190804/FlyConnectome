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
| 3. Datos de olores (Hallem y Carlson 2006) | Pendiente |
| 4. Modelos | Pendiente |
| 5. Experimentos | Pendiente |
| 6. Exportación a JSON | Pendiente |
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
Los datos de FlyWire se usan bajo **CC BY-NC 4.0** (solo uso no comercial, con atribución) y los Terms of Service de FlyWire. Citas y detalles en [CITATIONS.md](CITATIONS.md). Cita obligatoria: Dorkenwald et al., 2024, *Nature*, y Schlegel et al., 2024, *Nature*. Codex: http://dx.doi.org/10.13140/RG.2.2.35928.67844. Respuestas de receptores: Hallem y Carlson, 2006, *Cell*.
