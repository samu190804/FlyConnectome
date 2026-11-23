# Python (datos, modelos, experimentos)

Requisitos: Python 3.11+ y [uv](https://docs.astral.sh/uv/). Todos los comandos se ejecutan **desde la raíz del repositorio**.

## Preparar el entorno
```bash
uv sync                      # crea .venv e instala dependencias (incluye las de desarrollo)
```

## Datos
Pon los archivos de Codex en `data/raw/` (ver [README principal](../../README.md)). Después:
```bash
uv run python -m flyconn.data.build   # genera data/processed/neurons.parquet y edges.parquet
```
Imprime cuántas neuronas huérfanas se descartan por capa.

## Notebooks
```bash
uv run jupyter lab                    # abrir y editar notebooks/
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploracion.ipynb   # re-ejecutar
```

## Tests
```bash
uv run pytest -q                      # los de data/processed se saltan si no has ejecutado build
```

## Módulos
- `data/circuit.py`: carga neuronas y aristas del subcircuito (capas y aristas permitidas).
- `data/build.py`: poda de huérfanas, signos, posiciones y escritura de Parquet.

- `data/hallem.py`: extrae la Tabla S1 de `data/raw/hallem_carlson/mmc1.pdf` (necesita `pdftotext`): `uv run python -m flyconn.data.hallem`.
- `data/odors.py`: respuestas por ORN, partición few-shot por olores y generación de muestras con ruido.
- `models/`: `MaskedLinear`, `ConnectomeNet` (variantes A/B y control barajado con `shuffle_spec`) y `DenseMLP`.

Ejemplo de uso:
```python
from flyconn.models import load_spec, shuffle_spec, ConnectomeNet
spec = load_spec()
real = ConnectomeNet(spec, variant="B")
control = ConnectomeNet(shuffle_spec(spec, seed=0), variant="B")
```

## Experimentos
```bash
uv run python -m flyconn.train --config configs/default.yaml            # entrena los 5 modelos (≈20-40 min por cada 5 semillas en CPU)
uv run python -m flyconn.train --override "seeds=[5,6]" out_csv=results/otro.csv   # sobrescribir opciones
uv run python -m flyconn.evaluate --csv results/all.csv                 # tabla, diferencias emparejadas y gráficas
```
`results/all.csv` reúne las 15 semillas (0-4 de `main.csv` y 5-14 de `main_extra.csv`).

## Exportar para la web
```bash
uv run python -m flyconn.export          # escribe web/public/data/{graph,activations,results}.json
```
- `graph.json`: nodos (capa, tipo, lado, signo, posición 3D en µm centrada) y aristas (origen, destino, sinapsis).
- `activations.json`: actividad por capa y paso de tiempo de 10 olores de prueba (uno por clase) con la red conectoma B entrenada (semilla 0, 20 ejemplos por clase). Acierta 5 de 10.
- `results.json`: curvas, robustez al ruido y diferencias emparejadas, con las limitaciones.

Los pasos de entrenamiento, entrenamiento y exportación (`train.py`, `evaluate.py`, `export.py`) se añadirán y documentarán aquí.
