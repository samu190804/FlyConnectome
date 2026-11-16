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

Los pasos de modelos, entrenamiento y exportación (`train.py`, `evaluate.py`, `export.py`) se añadirán y documentarán aquí.
