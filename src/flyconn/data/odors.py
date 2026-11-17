"""Datos de olores (Hallem y Carlson 2006) y partición few-shot por olores.

Idea:
  - Cada olor es un vector de respuesta de 23 receptores (espigas/s sobre la actividad espontánea).
  - Cada ORN de FlyWire hereda la respuesta de su tipo de receptor (via glomérulo). Las ORN de tipos
    sin dato en Hallem reciben solo la actividad de fondo (la media de la actividad espontánea).
  - Una muestra = respuesta del olor + ruido gaussiano independiente por neurona.
  - Partición: los olores de test NO aparecen nunca en entrenamiento (se reserva una fracción
    de olores por clase). Los ejemplos de entrenamiento son muestras ruidosas de los olores de train.

Todo es reproducible: solo depende de las semillas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HALLEM_CSV = Path("data/processed/hallem_carlson_s1.csv")
CLASSES_CSV = Path("configs/odor_classes.csv")
RECEPTOR_MAP_CSV = Path("configs/receptor_to_glomerulus.csv")
NEURONS_PARQUET = Path("data/processed/neurons.parquet")

SPONT_ROW = "__spontaneous__"
RATE_SCALE = 100.0  # espigas/s -> unidades de entrada de la red (rate / RATE_SCALE)


@dataclass(frozen=True)
class OdorTable:
    """Respuestas por olor sobre cada ORN (en orden de layer_idx de la capa ORN)."""

    names: list[str]
    classes: np.ndarray        # (n_odors,) etiqueta entera 0..9
    class_names: list[str]
    clean: np.ndarray          # (n_odors, n_orn) tasa de disparo sin ruido, espigas/s
    has_data: np.ndarray       # (n_orn,) bool: la ORN tiene dato de Hallem


def load_odor_table(
    hallem_csv: Path = HALLEM_CSV,
    classes_csv: Path = CLASSES_CSV,
    map_csv: Path = RECEPTOR_MAP_CSV,
    neurons_parquet: Path = NEURONS_PARQUET,
) -> OdorTable:
    h = pd.read_csv(hallem_csv, index_col=0)
    spont = h.loc[SPONT_ROW]
    resp = h.drop(index=SPONT_ROW)
    cls = pd.read_csv(classes_csv).set_index("odor")["class"].reindex(resp.index)
    assert cls.notna().all(), "olores sin clase"
    class_names = sorted(cls.unique())

    rmap = pd.read_csv(map_csv)
    orn = (
        pd.read_parquet(neurons_parquet).query("layer == 'ORN'").sort_values("layer_idx")
    )
    type_to_receptor = dict(zip(rmap.orn_type, rmap.receptor))
    background = float(spont[rmap.receptor].mean())  # actividad de fondo de las ORN sin dato

    clean = np.full((len(resp), len(orn)), background, dtype=np.float64)
    has_data = np.zeros(len(orn), dtype=bool)
    for j, t in enumerate(orn.primary_type.to_numpy()):
        rec = type_to_receptor.get(t)
        if rec is not None:
            clean[:, j] = (resp[rec] + spont[rec]).to_numpy()  # respuesta total = espontánea + delta
            has_data[j] = True
    clean = np.clip(clean, 0.0, None)  # una tasa de disparo no puede ser negativa
    return OdorTable(
        names=resp.index.tolist(),
        classes=cls.map({c: i for i, c in enumerate(class_names)}).to_numpy(),
        class_names=class_names,
        clean=clean,
        has_data=has_data,
    )


@dataclass(frozen=True)
class OdorSplit:
    train_odors: np.ndarray  # índices de olor
    test_odors: np.ndarray


def split_odors(table: OdorTable, test_frac: float = 0.3, min_test: int = 2, seed: int = 0) -> OdorSplit:
    """Reserva olores de test por clase (al menos `min_test`, y siempre queda algún olor de train)."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in range(len(table.class_names)):
        idx = rng.permutation(np.flatnonzero(table.classes == c))
        n_test = min(max(min_test, round(test_frac * len(idx))), len(idx) - 1)
        test += idx[:n_test].tolist()
        train += idx[n_test:].tolist()
    return OdorSplit(np.array(sorted(train)), np.array(sorted(test)))


def sample(
    table: OdorTable, odors: np.ndarray, noise_sd: float, rng: np.random.Generator
) -> np.ndarray:
    """Muestras ruidosas (espigas/s, >= 0) para los olores dados, una por fila."""
    x = table.clean[odors] + rng.normal(0.0, noise_sd, size=(len(odors), table.clean.shape[1]))
    return np.clip(x, 0.0, None)


def make_fewshot_train(
    table: OdorTable, split: OdorSplit, n_per_class: int, noise_sd: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """`n_per_class` muestras ruidosas por clase, tomando olores de train (con reemplazo si hace falta).

    Devuelve (X en unidades de entrada de la red, y).
    """
    rng = np.random.default_rng(seed)
    odors, labels = [], []
    for c in range(len(table.class_names)):
        pool = split.train_odors[table.classes[split.train_odors] == c]
        # Sin reemplazo mientras haya olores distintos; después se repiten (con otro ruido).
        picks = np.concatenate([rng.permutation(pool) for _ in range(-(-n_per_class // len(pool)))])[:n_per_class]
        odors += picks.tolist()
        labels += [c] * n_per_class
    X = sample(table, np.array(odors), noise_sd, rng) / RATE_SCALE
    return X.astype(np.float32), np.array(labels)


def make_test(
    table: OdorTable, split: OdorSplit, samples_per_odor: int, noise_sd: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Conjunto de prueba fijo: `samples_per_odor` muestras ruidosas de cada olor de test."""
    rng = np.random.default_rng(seed)
    odors = np.repeat(split.test_odors, samples_per_odor)
    X = sample(table, odors, noise_sd, rng) / RATE_SCALE
    return X.astype(np.float32), table.classes[odors]
