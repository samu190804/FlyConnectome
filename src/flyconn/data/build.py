"""Construye el subcircuito procesado (Parquet) a partir de data/raw.

Uso:  uv run python -m flyconn.data.build
Genera data/processed/neurons.parquet y data/processed/edges.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .circuit import RAW, load_edges, load_neurons

PROCESSED = Path("data/processed")
LAYERS = ["ORN", "PN", "LN", "KC", "MBON"]
# Las coordenadas de Codex (coordinates.csv) están en nanómetros. Comprobado: con esta unidad el
# cerebro mide ~710 x 290 x 235 µm (tamaño real); suponer vóxeles de 4x4x40 nm lo estiraba 4, 4 y 40 veces.
NM_PER_UNIT = np.array([1.0, 1.0, 1.0])


def assign_sign(neurons: pd.DataFrame) -> pd.Series:
    """Signo (+1 excita, -1 inhibe) de cada neurona.

    Reglas acordadas:
      - ORN y KC: siempre excitadoras (son colinérgicas).
      - PN y MBON: según el neurotransmisor predicho (ACH +; GABA y GLUT -).
        Si falta predicción en MBON -> + (la mayoría es colinérgica).
      - LN: ACH +; GABA, GLUT y, si falta o es SER/OCT, -1 (las LN son sobre todo inhibidoras).
    """
    nt = neurons["nt_type"]
    by_nt = nt.map({"ACH": 1, "GABA": -1, "GLUT": -1})
    sign = by_nt.copy()
    sign[neurons.layer.isin(["ORN", "KC"])] = 1
    ln = neurons.layer == "LN"
    sign[ln & by_nt.isna()] = -1
    mbon = neurons.layer == "MBON"
    sign[mbon & by_nt.isna()] = 1
    return sign.astype(int)


def prune_unreachable(neurons: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Se queda solo con neuronas alcanzables desde las ORN siguiendo las aristas permitidas."""
    reached = set(neurons.root_id[neurons.layer == "ORN"])
    frontier = set(reached)
    by_pre = {k: g.post_root_id.to_numpy() for k, g in edges.groupby("pre_root_id")}
    while frontier:
        new = set()
        for r in frontier:
            new.update(by_pre.get(r, ()))
        frontier = new - reached
        reached |= new
    return neurons[neurons.root_id.isin(reached)].reset_index(drop=True)


def load_positions(raw: Path = RAW) -> pd.DataFrame:
    """Posición por neurona en micras: mediana de sus puntos marcados."""
    co = pd.read_csv(raw / "coordinates.csv.gz")
    xyz = co.position.str.strip("[]").str.split(expand=True).astype(float).to_numpy()
    co[["x_um", "y_um", "z_um"]] = xyz * NM_PER_UNIT / 1000.0
    return co.groupby("root_id")[["x_um", "y_um", "z_um"]].median().reset_index()


def build(raw: Path = RAW, out: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons = load_neurons(raw)
    edges = load_edges(neurons, raw)
    n_before = neurons.layer.value_counts().to_dict()

    neurons = prune_unreachable(neurons, edges)
    neurons["sign"] = assign_sign(neurons)
    neurons = neurons.merge(load_positions(raw), on="root_id", how="left")
    neurons["layer"] = pd.Categorical(neurons.layer, LAYERS, ordered=True)
    neurons = neurons.sort_values(["layer", "side", "root_id"]).reset_index(drop=True)
    neurons["idx"] = np.arange(len(neurons))  # índice global
    neurons["layer_idx"] = neurons.groupby("layer", observed=True).cumcount()  # índice dentro de la capa

    edges = edges[edges.pre_root_id.isin(neurons.root_id) & edges.post_root_id.isin(neurons.root_id)]
    ix = neurons.set_index("root_id")
    edges = edges.assign(
        pre_idx=edges.pre_root_id.map(ix.idx).to_numpy(),
        post_idx=edges.post_root_id.map(ix.idx).to_numpy(),
        pre_layer_idx=edges.pre_root_id.map(ix.layer_idx).to_numpy(),
        post_layer_idx=edges.post_root_id.map(ix.layer_idx).to_numpy(),
        sign=edges.pre_root_id.map(ix.sign).to_numpy(),
    ).sort_values(["pre_layer", "post_layer", "pre_idx", "post_idx"]).reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(out / "neurons.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)

    removed = {k: n_before[k] - int((neurons.layer == k).sum()) for k in LAYERS}
    print("Neuronas descartadas por no recibir entrada de la cadena:", removed)
    print(f"Guardado: {len(neurons)} neuronas, {len(edges)} aristas en {out}/")
    return neurons, edges


if __name__ == "__main__":
    build()
