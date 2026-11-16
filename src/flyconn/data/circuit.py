"""Carga y filtrado del subcircuito olfativo (ORN -> PN/LN -> KC -> MBON) de FlyWire v783."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

# Clase de FlyWire (classification.csv) -> capa del modelo
CLASS_TO_LAYER = {
    "olfactory": "ORN",
    "ALPN": "PN",
    "ALLN": "LN",
    "Kenyon_Cell": "KC",
    "MBON": "MBON",
}

# Aristas permitidas en la primera versión: cadena principal + neuronas locales (LN).
# Se excluyen las recurrentes ORN->ORN, PN->PN, KC->KC, MBON->* y las directas ORN->KC, etc.
ALLOWED_EDGES = {
    ("ORN", "PN"), ("ORN", "LN"),
    ("LN", "ORN"), ("LN", "PN"), ("LN", "LN"),
    ("PN", "LN"), ("PN", "KC"),
    ("KC", "MBON"),
}


def load_neurons(raw: Path = RAW) -> pd.DataFrame:
    """Tabla de neuronas del subcircuito, con capa, tipo, lado y neurotransmisor."""
    cls = pd.read_csv(raw / "classification.csv.gz")
    types = pd.read_csv(raw / "consolidated_cell_types.csv.gz")[["root_id", "primary_type"]]
    nt = pd.read_csv(raw / "neurons.csv.gz")[["root_id", "nt_type", "nt_type_score"]]
    df = cls.merge(types, on="root_id", how="left").merge(nt, on="root_id", how="left")
    df["layer"] = df["class"].map(CLASS_TO_LAYER)
    return df.dropna(subset=["layer"]).reset_index(drop=True)


def load_edges(neurons: pd.DataFrame, raw: Path = RAW) -> pd.DataFrame:
    """Aristas neurona->neurona (sinapsis sumadas entre neuropilos) dentro del subcircuito."""
    cn = pd.read_csv(raw / "connections_princeton.csv.gz")
    layer = neurons.set_index("root_id")["layer"]
    cn = cn[cn.pre_root_id.isin(layer.index) & cn.post_root_id.isin(layer.index)].copy()
    cn["pre_layer"] = cn.pre_root_id.map(layer)
    cn["post_layer"] = cn.post_root_id.map(layer)
    keep = [(a, b) in ALLOWED_EDGES for a, b in zip(cn.pre_layer, cn.post_layer)]
    cn = cn[keep]
    return (
        cn.groupby(["pre_root_id", "post_root_id", "pre_layer", "post_layer"], as_index=False)
        .syn_count.sum()
    )
