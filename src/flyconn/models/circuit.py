"""Descripción del circuito (tamaños, signos, aristas por bloque) y control de cableado barajado."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ["ORN", "PN", "LN", "KC", "MBON"]
PROCESSED = Path("data/processed")

Block = tuple[np.ndarray, np.ndarray, np.ndarray]  # (post_idx, pre_idx, nº sinapsis) en índices de capa


@dataclass
class CircuitSpec:
    sizes: dict[str, int]
    sign: dict[str, np.ndarray]                      # signo por neurona de cada capa (+1/-1)
    blocks: dict[tuple[str, str], Block] = field(default_factory=dict)

    def n_edges(self) -> int:
        return sum(len(b[0]) for b in self.blocks.values())


def load_spec(processed: Path = PROCESSED) -> CircuitSpec:
    n = pd.read_parquet(processed / "neurons.parquet")
    e = pd.read_parquet(processed / "edges.parquet")
    sizes = {l: int((n.layer == l).sum()) for l in LAYERS}
    sign = {l: n[n.layer == l].sort_values("layer_idx").sign.to_numpy() for l in LAYERS}
    blocks = {
        (a, b): (g.post_layer_idx.to_numpy(), g.pre_layer_idx.to_numpy(), g.syn_count.to_numpy(float))
        for (a, b), g in e.groupby(["pre_layer", "post_layer"])
    }
    return CircuitSpec(sizes, sign, blocks)


def shuffle_spec(spec: CircuitSpec, seed: int) -> CircuitSpec:
    """Control aleatorio: mismas neuronas, signos y nº de conexiones por bloque, cableado al azar.

    En cada bloque (capa pre -> capa post) se eligen al azar, sin repetir, tantos pares
    (pre, post) como aristas tenía el original. Los recuentos de sinapsis se reparten al azar
    entre las nuevas aristas (misma distribución de pesos). No se conserva el grado de cada neurona.
    """
    rng = np.random.default_rng(seed)
    blocks: dict[tuple[str, str], Block] = {}
    for (a, b), (post, _pre, syn) in spec.blocks.items():
        n_pre, n_post = spec.sizes[a], spec.sizes[b]
        flat = rng.choice(n_pre * n_post, size=len(post), replace=False)
        blocks[(a, b)] = (flat // n_pre, flat % n_pre, rng.permutation(syn))
    return CircuitSpec(spec.sizes, spec.sign, blocks)
