"""Comprobaciones básicas del subcircuito procesado (requiere data/processed/)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconn.data.circuit import ALLOWED_EDGES

P = Path("data/processed")
pytestmark = pytest.mark.skipif(not (P / "edges.parquet").exists(), reason="ejecuta flyconn.data.build")


@pytest.fixture(scope="module")
def nodes() -> pd.DataFrame:
    return pd.read_parquet(P / "neurons.parquet")


@pytest.fixture(scope="module")
def edges() -> pd.DataFrame:
    return pd.read_parquet(P / "edges.parquet")


def test_ids_unicos_e_indices_contiguos(nodes):
    assert nodes.root_id.is_unique
    assert (nodes.idx.to_numpy() == np.arange(len(nodes))).all()
    for _, g in nodes.groupby("layer", observed=True):
        assert (np.sort(g.layer_idx.to_numpy()) == np.arange(len(g))).all()


def test_aristas_solo_entre_capas_permitidas(nodes, edges):
    assert set(zip(edges.pre_layer, edges.post_layer)) <= ALLOWED_EDGES
    layer = nodes.set_index("idx").layer
    assert (edges.pre_idx.map(layer).astype(str) == edges.pre_layer).all()
    assert (edges.post_idx.map(layer).astype(str) == edges.post_layer).all()


def test_dimensiones_de_indices_por_capa(nodes, edges):
    size = nodes.groupby("layer", observed=True).size()
    for (a, b), g in edges.groupby(["pre_layer", "post_layer"]):
        assert g.pre_layer_idx.max() < size[a] and g.post_layer_idx.max() < size[b]
        assert not g.duplicated(["pre_idx", "post_idx"]).any()


def test_signos_y_pesos(nodes, edges):
    assert set(nodes.sign.unique()) <= {-1, 1}
    assert (nodes[nodes.layer.isin(["ORN", "KC"])].sign == 1).all()
    assert (edges.syn_count > 0).all()
    assert (edges.sign == edges.pre_idx.map(nodes.set_index("idx").sign)).all()


def test_sin_huerfanas_y_con_posicion(nodes, edges):
    post = set(edges.post_idx)
    assert nodes[(nodes.layer != "ORN") & ~nodes.idx.isin(post)].empty
    assert nodes[["x_um", "y_um", "z_um"]].notna().all().all()


def test_posiciones_en_micras_de_cerebro_real(nodes):
    # El subcircuito (lóbulos antenales + cuerpos fungiformes) ocupa la parte central de un cerebro
    # de ~700 x 290 x 235 µm. Si la unidad fuera incorrecta, esto saldría 4-40 veces mayor.
    q = nodes[["x_um", "y_um", "z_um"]]
    span = q.quantile(0.99) - q.quantile(0.01)
    assert ((span > 100) & (span < 700)).all(), span.to_dict()
