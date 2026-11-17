"""Partición few-shot por olores y generación de muestras."""
from pathlib import Path

import numpy as np
import pytest

from flyconn.data import odors as O

pytestmark = pytest.mark.skipif(
    not (O.HALLEM_CSV.exists() and O.NEURONS_PARQUET.exists()), reason="faltan datos procesados"
)


@pytest.fixture(scope="module")
def table() -> O.OdorTable:
    return O.load_odor_table()


def test_tabla(table):
    assert len(table.class_names) == 10
    assert table.clean.shape == (110, 2281)
    assert (table.clean >= 0).all()
    assert table.has_data.sum() == 1121  # ORN con dato de Hallem


def test_train_y_test_sin_solape_y_todas_las_clases(table):
    s = O.split_odors(table, seed=3)
    assert not set(s.train_odors) & set(s.test_odors)
    for c in range(10):
        assert (table.classes[s.train_odors] == c).any()
        assert (table.classes[s.test_odors] == c).sum() >= 2


def test_fewshot_forma_y_reproducibilidad(table):
    s = O.split_odors(table, seed=0)
    for k in (1, 2, 5, 10, 20):
        X, y = O.make_fewshot_train(table, s, k, 10.0, seed=1)
        assert X.shape == (10 * k, 2281) and np.bincount(y).tolist() == [k] * 10
    X1, _ = O.make_fewshot_train(table, s, 5, 10.0, seed=1)
    X2, _ = O.make_fewshot_train(table, s, 5, 10.0, seed=1)
    assert np.array_equal(X1, X2)


def test_orn_sin_dato_solo_fondo(table):
    s = O.split_odors(table, seed=0)
    X, _ = O.make_test(table, s, 3, 0.0, seed=0)  # sin ruido
    sin_dato = X[:, ~table.has_data]
    assert np.allclose(sin_dato, sin_dato[0])  # igual para todos los olores
