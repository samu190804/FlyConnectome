"""Comprobaciones de la tabla S1 de Hallem y Carlson extraída del PDF."""
from pathlib import Path

import pandas as pd
import pytest

CSV = Path("data/processed/hallem_carlson_s1.csv")
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="ejecuta flyconn.data.hallem")


def test_forma_y_sin_nulos():
    d = pd.read_csv(CSV, index_col=0)
    odors = d.drop(index="__spontaneous__")
    assert odors.shape == (110, 24)
    assert not d.isna().any().any()
    assert odors.index.is_unique
    assert (d.loc["__spontaneous__"] >= 0).all()
