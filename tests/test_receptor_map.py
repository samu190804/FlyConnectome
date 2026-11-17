"""El mapa receptor -> glomérulo debe apuntar a tipos de ORN reales de FlyWire y a olores de Hallem."""
from pathlib import Path

import pandas as pd
import pytest

MAP = Path("configs/receptor_to_glomerulus.csv")
NEURONS = Path("data/processed/neurons.parquet")
HALLEM = Path("data/processed/hallem_carlson_s1.csv")
pytestmark = pytest.mark.skipif(
    not (NEURONS.exists() and HALLEM.exists()), reason="faltan datos procesados"
)


def test_mapa_consistente():
    m = pd.read_csv(MAP)
    assert m.receptor.is_unique and m.orn_type.is_unique
    orn_types = set(pd.read_parquet(NEURONS).query("layer == 'ORN'").primary_type)
    assert set(m.orn_type) <= orn_types
    assert set(m.receptor.str.replace("Or", "Or")) <= set(pd.read_csv(HALLEM, index_col=0).columns)
