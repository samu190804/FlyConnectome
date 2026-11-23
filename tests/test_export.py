"""Los JSON exportados para la web son coherentes entre sí (requiere haber ejecutado flyconn.export)."""
import json
from pathlib import Path

import pytest

D = Path("web/public/data")
pytestmark = pytest.mark.skipif(not (D / "results.json").exists(), reason="ejecuta flyconn.export")


def load(name):
    return json.loads((D / name).read_text())


def test_grafo_consistente():
    g = load("graph.json"); n, e = g["nodes"], g["edges"]
    for col in ("layer", "type", "side", "sign", "layer_idx", "pos_um"):
        assert len(n[col]) == n["n"]
    assert len(e["source"]) == len(e["target"]) == len(e["syn"]) == e["n"]
    assert max(e["source"]) < n["n"] and max(e["target"]) < n["n"] and min(e["syn"]) > 0
    assert set(n["sign"]) <= {-1, 1}


def test_activaciones_coherentes_con_el_grafo():
    g, a = load("graph.json"), load("activations.json")
    size = {l: g["nodes"]["layer"].count(l) for l in g["layers"]}
    ex = a["examples"]
    assert len(ex) == 10 and len({x["class"] for x in ex}) == 10 and len({x["odor"] for x in ex}) == 10
    for x in ex:
        assert len(x["steps"]) == a["meta"]["n_steps"]
        for s in x["steps"]:
            assert all(len(s[l]) == size[l] for l in ("ORN", "PN", "LN"))
        assert len(x["MBON"]) == size["MBON"]
        assert len(x["KC"]["idx"]) <= a["meta"]["kc_active"] and max(x["KC"]["idx"]) < size["KC"]
        assert abs(sum(x["probs"]) - 1) < 0.01


def test_resultados():
    r = load("results.json")
    assert r["n_seeds"] >= 5 and set(r["models"]) == set(r["learning_curve"])
    assert all(0 <= v["mean"] <= 1 for m in r["learning_curve"].values() for v in m.values())
