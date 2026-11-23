"""Exporta los JSON que consume la web (web/public/data/).

Uso:  uv run python -m flyconn.export [--results results/all.csv] [--out web/public/data]
Genera:
  graph.json        nodos (capa, tipo, lado, signo, posición 3D) y aristas (sinapsis) del circuito
  activations.json  activaciones por capa y paso de tiempo para 10 olores de ejemplo (1 por clase)
  results.json      curvas comparativas de los modelos (de results/all.csv)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import odors as O
from .evaluate import LABEL, ORDER
from .models import load_spec
from .train import build_model, fit

LAYERS = ["ORN", "PN", "LN", "KC", "MBON"]
SOURCES = {
    "connectome": "FlyWire FAFB v783 (Dorkenwald et al., 2024; Schlegel et al., 2024), CC BY-NC 4.0",
    "odors": "Hallem y Carlson, 2006, Cell",
    "receptor_map": "Couto, Alenius y Dickson, 2005, Current Biology",
    "note": "Experimento de arquitectura inspirado en el conectoma; no es una simulación fiel de la mosca.",
}


def _dump(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    print(f"{path}  ({path.stat().st_size / 1e6:.2f} MB)")


def export_graph(out: Path, processed: Path = Path("data/processed")) -> None:
    n = pd.read_parquet(processed / "neurons.parquet")
    e = pd.read_parquet(processed / "edges.parquet")
    pos = n[["x_um", "y_um", "z_um"]].to_numpy()
    pos = pos - pos.mean(0)
    pos[:, 1] *= -1  # en FlyWire el eje y crece hacia abajo; así "arriba" queda arriba
    nodes = {
        "n": len(n),
        "layer": n.layer.astype(str).tolist(),
        "type": n.primary_type.fillna("").tolist(),
        "side": n.side.fillna("").tolist(),
        "sign": n.sign.tolist(),
        "layer_idx": n.layer_idx.tolist(),
        "pos_um": np.round(pos, 1).tolist(),
    }
    edges = {  # índices globales de nodo; el signo es el de la neurona presináptica
        "n": len(e), "source": e.pre_idx.tolist(), "target": e.post_idx.tolist(), "syn": e.syn_count.astype(int).tolist(),
    }
    _dump({"sources": SOURCES, "layers": LAYERS, "nodes": nodes, "edges": edges}, out / "graph.json")


def export_activations(out: Path, cfg: dict, k: int = 20, seed: int = 0) -> dict:
    """Entrena la red del conectoma (variante B) y guarda su actividad para 1 olor de prueba por clase."""
    spec, table = load_spec(), O.load_odor_table()
    split = O.split_odors(table, seed=seed)
    Xn, yn = O.make_fewshot_train(table, split, k, cfg["noise_sd"], seed=seed * 100 + k)
    model = build_model("connectome_B", spec, cfg, seed)
    train_acc = fit(model, torch.as_tensor(Xn), torch.as_tensor(yn), cfg, cfg["lr"])
    model.eval()

    examples = []
    for c, cname in enumerate(table.class_names):  # primer olor de prueba (alfabético) de cada clase
        idx = sorted([i for i in split.test_odors if table.classes[i] == c], key=lambda i: table.names[i])[0]
        x = torch.as_tensor(table.clean[idx : idx + 1] / O.RATE_SCALE, dtype=torch.float32)  # sin ruido
        with torch.no_grad():
            logits, act = model(x, return_activity=True)
        probs = torch.softmax(logits, 1)[0]
        kc = act["KC"][0]
        active = torch.nonzero(kc > 0).flatten()
        examples.append({
            "odor": table.names[idx], "class": cname, "pred": table.class_names[int(probs.argmax())],
            "probs": np.round(probs.numpy(), 3).tolist(),
            "steps": [{l: np.round(s[l][0].numpy(), 3).tolist() for l in ("ORN", "PN", "LN")} for s in act["steps"]],
            "KC": {"idx": active.tolist(), "val": np.round(kc[active].numpy(), 3).tolist()},  # solo KC activas
            "MBON": np.round(act["MBON"][0].numpy(), 3).tolist(),
        })
    meta = {
        "model": "connectome_B", "seed": seed, "examples_per_class": k, "train_acc": train_acc,
        "n_steps": cfg["n_steps"], "kc_active": model.k, "classes": table.class_names,
        "input_note": "Respuesta de cada ORN al olor sin ruido, en espigas/s dividido entre 100; los olores son de prueba (no vistos al entrenar).",
    }
    _dump({"meta": meta, "examples": examples}, out / "activations.json")
    return meta


def export_results(out: Path, csv: Path, noise: float = 10.0) -> None:
    df = pd.read_csv(csv)
    models = [m for m in ORDER if m in set(df.model)]
    stat = lambda d: {"mean": round(float(d.acc.mean()), 4), "std": round(float(d.acc.std()), 4)}
    curve = {m: {int(k): stat(g) for k, g in df[(df.model == m) & (df.test_noise_sd == noise)].groupby("n_per_class")} for m in models}
    kmax = int(df.n_per_class.max())
    rob = {m: {float(s): stat(g) for s, g in df[(df.model == m) & (df.n_per_class == kmax)].groupby("test_noise_sd")} for m in models}
    w = df[df.test_noise_sd == noise].pivot_table(index=["seed", "n_per_class"], columns="model", values="acc")
    pairs = {}
    for a, b in [("connectome_B", "random_B"), ("connectome_A", "random_A"), ("connectome_B", "mlp"), ("connectome_B", "connectome_A")]:
        d = (w[a] - w[b]).groupby("n_per_class").agg(["mean", "std"])
        pairs[f"{a}-{b}"] = {int(k): {"mean": round(float(r["mean"]), 4), "std": round(float(r["std"]), 4)} for k, r in d.iterrows()}
    _dump({
        "models": models, "labels": {m: LABEL[m] for m in models}, "n_seeds": int(df.seed.nunique()),
        "n_params": {m: int(df[df.model == m].n_params.iloc[0]) for m in models},
        "chance": 0.1, "test_noise_for_curve": noise, "robustness_examples_per_class": kmax,
        "learning_curve": curve, "noise_robustness": rob, "paired_differences": pairs,
        "caveats": [
            "La mitad de las ORN (30 de 53 tipos) no tiene datos de Hallem y recibe solo actividad de fondo.",
            "Conjunto de prueba: 33 olores; diferencias de pocos puntos están dentro del ruido.",
            "El control aleatorio conserva el nº de conexiones por bloque y los signos, no el de cada neurona.",
        ],
    }, out / "results.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results", default="results/all.csv")
    ap.add_argument("--out", default="web/public/data")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text())
    export_graph(out)
    meta = export_activations(out, cfg)
    print("entrenamiento: precisión", meta["train_acc"])
    export_results(out, Path(a.results))


if __name__ == "__main__":
    main()
