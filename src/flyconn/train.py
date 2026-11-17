"""Entrenamiento y evaluación de los modelos con varias semillas.

Uso:  uv run python -m flyconn.train --config configs/default.yaml
Escribe una fila por (modelo, semilla, ejemplos por clase, ruido de prueba) en el CSV de salida.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from .data import odors as O
from .models import ConnectomeNet, DenseMLP, load_spec, mlp_hidden_for, shuffle_spec
from .models.circuit import CircuitSpec

N_CLASSES = 10


def build_model(name: str, spec: CircuitSpec, cfg: dict[str, Any], seed: int) -> nn.Module:
    """Crea un modelo ('connectome_A', 'connectome_B', 'random_A', 'random_B' o 'mlp')."""
    torch.manual_seed(seed)
    kw = dict(n_steps=cfg["n_steps"], kc_active_frac=cfg["kc_active_frac"])
    if name == "mlp":
        target = ConnectomeNet(spec, variant="B", **kw).n_trainable()
        return DenseMLP(spec.sizes["ORN"], mlp_hidden_for(target, spec.sizes["ORN"]), N_CLASSES)
    kind, variant = name.split("_")
    s = shuffle_spec(spec, seed) if kind == "random" else spec
    return ConnectomeNet(s, N_CLASSES, variant, **kw)


def fit(model: nn.Module, X: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], lr: float) -> float:
    """Entrena con Adam, lote completo y nº fijo de épocas. Devuelve la precisión en entrenamiento."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg["epochs"]):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return accuracy(model, X, y)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(X).argmax(1) == y).float().mean())


def run(cfg: dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    spec = load_spec()
    table = O.load_odor_table()
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in cfg["seeds"]:
        split = O.split_odors(table, seed=seed)
        tests = {
            sd: tuple(map(torch.as_tensor, O.make_test(table, split, cfg["test_samples_per_odor"], sd, seed=10_000 + seed)))
            for sd in cfg["test_noise_sd"]
        }
        for k in cfg["n_per_class"]:
            Xn, yn = O.make_fewshot_train(table, split, k, cfg["noise_sd"], seed=seed * 100 + k)
            X, y = torch.as_tensor(Xn), torch.as_tensor(yn)
            for name in cfg["models"]:
                model = build_model(name, spec, cfg, seed)
                train_acc = fit(model, X, y, cfg, cfg.get("lr_per_model", {}).get(name, cfg["lr"]))
                for sd, (Xt, yt) in tests.items():
                    rows.append(dict(model=name, seed=seed, n_per_class=k, test_noise_sd=sd,
                                     acc=accuracy(model, Xt, yt), train_acc=train_acc,
                                     n_params=model.n_trainable()))
            if verbose:
                print(f"seed {seed} k={k} listo ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="clave=valor en YAML, p. ej. lr=0.001")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    for o in a.override:
        key, val = o.split("=", 1)
        cfg[key] = yaml.safe_load(val)
    df = run(cfg)
    out = Path(cfg["out_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} filas -> {out}")


if __name__ == "__main__":
    main()
