"""Resumen y gráficas de results/main.csv.

Uso:  uv run python -m flyconn.evaluate [--csv results/main.csv]
Genera results/summary.csv, results/learning_curve.png y results/noise_robustness.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["connectome_A", "connectome_B", "random_A", "random_B", "mlp"]
LABEL = {"connectome_A": "Conectoma A (pesos fijos)", "connectome_B": "Conectoma B (entrenable)",
         "random_A": "Aleatorio A", "random_B": "Aleatorio B", "mlp": "MLP denso"}
STYLE = {"connectome_A": ("tab:blue", "--"), "connectome_B": ("tab:blue", "-"),
         "random_A": ("tab:orange", "--"), "random_B": ("tab:orange", "-"), "mlp": ("tab:green", "-")}


def summarize(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    g = df.groupby(group).acc
    return g.agg(mean="mean", std="std", n="count").reset_index()


def _plot(df: pd.DataFrame, x: str, xlabel: str, out: Path, logx: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in [m for m in ORDER if m in set(df.model)]:
        s = summarize(df[df.model == m], [x])
        c, ls = STYLE[m]
        ax.errorbar(s[x], s["mean"], yerr=s["std"], label=LABEL[m], color=c, ls=ls, marker="o", capsize=3)
    ax.axhline(0.1, color="gray", lw=0.8, ls=":", label="azar")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("Precisión en prueba (olores no vistos)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/main.csv")
    ap.add_argument("--noise", type=float, default=10.0, help="ruido de prueba para la curva de aprendizaje")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    out = Path(a.csv).parent

    lc = df[df.test_noise_sd == a.noise]
    _plot(lc, "n_per_class", "Ejemplos de entrenamiento por clase", out / "learning_curve.png", logx=True)
    k_rob = int(df.n_per_class.max())
    _plot(df[df.n_per_class == k_rob], "test_noise_sd", f"Ruido en la prueba (espigas/s), entrenado con {k_rob} ej./clase",
          out / "noise_robustness.png")

    s = summarize(lc, ["model", "n_per_class"])
    s["n_params"] = s.model.map(df.groupby("model").n_params.first())
    s.to_csv(out / "summary.csv", index=False)
    piv = s.pivot(index="model", columns="n_per_class")
    table = pd.DataFrame({k: s[s.n_per_class == k].set_index("model").apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
                          for k in sorted(s.n_per_class.unique())}).reindex([m for m in ORDER if m in set(s.model)])
    print(table.to_string())

    # Diferencias emparejadas por semilla (misma partición de olores): ¿es consistente la ventaja?
    print("\nDiferencia emparejada por semilla (media ± desv.; n semillas), por ejemplos/clase:")
    w = lc.pivot_table(index=["seed", "n_per_class"], columns="model", values="acc")
    for a_, b_ in [("connectome_B", "random_B"), ("connectome_A", "random_A"),
                   ("connectome_B", "mlp"), ("connectome_B", "connectome_A")]:
        if a_ in w and b_ in w:
            d = (w[a_] - w[b_]).groupby("n_per_class")
            print(f"  {a_} - {b_}:", {k: f"{m:+.3f}±{sd:.3f}" for k, (m, sd) in d.agg(["mean", "std"]).iterrows()})


if __name__ == "__main__":
    main()
