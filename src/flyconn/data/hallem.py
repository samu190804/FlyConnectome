"""Tabla S1 de Hallem y Carlson (2006): respuestas de 24 receptores a 110 olores.

La tabla viene en el PDF suplementario (mmc1.pdf). Se extrae el texto con `pdftotext -layout`
(paquete poppler-utils) y se guarda como CSV en data/processed/hallem_carlson_s1.csv.

Uso:  uv run python -m flyconn.data.hallem
Valores: espigas/s (respuesta menos actividad espontánea; pueden ser negativos = inhibición).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

PDF = Path("data/raw/hallem_carlson/mmc1.pdf")
OUT = Path("data/processed/hallem_carlson_s1.csv")
N_RECEPTORS = 24
_HEADER = re.compile(r"^\s*2a\s+7a\s+9a\s+10a\s")  # fila de cabecera de las tablas


def parse_table_s1(pdf: Path = PDF) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (respuestas[olor x receptor], actividad espontánea por receptor)."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    start = next(i for i, l in enumerate(text) if "Table S1" in l and i > 100)
    header_i = next(i for i in range(start, len(text)) if _HEADER.match(text[i]))
    receptors = ["Or" + t for t in text[header_i].split()]
    assert len(receptors) == N_RECEPTORS, receptors

    rows: dict[str, list[float]] = {}
    spont: list[float] | None = None
    for line in text[header_i + 1:]:
        if "Table S2" in line:  # fin de la tabla S1
            break
        tok = line.split()
        if len(tok) < N_RECEPTORS + 1:
            continue
        try:
            vals = [float(t) for t in tok[-N_RECEPTORS:]]
        except ValueError:
            continue
        name = " ".join(tok[:-N_RECEPTORS])
        # El primer valor puede quedar pegado al nombre ("ammonium hydroxide 3"):
        # con 24 números al final el nombre ya queda limpio.
        if name.startswith("spontaneous firing rate"):
            spont = vals
        elif _is_new_name(name, rows):
            rows[name] = vals
    assert spont is not None
    df = pd.DataFrame.from_dict(rows, orient="index", columns=receptors)
    df.index.name = "odor"
    return df, pd.Series(spont, index=receptors, name="spontaneous")


def _is_new_name(name: str, rows: dict) -> bool:
    return bool(name) and name not in rows


def main() -> None:
    df, spont = parse_table_s1()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([df, spont.to_frame().T.rename(index={"spontaneous": "__spontaneous__"})])
    out.to_csv(OUT)
    print(f"{df.shape[0]} olores x {df.shape[1]} receptores -> {OUT}")


if __name__ == "__main__":
    main()
