"""Redes comparadas: conectoma (variantes A/B), control barajado y MLP denso.

Todas toman X (batch, n_ORN) con la tasa de entrada de cada ORN y devuelven logits (batch, n_clases).
"""
from __future__ import annotations

import torch
from torch import nn

from .circuit import CircuitSpec
from .masked_linear import MaskedLinear

# Bloques del lóbulo antenal que se actualizan en el bucle temporal, y las dos etapas siguientes.
AL_BLOCKS = [("LN", "ORN"), ("ORN", "PN"), ("LN", "PN"), ("ORN", "LN"), ("PN", "LN"), ("LN", "LN")]


def _init_values(spec: CircuitSpec, gain: float) -> dict[tuple[str, str], torch.Tensor]:
    """Magnitud inicial proporcional al nº de sinapsis, normalizada por neurona postsináptica.

    Para cada neurona de la capa destino, la suma de las magnitudes de TODAS sus entradas vale `gain`.
    Así se conserva la proporción de sinapsis entre fuentes (p. ej. ORN frente a LN hacia una PN).
    """
    totals: dict[str, torch.Tensor] = {l: torch.zeros(n) for l, n in spec.sizes.items()}
    for (_a, b), (post, _pre, syn) in spec.blocks.items():
        totals[b].index_add_(0, torch.as_tensor(post.copy()), torch.as_tensor(syn, dtype=torch.float32))
    out = {}
    for (a, b), (post, _pre, syn) in spec.blocks.items():
        s = torch.as_tensor(syn, dtype=torch.float32)
        out[(a, b)] = gain * s / totals[b][torch.as_tensor(post.copy())]
    return out


class ConnectomeNet(nn.Module):
    """ORN -> (lóbulo antenal recurrente, T pasos) -> PN -> KC (top-k) -> MBON -> lectura lineal.

    Variante A (`train_internal=False`): pesos fijos proporcionales a las sinapsis; solo se entrenan
    KC->MBON y la lectura. Variante B (`train_internal=True`): todos los pesos de la máscara se
    entrenan (siempre respetando máscara y signos).
    El control barajado es la misma clase construida con `shuffle_spec(spec)`.
    """

    def __init__(
        self,
        spec: CircuitSpec,
        n_classes: int = 10,
        variant: str = "B",
        n_steps: int = 3,
        kc_active_frac: float = 0.05,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        assert variant in ("A", "B")
        self.n_steps = n_steps
        self.k = max(1, round(kc_active_frac * spec.sizes["KC"]))
        init = _init_values(spec, gain)
        self.layers = nn.ModuleDict()
        for (a, b), (post, pre, _syn) in spec.blocks.items():
            trainable = variant == "B" or (a, b) == ("KC", "MBON")
            self.layers[f"{a}_{b}"] = MaskedLinear(
                spec.sizes[a], spec.sizes[b], post, pre, init[(a, b)].numpy(), spec.sign[a], trainable
            )
        self.head = nn.Linear(spec.sizes["MBON"], n_classes)
        self.n_pn, self.n_ln = spec.sizes["PN"], spec.sizes["LN"]

    def _l(self, a: str, b: str) -> MaskedLinear:
        return self.layers[f"{a}_{b}"]

    def forward(self, x: torch.Tensor, return_activity: bool = False):
        pn = x.new_zeros(x.shape[0], self.n_pn)
        ln = x.new_zeros(x.shape[0], self.n_ln)
        for _ in range(self.n_steps):  # lóbulo antenal desenrollado en el tiempo
            orn = torch.relu(x + self._l("LN", "ORN")(ln))  # inhibición presináptica de las ORN
            pn, ln = (
                torch.relu(self._l("ORN", "PN")(orn) + self._l("LN", "PN")(ln)),
                torch.relu(
                    self._l("ORN", "LN")(orn) + self._l("PN", "LN")(pn) + self._l("LN", "LN")(ln)
                ),
            )
        kc = torch.relu(self._l("PN", "KC")(pn))
        thr = kc.topk(self.k, dim=1).values[:, -1:]  # dispersión: solo las k KC más activas
        kc = kc * (kc >= thr)
        mbon = torch.relu(self._l("KC", "MBON")(kc))
        logits = self.head(mbon)
        if return_activity:
            return logits, {"ORN": orn, "PN": pn, "LN": ln, "KC": kc, "MBON": mbon}
        return logits

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenseMLP(nn.Module):
    """MLP denso con una capa oculta, con el tamaño elegido para igualar parámetros entrenables."""

    def __init__(self, n_in: int, hidden: int, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mlp_hidden_for(target_params: int, n_in: int, n_classes: int = 10) -> int:
    """Neuronas ocultas para que el MLP tenga ~`target_params` parámetros (n_in*h + h + h*C + C)."""
    return max(1, round((target_params - n_classes) / (n_in + 1 + n_classes)))
