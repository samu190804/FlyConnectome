"""Capa lineal con máscara binaria (conexiones reales) y signo por neurona presináptica."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """y = x @ W^T con W[post, pre] = signo[pre] * |valor| solo en las conexiones de la máscara.

    - Solo existen parámetros para las aristas de la máscara (`values`, uno por arista), así que
      sin sinapsis => peso 0 y el número de parámetros entrenables es exactamente el de aristas.
    - El signo viene de la neurona presináptica (excitadora +1, inhibidora -1) y no cambia al
      entrenar: se aplica sobre |valor|, de modo que el entrenamiento solo mueve magnitudes.
    - `trainable=False` congela los valores (pesos fijos, variante A).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        post_idx: np.ndarray,
        pre_idx: np.ndarray,
        init_values: np.ndarray,
        pre_sign: np.ndarray,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        assert len(post_idx) == len(pre_idx) == len(init_values)
        self.n_pre, self.n_post = n_pre, n_post
        self.register_buffer("post_idx", torch.as_tensor(np.array(post_idx), dtype=torch.long))
        self.register_buffer("pre_idx", torch.as_tensor(np.array(pre_idx), dtype=torch.long))
        self.register_buffer("edge_sign", torch.as_tensor(pre_sign[pre_idx], dtype=torch.float32))
        self.values = nn.Parameter(
            torch.as_tensor(np.array(init_values), dtype=torch.float32), requires_grad=trainable
        )

    @property
    def n_edges(self) -> int:
        return int(self.values.numel())

    def weight(self) -> torch.Tensor:
        """Matriz densa (n_post, n_pre) con ceros fuera de la máscara."""
        w = torch.zeros(self.n_post, self.n_pre, device=self.values.device, dtype=self.values.dtype)
        return w.index_put((self.post_idx, self.pre_idx), self.edge_sign * self.values.abs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight())
