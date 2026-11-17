"""Modelos: máscara, signos, variantes A/B, control barajado, dispersión de KC y MLP."""
import numpy as np
import pytest
import torch

from flyconn.models import ConnectomeNet, DenseMLP, MaskedLinear, load_spec, mlp_hidden_for, shuffle_spec

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/edges.parquet").exists(), reason="falta data/processed"
)


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_masked_linear_mascara_y_signos():
    # 3 neuronas pre (signos +,-,+) y 2 post; aristas (post,pre): (0,0), (0,1), (1,2)
    layer = MaskedLinear(3, 2, np.array([0, 0, 1]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]),
                         np.array([1, -1, 1]))
    w = layer.weight()
    assert w.tolist() == [[1.0, -2.0, 0.0], [0.0, 0.0, 3.0]]
    # tras un paso de gradiente fuerte los signos y los ceros se mantienen
    opt = torch.optim.SGD(layer.parameters(), lr=100.0)
    layer(torch.ones(4, 3)).sum().backward(); opt.step()
    w2 = layer.weight()
    assert (w2[w == 0] == 0).all() and (torch.sign(w2) == torch.sign(w)).all()


def test_variante_a_solo_entrena_kc_mbon_y_lectura(spec):
    m = ConnectomeNet(spec, variant="A")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"layers.KC_MBON.values", "head.weight", "head.bias"}
    assert ConnectomeNet(spec, variant="B").n_trainable() == spec.n_edges() + 83 * 10 + 10


def test_forward_forma_y_dispersion_kc(spec):
    m = ConnectomeNet(spec, variant="B", kc_active_frac=0.05)
    x = torch.rand(4, spec.sizes["ORN"])
    logits, act = m(x, return_activity=True)
    assert logits.shape == (4, 10)
    assert ((act["KC"] > 0).sum(1) <= m.k).all()
    logits.sum().backward()  # el gradiente fluye


def test_control_barajado(spec):
    sh = shuffle_spec(spec, seed=0)
    assert sh.sizes == spec.sizes
    for k in spec.blocks:
        post, pre, syn = sh.blocks[k]
        assert len(post) == len(spec.blocks[k][0])
        assert len(set(zip(post, pre))) == len(post)            # sin aristas repetidas
        assert np.isclose(syn.sum(), spec.blocks[k][2].sum())    # misma distribución de pesos
        assert post.max() < spec.sizes[k[1]] and pre.max() < spec.sizes[k[0]]
    assert not np.array_equal(sh.blocks[("PN", "KC")][1], spec.blocks[("PN", "KC")][1])
    assert ConnectomeNet(sh, variant="B").n_trainable() == ConnectomeNet(spec, variant="B").n_trainable()


def test_mlp_parametros_parecidos(spec):
    target = ConnectomeNet(spec, variant="B").n_trainable()
    mlp = DenseMLP(spec.sizes["ORN"], mlp_hidden_for(target, spec.sizes["ORN"]))
    assert abs(mlp.n_trainable() - target) / target < 0.01
