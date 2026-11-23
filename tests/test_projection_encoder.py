"""Tests for the row_encoder="projection" ablation (docs/design/no_col_row_attention.md).

The first test is the one that matters for everything else in the repository: with the
flag at its default, TabICL must be byte-for-byte the model it was before the flag
existed. The rest check that the ablation honors the contract ICLearning imposes on its
input.
"""

import pytest
import torch

from tabicl._model.inference_config import InferenceConfig
from tabicl._model.projection import ProjectionEncoder
from tabicl._model.tabicl import TabICL


def cpu_config() -> InferenceConfig:
    """An InferenceConfig pinned to CPU.

    The managers resolve ``device=None`` to CUDA when it is present, which would strand
    a CPU model's weights; pinning keeps these tests independent of the host.
    """
    config = InferenceConfig()
    for stage in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
        stage.device = "cpu"
        stage.use_amp = False
    return config

CONFIG = dict(
    max_classes=0,
    num_quantiles=99,
    embed_dim=32,
    col_num_blocks=1,
    col_nhead=4,
    col_num_inds=8,
    row_num_blocks=1,
    row_nhead=4,
    row_num_cls=4,
    icl_num_blocks=1,
    icl_nhead=4,
    ff_factor=2,
)


def test_default_builds_the_attention_stages_and_no_projection():
    model = TabICL(**CONFIG)
    assert model.row_encoder == "attention"
    assert hasattr(model, "col_embedder") and hasattr(model, "row_interactor")
    assert not hasattr(model, "proj_encoder")
    assert not any(k.startswith("proj_encoder.") for k in model.state_dict())


def test_projection_builds_neither_attention_stage():
    model = TabICL(**CONFIG, row_encoder="projection", proj_max_features=16)
    assert not hasattr(model, "col_embedder") and not hasattr(model, "row_interactor")
    keys = model.state_dict().keys()
    assert any(k.startswith("proj_encoder.") for k in keys)
    assert not any(k.startswith(("col_embedder.", "row_interactor.")) for k in keys)


def test_rejects_an_unknown_encoder():
    with pytest.raises(ValueError, match="row_encoder must be"):
        TabICL(**CONFIG, row_encoder="attenton")


@pytest.mark.parametrize("n_features", [4, 16])
def test_projection_forward_shape_up_to_max_features(n_features):
    """Narrower inputs are zero-padded; the output width never moves."""
    torch.manual_seed(0)
    model = TabICL(**CONFIG, row_encoder="projection", proj_max_features=16).eval()
    X = torch.randn(2, 40, n_features)
    y = torch.randn(2, 30)
    with torch.no_grad():
        out = model(X, y, inference_config=cpu_config())
    assert out.shape == (2, 10, CONFIG["num_quantiles"])


def test_too_many_features_is_refused_not_truncated():
    """Silently dropping columns would produce a plausible wrong number, so it raises."""
    enc = ProjectionEncoder(max_features=16, embed_dim=32, num_cls=4).eval()
    with torch.no_grad():
        with pytest.raises(ValueError, match="fixed input width"):
            enc(torch.randn(1, 8, 17))


def test_output_chunks_are_layer_normalized():
    """ICLearning reads 4 chunks that RowInteraction LayerNorms; the ablation must match."""
    torch.manual_seed(0)
    enc = ProjectionEncoder(max_features=16, embed_dim=32, num_cls=4).eval()
    with torch.no_grad():
        R = enc(torch.randn(2, 64, 16))
    assert R.shape == (2, 64, 128)
    chunks = R.unflatten(-1, (4, 32))
    # LayerNorm has a learned affine, so check the normalization before it is applied.
    w, b = enc.out_ln.weight, enc.out_ln.bias
    pre = (chunks - b) / w
    assert torch.allclose(pre.mean(-1), torch.zeros_like(pre.mean(-1)), atol=1e-4)
    assert torch.allclose(pre.std(-1, unbiased=False), torch.ones_like(pre.std(-1)), atol=1e-3)


def test_chunking_does_not_change_the_result():
    torch.manual_seed(0)
    X = torch.randn(1, 500, 16)
    enc = ProjectionEncoder(max_features=16, embed_dim=32, num_cls=4).eval()
    with torch.no_grad():
        whole = enc(X)
        enc.chunk_size = 64
        chunked = enc(X)
    assert torch.allclose(whole, chunked, atol=1e-6)


def test_encoder_ignores_the_target():
    """Target-agnostic by construction: y reaches the model only at the ICL stage."""
    torch.manual_seed(0)
    enc = ProjectionEncoder(max_features=16, embed_dim=32, num_cls=4).eval()
    X = torch.randn(1, 32, 16)
    with torch.no_grad():
        a = enc(X, y_train=torch.randn(1, 20))
        b = enc(X, y_train=torch.randn(1, 20) * 100)
    assert torch.equal(a, b)


def test_column_order_is_not_a_symmetry():
    """Unlike the stages it replaces, the projection is not permutation invariant.

    This is what collapses the feature-shuffle half of the ensemble, so it is asserted
    rather than left implicit.
    """
    torch.manual_seed(0)
    enc = ProjectionEncoder(max_features=8, embed_dim=32, num_cls=4).eval()
    X = torch.randn(1, 16, 8)
    with torch.no_grad():
        a = enc(X)
        b = enc(X[..., torch.randperm(8)])
    assert not torch.allclose(a, b, atol=1e-3)


def test_with_pretrained_icl_keeps_icl_weights_and_freshens_the_encoder():
    torch.manual_seed(0)
    donor = TabICL(**CONFIG)
    ckpt = {"config": {**CONFIG, "recompute": False}, "state_dict": donor.state_dict()}

    model = TabICL.with_pretrained_icl(ckpt, proj_max_features=16)
    assert model.row_encoder == "projection"
    for name, param in donor.icl_predictor.named_parameters():
        assert torch.equal(param, dict(model.icl_predictor.named_parameters())[name])
    assert not hasattr(model, "col_embedder")


def test_with_pretrained_icl_refuses_an_override_that_breaks_the_icl_shapes():
    donor = TabICL(**CONFIG)
    ckpt = {"config": {**CONFIG, "recompute": False}, "state_dict": donor.state_dict()}
    with pytest.raises(ValueError, match="pretrained ICL weights depend on"):
        TabICL.with_pretrained_icl(ckpt, embed_dim=64)


def test_with_pretrained_icl_rejects_a_checkpoint_without_icl_weights():
    donor = TabICL(**CONFIG)
    state = {k: v for k, v in donor.state_dict().items() if not k.startswith("icl_predictor.")}
    ckpt = {"config": {**CONFIG, "recompute": False}, "state_dict": state}
    with pytest.raises(ValueError, match="no 'icl_predictor"):
        TabICL.with_pretrained_icl(ckpt, proj_max_features=16)
