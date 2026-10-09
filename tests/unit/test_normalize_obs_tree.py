"""NormalizeObs v2: leaf path, source, any input dtype (SP2 T2.3)."""
import pytest
import torch

from colosseum.networks.normalization import NormalizeObs


def test_uint8_input_is_normalized_to_float():
    norm = NormalizeObs(shape=(2,), path=("grid",))
    norm.update(torch.tensor([[0, 10], [20, 30]], dtype=torch.uint8))
    out = norm(torch.tensor([[10, 20]], dtype=torch.uint8))
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.zeros(1, 2), atol=1e-3)
    assert norm.path == ("grid",) and norm.source == "obs"


def test_update_takes_any_leading_dims_and_checks_trailing_dims():
    norm = NormalizeObs(shape=(3,), source="global_state")
    norm.update(torch.ones(4, 5, 3))
    assert torch.allclose(norm.rms.mean, torch.ones(3), atol=1e-3)
    with pytest.raises(ValueError, match=r"trailing dims must equal \(3,\)"):
        norm.update(torch.ones(4, 2))
    with pytest.raises(ValueError, match="source"):
        NormalizeObs(shape=(3,), source="state")


def test_statistics_are_buffers():
    keys = set(NormalizeObs(shape=(2,)).state_dict())
    assert keys == {"rms.mean", "rms.var", "rms.count"}
