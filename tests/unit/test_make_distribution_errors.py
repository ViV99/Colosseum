"""make_distribution names the action group for every parameter/shape mismatch (SP2 T2.2 ruling)."""
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.dist import DiagGaussianDist, make_distribution

SPEC = ActionSpec.from_space(Dict([
    ("move", Discrete(3)),
    ("aim", Box(-1.0, 1.0, (2,))),
    ("workers", Units(3, Dict([("kind", Discrete(2)), ("thrust", Box(-1.0, 1.0, (2,)))]))),
]))


def _params(**override):
    params = {"move": torch.zeros(1, 3), "aim": {"mean": torch.zeros(1, 2), "log_std": torch.zeros(2)},
              "workers": {"kind": torch.zeros(1, 3, 2),
                          "thrust": {"mean": torch.zeros(1, 3, 2), "log_std": torch.zeros(2)}}}
    params.update(override)
    return params


def test_valid_params_build():
    assert make_distribution(SPEC, _params()).num_deciders == 1 + 3


@pytest.mark.parametrize(("override", "match"), [
    ({"move": torch.zeros(3)}, r"action group move: CategoricalDist: logits must be \[B, n\]"),
    ({"aim": {"mean": torch.zeros(1, 2), "log_std": torch.zeros(3)}}, r"action group aim: .*log_std"),
    ({"aim": {"log_std": torch.zeros(2)}}, r"action group aim: .*'mean'"),
    ({"aim": torch.zeros(1, 2)}, r"action group aim: .*'mean'.*got Tensor"),
    ({"workers": {"kind": torch.zeros(1, 3, 3), "thrust": {"mean": torch.zeros(1, 3, 2),
                                                           "log_std": torch.zeros(2)}}},
     r"action group workers: UnitsDist: component 'kind' needs logits \[B, 3, 2\]"),
    ({"workers": {"kind": torch.zeros(1, 3, 2), "thrust": {"mean": torch.zeros(1, 3, 2),
                                                           "log_std": torch.zeros(3)}}},
     r"action group workers: .*'thrust'.*log_std"),
], ids=["1d-logits", "box-log-std-shape", "box-missing-mean", "box-not-a-dict", "units-logits-shape",
        "units-box-log-std-shape"])
def test_parameter_mismatch_names_the_group(override, match):
    with pytest.raises(ValueError, match=match):
        make_distribution(SPEC, _params(**override))


def test_gaussian_checks_the_log_std_shape():
    with pytest.raises(ValueError, match=r"log_std must be \[B, d\] or \[d\]"):
        DiagGaussianDist(torch.zeros(3, 2), torch.zeros(3))
    assert DiagGaussianDist(torch.zeros(3, 2), torch.zeros(2)).log_std.shape == (3, 2)
    assert DiagGaussianDist(torch.zeros(3, 2), torch.zeros(3, 2)).log_std.shape == (3, 2)
