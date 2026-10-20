import torch

from hise.address import StructuralAddress
from hise.injection import SGAMBias, sgam_bias


def test_zero_structural_address_produces_zero_bias() -> None:
    zero_bias = sgam_bias([
        StructuralAddress(0, 0, 0, 0),
        StructuralAddress(0, 0, 0, 0),
    ])
    assert zero_bias.shape == (2, 2)
    assert torch.allclose(zero_bias, torch.zeros_like(zero_bias))


def test_same_structure_has_zero_bias() -> None:
    addresses = [
        StructuralAddress(0, 1, 10, 0),
        StructuralAddress(2, 1, 10, 0),
        StructuralAddress(3, 1, 10, 0),
    ]
    bias = sgam_bias(addresses)
    assert bias[0, 0] == 0.0
    assert bias[0, 1] == 0.0
    assert bias[1, 2] == 0.0


def test_sgam_bias_is_symmetric() -> None:
    addresses = [
        StructuralAddress(0, 1, 10, 0),
        StructuralAddress(1, 1, 10, 1),
        StructuralAddress(2, 2, 20, 0),
    ]
    bias = sgam_bias(addresses)
    assert torch.allclose(bias, bias.T)


def test_sgam_bias_is_bounded() -> None:
    addresses = [
        StructuralAddress(0, 0, 0, 0),
        StructuralAddress(1, 5, 99, 7),
        StructuralAddress(9, 4, 50, 3),
    ]
    bias = sgam_bias(addresses)
    assert torch.isfinite(bias).all()
    assert bias.max().item() <= 0.0


def test_sgam_module_keeps_learnable_weights() -> None:
    module = SGAMBias(depth_weight=0.5, parent_weight=1.5, sibling_weight=2.0)
    addresses = [
        StructuralAddress(0, 1, 10, 0),
        StructuralAddress(1, 1, 11, 1),
    ]
    bias = module(addresses)
    assert bias.shape == (2, 2)
    assert module.depth_weight.grad is None
    assert module.parent_weight.requires_grad
    assert module.sibling_weight.requires_grad
