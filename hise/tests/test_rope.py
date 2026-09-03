import torch

from hise.address import StructuralAddress
from hise.rope import MultiDimensionalRoPE


def test_structural_coordinates_change_rotated_representation() -> None:
    encoder = MultiDimensionalRoPE(dim=8)
    values = torch.ones(1, 8)
    performance = encoder.addresses([StructuralAddress(5, 2, 10, 0)])
    risks = encoder.addresses([StructuralAddress(5, 2, 20, 0)])
    assert not torch.allclose(encoder(values, performance), encoder(values, risks))


def test_degenerate_case_matches_standard_rope() -> None:
    encoder = MultiDimensionalRoPE(dim=8)
    values = torch.randn(2, 8)
    addresses = torch.tensor([[3.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]])
    encoder.with_standard_rope()
    actual = encoder(values, addresses)
    expected = MultiDimensionalRoPE(dim=8)
    expected.with_standard_rope()
    assert torch.allclose(actual, expected(values, addresses))


def test_encoder_preserves_vector_norm() -> None:
    encoder = MultiDimensionalRoPE(dim=8)
    values = torch.randn(3, 8)
    addresses = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3)
    assert torch.allclose(values.norm(dim=-1), encoder(values, addresses).norm(dim=-1), atol=1e-5)
