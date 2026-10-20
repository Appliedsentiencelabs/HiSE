import torch

from hise.address import StructuralAddress
from hise.injection import HiSEEmbeddingInjector, sgam_bias


def test_hise_embedding_injector_modifies_content_vectors() -> None:
    injector = HiSEEmbeddingInjector(embedding_dim=8)
    base = torch.randn(3, 8)
    addresses = [
        StructuralAddress(sequence_position=7, depth=2, parent_id=10, sibling_index=1),
        StructuralAddress(sequence_position=8, depth=2, parent_id=11, sibling_index=2),
        StructuralAddress(sequence_position=9, depth=3, parent_id=10, sibling_index=0),
    ]
    output = injector(base, addresses)
    assert output.shape == base.shape
    assert not torch.allclose(output, base)


def test_sgam_bias_respects_structural_distance() -> None:
    addresses = [
        StructuralAddress(sequence_position=0, depth=1, parent_id=10, sibling_index=0),
        StructuralAddress(sequence_position=1, depth=1, parent_id=10, sibling_index=1),
        StructuralAddress(sequence_position=2, depth=2, parent_id=20, sibling_index=0),
    ]
    bias = sgam_bias([*addresses])
    assert bias.shape == (3, 3)
    assert bias[0, 0] == 0.0
    assert bias[0, 2] < bias[0, 1]
