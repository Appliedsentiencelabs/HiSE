import torch

from hise.encoding.path_encoder import PathEncoder


def test_parent_recoverable_from_child() -> None:
    encoder = PathEncoder(d_struct=154)
    parent_path = [0, 1]
    child_sibling_idx = 2
    child_path = parent_path + [child_sibling_idx]
    parent_rot = encoder.encode_path(parent_path)
    child_rot = encoder.encode_path(child_path)
    recovered = encoder.get_parent_rotation(child_rot, child_sibling_idx)
    assert torch.allclose(parent_rot, recovered, atol=1e-5)
