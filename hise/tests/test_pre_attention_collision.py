import torch

from hise.encoding.path_encoder import PathEncoder


def test_distinct_paths_produce_distinct_rotations() -> None:
    encoder = PathEncoder(d_struct=154)
    rot_a = encoder.encode_path([0, 1, 0])
    rot_b = encoder.encode_path([0, 0, 1])
    assert not torch.allclose(rot_a, rot_b, atol=1e-6)
