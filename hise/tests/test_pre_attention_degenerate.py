import torch

from hise.config import HiSAConfig
from hise.encoding.subspace import DirectSumSubspace
from hise.encoding.utils import apply_rope, rope_freqs


def test_degenerate_equals_standard_rope() -> None:
    config = HiSAConfig(d_model=768)
    model = DirectSumSubspace(config)
    x = torch.randn(2, 16, 768)
    positions = torch.arange(16).unsqueeze(0).expand(2, -1)
    addr_zero = {
        'm': positions,
        'd': torch.zeros(2, 16, dtype=torch.long),
        'p': [[[] for _ in range(16)] for _ in range(2)],
        's': torch.zeros(2, 16, dtype=torch.long),
    }
    hisa_out = model(x, addr_zero)
    rope_out = apply_rope(x[..., :config.d_m], rope_freqs(config.d_m, config.rope_base, d_model=config.d_model), positions)
    assert torch.allclose(hisa_out[..., :config.d_m], rope_out, atol=1e-6)


def test_single_sequence_input_is_supported() -> None:
    config = HiSAConfig(d_model=768)
    model = DirectSumSubspace(config)
    x = torch.randn(4, 768)
    addr = {
        'm': torch.tensor([0, 1, 2, 3], dtype=torch.long),
        'd': torch.tensor([1, 1, 2, 2], dtype=torch.long),
        'p': [[[]], [[]], [[]], [[]]],
        's': torch.tensor([0, 1, 0, 2], dtype=torch.long),
    }
    out = model(x, addr)
    assert out.shape == (4, 768)
