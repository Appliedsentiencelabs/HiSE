import torch
import torch.nn as nn

from ..config import HiSAConfig
from .subspace import DirectSumSubspace


class HiSAEncoder(nn.Module):
    """Concrete HiSA pre-attention encoder using direct-sum subspaces."""

    def __init__(self, config: HiSAConfig | None = None) -> None:
        super().__init__()
        self.config = config or HiSAConfig()
        self.subspace = DirectSumSubspace(self.config)

    def forward(self, x: torch.Tensor, structural_address: dict) -> torch.Tensor:
        return self.subspace(x, structural_address)
