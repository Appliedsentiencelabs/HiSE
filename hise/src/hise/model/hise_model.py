import torch
import torch.nn as nn

from ..config import HiSAConfig
from ..encoding.hisa import HiSAEncoder


class HiSEEmbeddingModel(nn.Module):
    """Simple structure-aware embedding wrapper around a frozen base embedding vector."""

    def __init__(self, embedding_dim: int = 768, config: HiSAConfig | None = None) -> None:
        super().__init__()
        self.base = nn.Identity()
        self.config = config or HiSAConfig(d_model=embedding_dim)
        self.hisa = HiSAEncoder(self.config)

    def forward(self, embeddings: torch.Tensor, structural_address: dict) -> torch.Tensor:
        if embeddings.shape[-1] != self.config.d_model:
            raise ValueError(f"Expected embedding dimension {self.config.d_model}, got {embeddings.shape[-1]}")
        return self.hisa(self.base(embeddings), structural_address)
