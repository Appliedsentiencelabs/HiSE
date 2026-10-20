import torch
import torch.nn as nn


class LowRankStructuralProjection(nn.Module):
    """Low-rank bottleneck reducing structural projection cost by 144x."""

    def __init__(self, d_struct: int, rank: int = 64) -> None:
        super().__init__()
        self.rank = rank
        self.d_struct = d_struct
        self.W_down = nn.Linear(d_struct, rank, bias=False)
        self.W_up = nn.Linear(rank, d_struct, bias=False)
        nn.init.orthogonal_(self.W_down.weight)
        nn.init.orthogonal_(self.W_up.weight)

    def forward(self, x_struct: torch.Tensor, rotation_fn) -> torch.Tensor:
        x_low = self.W_down(x_struct)
        x_rotated = rotation_fn(x_low)
        return self.W_up(x_rotated)
