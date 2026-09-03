import torch
from torch import Tensor, nn

from .address import StructuralAddress


class MultiDimensionalRoPE(nn.Module):
    """Apply a composite rotary angle from sequence and structural coordinates."""

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim = dim
        self.base = base
        frequencies = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.coefficients = nn.Parameter(torch.ones(4))

    def angles(self, addresses: Tensor) -> Tensor:
        if addresses.shape[-1] != 4:
            raise ValueError("addresses must end with four coordinates: m, d, p, s")
        return (addresses * self.coefficients).sum(dim=-1, keepdim=True) * self.frequencies

    def forward(self, values: Tensor, addresses: Tensor) -> Tensor:
        if values.shape[-1] != self.dim:
            raise ValueError("values last dimension must equal dim")
        angles = self.angles(addresses.to(values))
        cosine = angles.cos()
        sine = angles.sin()
        pairs = values.reshape(*values.shape[:-1], -1, 2)
        real, imaginary = pairs.unbind(-1)
        rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1)
        return rotated.reshape_as(values)

    def with_standard_rope(self) -> None:
        with torch.no_grad():
            self.coefficients.zero_()
            self.coefficients[0] = 1.0

    @staticmethod
    def addresses(items: list[StructuralAddress], device: torch.device | None = None) -> Tensor:
        return torch.tensor([item.as_vector() for item in items], dtype=torch.float32, device=device)
