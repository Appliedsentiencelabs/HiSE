import torch
from torch import Tensor, nn

from .address import StructuralAddress


def complex_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply complex-valued tensors represented as (real, imag) pairs."""
    a_re = a[..., 0]
    a_im = a[..., 1]
    b_re = b[..., 0]
    b_im = b[..., 1]
    return torch.stack([a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re], dim=-1)


def complex_power(r: torch.Tensor, n: int) -> torch.Tensor:
    """Raise a complex rotation to integer power using repeated squaring."""
    if n < 0:
        r = torch.stack([r[..., 0], -r[..., 1]], dim=-1)
        n = -n
    result = torch.zeros_like(r)
    result[..., 0] = 1.0
    base = r
    while n > 0:
        if n % 2 == 1:
            result = complex_multiply(result, base)
        base = complex_multiply(base, base)
        n //= 2
    return result


def normalize_to_unit_circle(r: torch.Tensor) -> torch.Tensor:
    """Project complex vectors onto the unit circle."""
    norm = torch.norm(r, dim=-1, keepdim=True)
    return r / (norm + 1e-8)


def rope_freqs(d: int, base: float = 10_000.0, d_model: int | None = None) -> torch.Tensor:
    """Return RoPE frequencies for even-length vectors."""
    if d % 2:
        raise ValueError("d must be even")
    denominator = d_model if d_model is not None else d
    positions = torch.arange(0, d, 2, dtype=torch.float32)
    return 1.0 / (base ** (positions / denominator))


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Apply standard RoPE rotation to a tensor with even-sized last dimension."""
    if x.shape[-1] % 2:
        raise ValueError("x last dimension must be even")
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    pos = position.to(device=x.device, dtype=x.dtype)
    angles = pos.unsqueeze(-1) * freqs.to(device=x.device, dtype=x.dtype).view(*([1] * (pos.dim() - 1)), -1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)


class MultiDimensionalRoPE(nn.Module):
    """Apply a composite rotary angle from sequence and structural coordinates.

    Standard RoPE is recovered when the structural coordinates are all zero, or when the
    learned structural coefficients are zeroed via ``with_standard_rope``.
    """

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim = dim
        self.base = base
        frequencies = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.coefficients = nn.Parameter(torch.tensor([1.0, 0.5, 0.35, 0.25], dtype=torch.float32))

    def angles(self, addresses: Tensor) -> Tensor:
        if addresses.shape[-1] != 4:
            raise ValueError("addresses must end with four coordinates: m, d, p, s")

        sequence_term = addresses[..., 0:1] * self.coefficients[0]
        depth_term = addresses[..., 1:2] * self.coefficients[1] / 2.0
        parent_term = addresses[..., 2:3] * self.coefficients[2] / 3.0
        sibling_term = addresses[..., 3:4] * self.coefficients[3] / 5.0
        combined = sequence_term + depth_term + parent_term + sibling_term
        return combined * self.frequencies

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
