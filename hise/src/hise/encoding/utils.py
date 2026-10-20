import torch


def complex_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply complex-valued tensors represented as (real, imag) pairs."""
    a_re = a[..., 0]
    a_im = a[..., 1]
    b_re = b[..., 0]
    b_im = b[..., 1]
    return torch.stack(
        [
            a_re * b_re - a_im * b_im,
            a_re * b_im + a_im * b_re,
        ],
        dim=-1,
    )


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
    """Return RoPE frequency table for even-length vectors."""
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
    rotated = torch.cat(
        [
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ],
        dim=-1,
    )
    return rotated
