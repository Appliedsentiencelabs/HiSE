import torch
import torch.nn as nn

from ..config import HiSAConfig
from .path_encoder import PathEncoder
from .utils import apply_rope, rope_freqs


class DirectSumSubspace(nn.Module):
    """Apply HiSA across four independent structural subspaces."""

    def __init__(self, config: HiSAConfig) -> None:
        super().__init__()
        self.config = config
        self.m_start, self.m_end = 0, config.d_m
        self.d_start, self.d_end = config.d_m, config.d_m + config.d_d
        self.p_start, self.p_end = config.d_m + config.d_d, config.d_m + config.d_d + config.d_p
        self.s_start, self.s_end = config.d_m + config.d_d + config.d_p, config.d_model
        self.register_buffer("freqs_m", rope_freqs(config.d_m, config.rope_base, d_model=config.d_model))
        self.register_buffer("freqs_d", rope_freqs(config.d_d, config.rope_base, d_model=config.d_model))
        self.register_buffer("freqs_s", rope_freqs(config.d_s, config.rope_base, d_model=config.d_model))
        self.path_encoder = PathEncoder(config.d_p)
        self.lambda_d = nn.Parameter(torch.ones(1) * config.lambda_d_init)
        self.lambda_s = nn.Parameter(torch.ones(1) * config.lambda_s_init)

    def forward(self, x: torch.Tensor, structural_address: dict) -> torch.Tensor:
        original_2d = x.dim() == 2
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"Expected embeddings to be 2D or 3D, got {tuple(x.shape)}")

        x_out = x.clone()
        batch_size = x.shape[0]
        positions = structural_address.get("m", torch.zeros(x.shape[:2], dtype=torch.long))
        depths = structural_address.get("d", torch.zeros(x.shape[:2], dtype=torch.long))
        paths = structural_address.get("p", [[[] for _ in range(x.shape[1])] for _ in range(batch_size)])
        siblings = structural_address.get("s", torch.zeros(x.shape[:2], dtype=torch.long))

        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        if depths.dim() == 1:
            depths = depths.unsqueeze(0).expand(batch_size, -1)
        if siblings.dim() == 1:
            siblings = siblings.unsqueeze(0).expand(batch_size, -1)
        if batch_size == 1 and isinstance(paths, list) and len(paths) == x.shape[1] and paths and isinstance(paths[0], list) and paths[0] and isinstance(paths[0][0], list):
            paths = [paths]

        if x.shape[-1] != self.config.d_model:
            raise ValueError(f"Expected x last dimension {self.config.d_model}, got {x.shape[-1]}")

        x_m = x[..., self.m_start:self.m_end]
        x_out[..., self.m_start:self.m_end] = apply_rope(x_m, self.freqs_m, positions)

        x_d = x[..., self.d_start:self.d_end]
        x_out[..., self.d_start:self.d_end] = apply_rope(x_d, self.freqs_d, (depths * self.lambda_d).long())

        x_p = x[..., self.p_start:self.p_end]
        x_out[..., self.p_start:self.p_end] = self._apply_path_rotation(x_p, paths)

        x_s = x[..., self.s_start:self.s_end]
        x_out[..., self.s_start:self.s_end] = apply_rope(x_s, self.freqs_s, (siblings * self.lambda_s).long())
        if original_2d:
            return x_out[0]
        return x_out

    def _apply_path_rotation(self, x_p: torch.Tensor, paths: list) -> torch.Tensor:
        x_out = x_p.clone()
        for b in range(x_p.shape[0]):
            for t in range(x_p.shape[1]):
                path = paths[b][t]
                rotation = self.path_encoder.encode_path(path)
                x_out[b, t] = self._rotate_by(x_p[b, t], rotation)
        return x_out

    def _rotate_by(self, x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        d = x.shape[0] // 2
        x_complex = x.view(d, 2)
        from .utils import complex_multiply

        rotated = complex_multiply(x_complex, rotation.view(d, 2))
        return rotated.view(-1)
