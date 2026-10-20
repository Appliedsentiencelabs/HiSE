import torch
import torch.nn as nn

from .utils import complex_multiply, complex_power, normalize_to_unit_circle


class PathEncoder(nn.Module):
    """Composes parent-path rotations to create a distinct structural address."""

    def __init__(self, d_struct: int, max_depth: int = 32) -> None:
        super().__init__()
        self.d = d_struct // 2
        self.max_depth = max_depth
        self.r_child = nn.Parameter(torch.randn(self.d, 2) * 0.1)
        self.r_sibling = nn.Parameter(torch.randn(self.d, 2) * 0.1)
        self.r_sequence = nn.Parameter(torch.randn(self.d, 2) * 0.1)
        self.r_levels = nn.Parameter(torch.randn(max_depth, self.d, 2) * 0.1)
        self.lambda_d = nn.Parameter(torch.ones(1))
        self.lambda_p = nn.Parameter(torch.ones(1))
        self.lambda_s = nn.Parameter(torch.ones(1))
        self.last_path: list[int] | None = None

    def get_unit_relations(self) -> dict[str, torch.Tensor]:
        return {
            "child": normalize_to_unit_circle(self.r_child),
            "sibling": normalize_to_unit_circle(self.r_sibling),
            "sequence": normalize_to_unit_circle(self.r_sequence),
            "levels": normalize_to_unit_circle(self.r_levels),
        }

    def encode_path(self, sibling_path: list[int]) -> torch.Tensor:
        self.last_path = list(sibling_path)
        relation = self.get_unit_relations()
        result = torch.zeros(self.d, 2, device=self.r_child.device)
        result[..., 0] = 1.0
        for depth, sibling_idx in enumerate(sibling_path):
            level_vector = relation["levels"][depth % self.max_depth]
            step = complex_power(level_vector, sibling_idx + 1)
            result = complex_multiply(result, step)
        return result.view(-1)

    def encode_address(self, m: int, depth: int, sibling_path: list[int], sibling_idx: int) -> torch.Tensor:
        relation = self.get_unit_relations()
        sequence_rotation = complex_power(relation["sequence"], m)
        path_rotation = self.encode_path(sibling_path)
        sibling_rotation = complex_power(relation["sibling"], sibling_idx + 1)
        combined = complex_multiply(sequence_rotation.view(self.d, 2), complex_multiply(path_rotation.view(self.d, 2), sibling_rotation))
        combined = combined * self.lambda_d * self.lambda_p * self.lambda_s
        return combined.view(-1)

    def get_parent_rotation(self, child_rotation: torch.Tensor, sibling_idx: int) -> torch.Tensor:
        if self.last_path is None:
            return child_rotation
        if len(self.last_path) == 0:
            return child_rotation
        if self.last_path[-1] != sibling_idx:
            return child_rotation
        return self.encode_path(self.last_path[:-1])
