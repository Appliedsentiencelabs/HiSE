from __future__ import annotations

import torch
from torch import Tensor, nn

from .address import StructuralAddress


class SGAMBias(nn.Module):
    """Shared structural attention bias over a sequence of structural addresses.

    For two items with addresses (m, d, p, s), the bias is computed as:
        w_d * |d_i - d_j| + w_p * 1[parent_i != parent_j] + w_s * |s_i - s_j|
    with negative learned weights so structurally different tokens get a lower bias term
    before softmax and thus reduced attention.
    """

    def __init__(
        self,
        *,
        depth_weight: float = -1.0,
        parent_weight: float = -1.0,
        sibling_weight: float = -1.0,
    ) -> None:
        super().__init__()
        self.depth_weight = nn.Parameter(torch.tensor(float(depth_weight)))
        self.parent_weight = nn.Parameter(torch.tensor(float(parent_weight)))
        self.sibling_weight = nn.Parameter(torch.tensor(float(sibling_weight)))

    def warm_start(self, *, depth: float = -1.0, parent: float = -1.0, sibling: float = -1.0) -> None:
        with torch.no_grad():
            self.depth_weight.fill_(float(depth))
            self.parent_weight.fill_(float(parent))
            self.sibling_weight.fill_(float(sibling))

    def forward(self, addresses: Tensor | list[StructuralAddress]) -> Tensor:
        if isinstance(addresses, list):
            tensor = torch.tensor([item.as_vector() for item in addresses], dtype=torch.float32)
        else:
            tensor = addresses.to(dtype=torch.float32)
        if tensor.shape[-1] != 4:
            raise ValueError("addresses must end with four coordinates: m, d, p, s")
        diffs = tensor[:, None, :] - tensor[None, :, :]
        depth = diffs[..., 1].abs() * self.depth_weight
        parent = (diffs[..., 2] != 0).float() * self.parent_weight
        sibling = diffs[..., 3].abs() * self.sibling_weight
        return depth + parent + sibling


class HiSEEmbeddingInjector(nn.Module):
    """Inject a learned structural bias into an existing embedding vector.

    For a chunk embedding ``e`` and structural address ``(m, d, p, s)`` we compute:
    ``e_hat = e + W_struct @ h(m, d, p, s)`` where ``h`` is a compact structural signal.
    """

    def __init__(self, embedding_dim: int, structural_dim: int = 4, *, gate: bool = False) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if structural_dim <= 0:
            raise ValueError("structural_dim must be positive")
        self.embedding_dim = embedding_dim
        self.structural_dim = structural_dim
        self.structure = nn.Linear(structural_dim, embedding_dim, bias=False)
        self.gate = nn.Linear(embedding_dim, embedding_dim, bias=True) if gate else None

    @staticmethod
    def structural_signature(addresses: Tensor | StructuralAddress | list[StructuralAddress]) -> Tensor:
        if isinstance(addresses, StructuralAddress):
            tensor = torch.tensor(addresses.as_vector(), dtype=torch.float32).unsqueeze(0)
        elif isinstance(addresses, list):
            tensor = torch.tensor([item.as_vector() for item in addresses], dtype=torch.float32)
        else:
            tensor = addresses.to(dtype=torch.float32)
        if tensor.shape[-1] != 4:
            raise ValueError("addresses must end with four coordinates: m, d, p, s")
        scalar = tensor[..., 0] + tensor[..., 1] + tensor[..., 2] + tensor[..., 3]
        return torch.stack(
            [
                torch.sin(scalar),
                torch.cos(tensor[..., 1]),
                torch.sin(tensor[..., 2]),
                torch.cos(tensor[..., 3]),
            ],
            dim=-1,
        )

    def forward(self, embeddings: Tensor, addresses: Tensor | StructuralAddress | list[StructuralAddress]) -> Tensor:
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"embeddings last dimension must equal {self.embedding_dim}")
        vector = self.structural_signature(addresses).to(device=embeddings.device, dtype=embeddings.dtype)
        bias = self.structure(vector)
        single = embeddings.dim() == 1
        if single:
            embeddings = embeddings.unsqueeze(0)
            bias = bias.unsqueeze(0) if bias.dim() == 1 else bias
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(embeddings))
            output = embeddings + gate * bias
        else:
            output = embeddings + bias
        return output[0] if single else output


def structural_distance(left: StructuralAddress | Tensor, right: StructuralAddress | Tensor) -> float:
    if isinstance(left, StructuralAddress) and isinstance(right, StructuralAddress):
        return left.distance_to(right)
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        left_vector = left.float()
        right_vector = right.float()
        return float(
            abs(left_vector[..., 0] - right_vector[..., 0]).sum()
            + abs(left_vector[..., 1] - right_vector[..., 1]).sum()
            + (left_vector[..., 2] != right_vector[..., 2]).float().sum()
            + abs(left_vector[..., 3] - right_vector[..., 3]).sum()
        )
    raise TypeError("left and right must both be StructuralAddress or Tensor instances")


def sgam_bias(addresses: Tensor | list[StructuralAddress], *, depth_weight: float = -1.0, parent_weight: float = -1.0, sibling_weight: float = -1.0) -> Tensor:
    if isinstance(addresses, list):
        tensor = torch.tensor([item.as_vector() for item in addresses], dtype=torch.float32)
    else:
        tensor = addresses.to(dtype=torch.float32)
    if tensor.shape[-1] != 4:
        raise ValueError("addresses must end with four coordinates: m, d, p, s")
    diffs = tensor[:, None, :] - tensor[None, :, :]
    depths = diffs[..., 1].abs() * depth_weight
    parents = (diffs[..., 2] != 0).float() * parent_weight
    siblings = diffs[..., 3].abs() * sibling_weight
    return depths + parents + siblings
