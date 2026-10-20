from dataclasses import dataclass
import math
import re
from collections.abc import Sequence

import torch

from .address import StructuralAddress
from .injection import SGAMBias, sgam_bias


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_id: str
    candidates: tuple[tuple[str, str], ...]


def token_vector(value: str) -> dict[str, float]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(item * item for item in counts.values())) or 1.0
    return {token: count / norm for token, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def lexical_rank(query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
    query_vector = token_vector(query)
    ranked = [(candidate_id, cosine(query_vector, token_vector(content))) for candidate_id, content in candidates]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def top_one_is_false_positive(case: RetrievalCase) -> bool:
    ranked = lexical_rank(case.query, case.candidates)
    return ranked[0][0] != case.relevant_id


def evaluate_sgam(addresses: Sequence[StructuralAddress]) -> dict[str, bool | float]:
    """Evaluate the SGAM invariants that matter for structure-aware attention."""
    tensor = sgam_bias(list(addresses))
    zero_case = [StructuralAddress(0, 0, 0, 0), StructuralAddress(0, 0, 0, 0)]
    zero_bias = bool(torch.allclose(sgam_bias(zero_case), torch.zeros_like(sgam_bias(zero_case))))

    same_structure = [
        StructuralAddress(0, 1, 10, 0),
        StructuralAddress(2, 1, 10, 0),
        StructuralAddress(3, 1, 10, 0),
    ]
    same_structure_zero = bool(sgam_bias(same_structure)[0, 1].item() == 0.0 and sgam_bias(same_structure)[0, 2].item() == 0.0)

    symmetric = bool(torch.allclose(tensor, tensor.T))
    bounded = bool(torch.isfinite(tensor).all() and tensor.max().item() <= 0.0)

    module = SGAMBias(depth_weight=-1.0, parent_weight=-2.0, sibling_weight=-0.5)
    loss = module(list(addresses)).sum()
    loss.backward()
    gradient_values = [
        module.depth_weight.grad.abs().sum().item(),
        module.parent_weight.grad.abs().sum().item(),
        module.sibling_weight.grad.abs().sum().item(),
    ]
    gradient_flow = bool(all(value > 0.0 for value in gradient_values))

    report = {
        "zero_bias": zero_bias,
        "same_structure_zero": same_structure_zero,
        "symmetric": symmetric,
        "bounded": bounded,
        "gradient_flow": gradient_flow,
        "all_pass": all([zero_bias, same_structure_zero, symmetric, bounded, gradient_flow]),
    }
    return report
