from dataclasses import dataclass
import math
import re
from collections.abc import Sequence


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
