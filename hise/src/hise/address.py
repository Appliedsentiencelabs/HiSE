from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class StructuralAddress:
    """Unified structural coordinates for one token or content unit."""

    sequence_position: int
    depth: int = 0
    parent_id: int = 0
    sibling_index: int = 0

    @staticmethod
    def stable_parent_id(path: str) -> int:
        digest = hashlib.blake2b(path.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.sequence_position, self.depth, self.parent_id, self.sibling_index)

    def as_vector(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self.as_tuple())
