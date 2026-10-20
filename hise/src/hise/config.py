from dataclasses import dataclass


@dataclass
class HiSAConfig:
    """Configuration for the structural subspace decomposition."""

    d_model: int = 768
    alpha: float = 0.5
    beta: float = 0.2
    gamma: float = 0.2
    delta: float = 0.1
    rank: int = 64
    rope_base: float = 10_000.0
    lambda_d_init: float = 1.0
    lambda_p_init: float = 1.0
    lambda_s_init: float = 1.0
    max_depth: int = 20
    max_siblings: int = 100
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not (self.alpha + self.beta + self.gamma + self.delta > 0):
            raise ValueError("subspace weights must sum to a positive value")
        total = self.alpha + self.beta + self.gamma + self.delta
        self.alpha = self.alpha / total
        self.beta = self.beta / total
        self.gamma = self.gamma / total
        self.delta = self.delta / total

    @property
    def d_m(self) -> int:
        return int(round(self.d_model * self.alpha))

    @property
    def d_d(self) -> int:
        return int(round(self.d_model * self.beta))

    @property
    def d_p(self) -> int:
        return int(round(self.d_model * self.gamma))

    @property
    def d_s(self) -> int:
        return self.d_model - self.d_m - self.d_d - self.d_p

    @property
    def subspace_sizes(self) -> tuple[int, int, int, int]:
        return (self.d_m, self.d_d, self.d_p, self.d_s)
