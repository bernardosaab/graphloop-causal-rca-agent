"""Small, dependency-free statistics for honest agent evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def __str__(self) -> str:
        return f"{self.rate:.0%} [{self.low:.0%}, {self.high:.0%}]"


def wilson(successes: int, n: int, z: float = 1.96) -> Proportion:
    """Wilson score interval: well behaved for small n and rates near 0 or 1."""
    if not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    if n == 0:
        return Proportion(0, 0, 0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Proportion(successes, n, max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class McNemar:
    """Paired comparison on the same cases. ``b``: only A succeeded; ``c``: only B succeeded."""

    b: int
    c: int
    p_value: float


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> McNemar:
    """Exact (binomial) McNemar test on paired binary outcomes."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return McNemar(only_a, only_b, 1.0)
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return McNemar(only_a, only_b, min(1.0, 2 * tail))


def pass_hat_k(outcomes_per_case: Sequence[Sequence[bool]]) -> float:
    """pass^k: share of cases where *all* k repeated attempts succeed (reliability)."""
    if not outcomes_per_case:
        return 0.0
    return sum(1 for runs in outcomes_per_case if runs and all(runs)) / len(outcomes_per_case)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
