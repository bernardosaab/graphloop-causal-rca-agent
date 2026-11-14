"""Bayesian belief over root-cause hypotheses.

The model is intentionally simple and fully inspectable: every finding multiplies
each hypothesis by a likelihood ratio (LR) from a small, documented table, then the
distribution is renormalised. See ``docs/adr/0006-bayesian-confidence-model.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import SIGNATURE_OF, Diagnosis, FaultType, Finding, Signal

# --- Likelihood-ratio table -------------------------------------------------------------
LR_SIGNATURE_MATCH = 12.0
LR_SIGNATURE_OTHER_FAULT_SAME_SERVICE = 0.3
LR_SIGNATURE_ELSEWHERE = 0.4
LR_HEALTHY = 0.05
LR_VICTIM = 0.15  # the service itself reports timeouts towards a callee
LR_POINTED_AT = 3.0  # hypothesis sits at or below the callee a victim points at
LR_POINTED_AWAY = 0.2
LR_SUPPORTING_CHANGE = 3.0
LR_CPU_FOR_EXHAUSTION = 6.0
LR_GENERIC_SYMPTOM = 1.5

_SUPPORTING_CHANGE: dict[Signal, FaultType] = {
    Signal.RECENT_DEPLOY: FaultType.BAD_DEPLOY,
    Signal.RECENT_CONFIG_CHANGE: FaultType.CONFIG_DRIFT,
}


def likelihood_ratio(h: Diagnosis, finding: Finding, topology: Topology) -> float:
    """How much more likely ``finding`` is if ``h`` is the root cause than if it is not."""
    sig = finding.signal
    if finding.service == h.service:
        return _same_service_lr(h, sig)
    if sig in SIGNATURE_OF:
        return LR_SIGNATURE_ELSEWHERE
    if sig is Signal.UPSTREAM_TIMEOUT and finding.target is not None:
        reachable = {finding.target} | topology.downstream(finding.target)
        return LR_POINTED_AT if h.service in reachable else LR_POINTED_AWAY
    return 1.0


def _same_service_lr(h: Diagnosis, sig: Signal) -> float:
    if sig is Signal.HEALTHY:
        return LR_HEALTHY
    if sig in SIGNATURE_OF:
        return (
            LR_SIGNATURE_MATCH
            if SIGNATURE_OF[sig] is h.fault
            else LR_SIGNATURE_OTHER_FAULT_SAME_SERVICE
        )
    if sig in _SUPPORTING_CHANGE:
        return LR_SUPPORTING_CHANGE if _SUPPORTING_CHANGE[sig] is h.fault else 1.0
    if sig is Signal.CPU_SATURATION:
        return LR_CPU_FOR_EXHAUSTION if h.fault is FaultType.RESOURCE_EXHAUSTION else 1.0
    if sig is Signal.UPSTREAM_TIMEOUT:
        return LR_VICTIM
    if sig in (Signal.ERROR_SPIKE, Signal.LATENCY_SPIKE):
        return LR_GENERIC_SYMPTOM
    return 1.0


@dataclass(frozen=True)
class BeliefState:
    """An immutable, normalised probability distribution over hypotheses.

    Stored as sorted ``(hypothesis, probability)`` pairs so it is hashable and serialises
    cleanly into LangGraph checkpoints.
    """

    entries: tuple[tuple[Diagnosis, float], ...]

    def __post_init__(self) -> None:
        # Checkpoint round-trips may turn tuples into lists; normalise to keep hashability.
        object.__setattr__(self, "entries", tuple((d, float(p)) for d, p in self.entries))

    @classmethod
    def from_weights(cls, weights: Mapping[Diagnosis, float]) -> BeliefState:
        if not weights:
            raise ValueError("at least one hypothesis is required")
        if any(w < 0 for w in weights.values()):
            raise ValueError("weights must be non-negative")
        return cls._of(_normalise(dict(weights)))

    @classmethod
    def _of(cls, posteriors: dict[Diagnosis, float]) -> BeliefState:
        return cls(tuple(sorted(posteriors.items())))

    @property
    def posteriors(self) -> dict[Diagnosis, float]:
        return dict(self.entries)

    def update(self, findings: Iterable[Finding], topology: Topology) -> BeliefState:
        weights = dict(self.posteriors)
        for finding in findings:
            for h in weights:
                weights[h] *= likelihood_ratio(h, finding, topology)
        return BeliefState._of(_normalise(weights))

    def eliminate(self, hypothesis: Diagnosis) -> BeliefState:
        weights = dict(self.posteriors)
        weights[hypothesis] = 0.0
        return BeliefState._of(_normalise(weights))

    def ranked(self) -> list[tuple[Diagnosis, float]]:
        return sorted(self.entries, key=lambda kv: (-kv[1], kv[0]))

    def top(self, k: int = 1) -> list[tuple[Diagnosis, float]]:
        return self.ranked()[:k]

    @property
    def best(self) -> tuple[Diagnosis, float]:
        return self.ranked()[0]

    def entropy(self) -> float:
        return -sum(p * math.log2(p) for _, p in self.entries if p > 0)


def _normalise(weights: dict[Diagnosis, float]) -> dict[Diagnosis, float]:
    total = sum(weights.values())
    if total <= 0 or not math.isfinite(total):
        # Every hypothesis was ruled out: fall back to uniform (maximum ignorance).
        n = len(weights)
        return dict.fromkeys(weights, 1.0 / n)
    return {h: w / total for h, w in weights.items()}
