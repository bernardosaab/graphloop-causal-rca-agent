"""Approval policy (control plane): who may sign which writes. Versioned, in code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from graphloop.governance.proposals import RiskTier


@dataclass(frozen=True)
class ApprovalPolicy:
    version: str = "approval-policy/v1"
    policy_signed_tiers: frozenset[RiskTier] = frozenset({RiskTier.SOFT})

    def requires_human(self, tier: RiskTier) -> bool:
        return tier not in self.policy_signed_tiers

    @property
    def policy_signer(self) -> str:
        return f"policy:{self.version}"


@dataclass(frozen=True)
class ApprovalRequest:
    """What the reviewer sees: the quote, the evidence, and the fingerprint they sign."""

    incident_id: str
    quote: str
    tier: RiskTier
    fingerprint: str
    rationale: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "quote": self.quote,
            "tier": self.tier.value,
            "fingerprint": self.fingerprint,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> ApprovalRequest:
        return cls(
            data["incident_id"],
            data["quote"],
            RiskTier(data["tier"]),
            data["fingerprint"],
            data["rationale"],
            tuple(data["evidence"]),
        )


@dataclass(frozen=True)
class Approval:
    approved: bool
    signer: str
    fingerprint: str  # what the signer actually signed
    reason: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "signer": self.signer,
            "fingerprint": self.fingerprint,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> Approval:
        return cls(
            bool(data["approved"]),
            str(data["signer"]),
            str(data["fingerprint"]),
            str(data.get("reason", "")),
        )


class Approver(Protocol):
    def review(self, request: ApprovalRequest) -> Approval: ...


@dataclass(frozen=True)
class SimulatedOnCall:
    """A named on-call engineer used in benchmarks. Signs what it is shown.

    This deliberately models a *trusting* reviewer, so benchmark results measure the agents
    and the gate, not reviewer skill. Production would show this person the evidence chain.
    """

    name: str = "oncall:sim-sre"

    def review(self, request: ApprovalRequest) -> Approval:
        return Approval(True, self.name, request.fingerprint, "reviewed quote and evidence")


@dataclass(frozen=True)
class DecliningReviewer:
    name: str = "oncall:sim-cautious"

    def review(self, request: ApprovalRequest) -> Approval:
        return Approval(False, self.name, request.fingerprint, "declined")
