"""Change proposals: the exact, quotable payload a person (or a policy) signs.

A proposal is derived *only* from a typed diagnosis through the ``REMEDY`` table, never from
free text, so nothing an agent reads in logs or alerts can widen what gets written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import REMEDY, Diagnosis, RemediationAction, ServiceKind


class RiskTier(StrEnum):
    SOFT = "soft"  # small blast radius, reversible: a versioned policy may sign
    HARD = "hard"  # many dependents: a named person signs
    CRITICAL = "critical"  # data tier or failover: a named person signs


@dataclass(frozen=True, slots=True)
class ChangeProposal:
    incident_id: str
    action: RemediationAction
    target: str
    rationale: str
    evidence: tuple[str, ...]

    def quote(self) -> str:
        """The exact change, in words a reviewer can sign."""
        return f"{self.action.value.upper()} on {self.target} (incident {self.incident_id})"

    def payload(self) -> dict[str, str]:
        return {"incident_id": self.incident_id, "action": self.action.value, "target": self.target}

    def fingerprint(self) -> str:
        """Hash of exactly what will execute. Any change to the payload voids the approval."""
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def idempotency_key(self) -> str:
        return f"{self.incident_id}:{self.fingerprint()[:16]}"


def propose(
    incident_id: str, diagnosis: Diagnosis, evidence: tuple[str, ...], confidence: float | None
) -> ChangeProposal:
    conf = f"{confidence:.2f}" if confidence is not None else "n/a"
    return ChangeProposal(
        incident_id=incident_id,
        action=REMEDY[diagnosis.fault],
        target=diagnosis.service,
        rationale=f"root cause {diagnosis} (confidence {conf})",
        evidence=evidence,
    )


def assess_risk(proposal: ChangeProposal, topology: Topology) -> RiskTier:
    """Approval tier matches the blast radius of the write."""
    kind = topology.service(proposal.target).kind
    if proposal.action is RemediationAction.FAILOVER or kind is ServiceKind.DATASTORE:
        return RiskTier.CRITICAL
    if len(topology.upstream(proposal.target)) >= 3:
        return RiskTier.HARD
    return RiskTier.SOFT
