"""The write gate: executes only what was signed, exactly once, then reads the result back."""

from __future__ import annotations

from dataclasses import dataclass

from graphloop.governance.audit import DecisionLog
from graphloop.governance.policy import Approval, ApprovalPolicy
from graphloop.governance.proposals import ChangeProposal
from graphloop.simulator.environment import IncidentEnvironment


class FailClosedError(RuntimeError):
    """A write was refused. Refusal is the default; execution needs every check to pass."""


@dataclass(frozen=True)
class ExecutionReceipt:
    idempotency_key: str
    duplicate: bool


class WriteGate:
    def __init__(self, env: IncidentEnvironment, log: DecisionLog, policy: ApprovalPolicy) -> None:
        self._env = env
        self._log = log
        self._policy = policy
        self._executed: dict[str, ExecutionReceipt] = {}

    def execute(self, proposal: ChangeProposal, approval: Approval) -> ExecutionReceipt:
        inc, ver = proposal.incident_id, self._policy.version
        if not approval.approved:
            self._log.append(inc, "refused", ver, reason="not approved", signer=approval.signer)
            raise FailClosedError(f"{proposal.quote()} was not approved")
        if approval.fingerprint != proposal.fingerprint():
            self._log.append(
                inc,
                "refused",
                ver,
                reason="fingerprint mismatch",
                signed=approval.fingerprint,
                actual=proposal.fingerprint(),
            )
            raise FailClosedError("payload differs from what was signed")
        key = proposal.idempotency_key
        if key in self._executed:
            self._log.append(inc, "duplicate", ver, idempotency_key=key)
            return ExecutionReceipt(key, duplicate=True)
        self._env.apply_remediation(proposal.action, proposal.target)
        receipt = ExecutionReceipt(key, duplicate=False)
        self._executed[key] = receipt
        self._log.append(
            inc,
            "executed",
            ver,
            idempotency_key=key,
            quote=proposal.quote(),
            signer=approval.signer,
            fingerprint=approval.fingerprint,
        )
        return receipt

    def read_back(self, proposal: ChangeProposal) -> bool:
        """Score the write against the system of record, not against the agent's claim."""
        resolved = self._env.is_resolved()
        self._log.append(proposal.incident_id, "read_back", self._policy.version, resolved=resolved)
        return resolved
