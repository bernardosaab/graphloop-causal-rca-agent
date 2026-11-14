"""Remediation workflow (outer harness): propose -> assess -> gate -> execute -> read back.

It is shared by every agent: agents only *diagnose*; this code decides what may be written.
High-risk writes pause on a LangGraph ``interrupt`` until a named person signs; the run is
checkpointed, so waiting for a person is a state, not a blocked thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from graphloop.agents.common import Investigation
from graphloop.governance.audit import DecisionLog
from graphloop.governance.gate import FailClosedError, WriteGate
from graphloop.governance.policy import Approval, ApprovalPolicy, ApprovalRequest, Approver
from graphloop.governance.proposals import ChangeProposal, RiskTier, assess_risk, propose
from graphloop.harness.durable import new_checkpointer
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.tracing.tracer import EventKind, Tracer


class RemediationState(TypedDict, total=False):
    proposal: ChangeProposal
    tier: RiskTier
    approval: Approval
    executed: bool
    verified: bool
    refused: str


@dataclass(frozen=True)
class RemediationOutcome:
    proposal: ChangeProposal | None
    tier: RiskTier | None
    approval: Approval | None
    executed: bool
    verified: bool  # the system of record confirms the incident is resolved
    refused: str | None = None


ESCALATED_OUTCOME = RemediationOutcome(None, None, None, executed=False, verified=False)


class RemediationWorkflow:
    def __init__(
        self,
        env: IncidentEnvironment,
        approver: Approver,
        log: DecisionLog,
        tracer: Tracer,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        self._env = env
        self._approver = approver
        self._log = log
        self._tracer = tracer
        self._policy = policy or ApprovalPolicy()
        self._gate = WriteGate(env, log, self._policy)

    def run(self, investigation: Investigation) -> RemediationOutcome:
        diagnosis = investigation.committed
        if diagnosis is None:
            self._log.append(
                investigation.incident_id,
                "escalated_to_human",
                self._policy.version,
                suggestion=str(investigation.diagnosis),
            )
            return ESCALATED_OUTCOME
        proposal = propose(
            investigation.incident_id,
            diagnosis,
            investigation.evidence_ids,
            investigation.confidence,
        )
        app = self._build().compile(checkpointer=new_checkpointer())
        config: RunnableConfig = {
            "configurable": {"thread_id": f"remediate:{investigation.incident_id}"}
        }
        initial: RemediationState = {"proposal": proposal}
        result: dict[str, Any] = app.invoke(initial, config)
        while "__interrupt__" in result:  # a named person must sign
            request = ApprovalRequest.from_payload(result["__interrupt__"][0].value)
            approval = self._approver.review(request)
            resume: Command[Any] = Command(resume=approval.as_payload())
            result = app.invoke(resume, config)
        return RemediationOutcome(
            proposal=result["proposal"],
            tier=result.get("tier"),
            approval=result.get("approval"),
            executed=result.get("executed", False),
            verified=result.get("verified", False),
            refused=result.get("refused"),
        )

    # ---- nodes ----------------------------------------------------------------------------
    def _assess(self, state: RemediationState) -> dict[str, Any]:
        proposal = state["proposal"]
        tier = assess_risk(proposal, self._env.catalog())
        self._log.append(
            proposal.incident_id,
            "proposed",
            self._policy.version,
            quote=proposal.quote(),
            tier=tier.value,
            fingerprint=proposal.fingerprint(),
            rationale=proposal.rationale,
        )
        self._tracer.emit(EventKind.GOVERNANCE, "proposed", quote=proposal.quote(), tier=tier.value)
        return {"tier": tier}

    def _gate_node(self, state: RemediationState) -> dict[str, Any]:
        proposal, tier = state["proposal"], state["tier"]
        if not self._policy.requires_human(tier):
            approval = Approval(
                True,
                self._policy.policy_signer,
                proposal.fingerprint(),
                f"{tier.value} tier is policy-signed",
            )
        else:
            request = ApprovalRequest(
                proposal.incident_id,
                proposal.quote(),
                tier,
                proposal.fingerprint(),
                proposal.rationale,
                proposal.evidence,
            )
            approval = Approval.from_payload(interrupt(request.as_payload()))
        kind = "approved" if approval.approved else "declined"
        self._log.append(
            proposal.incident_id,
            kind,
            self._policy.version,
            signer=approval.signer,
            fingerprint=approval.fingerprint,
        )
        self._tracer.emit(EventKind.GOVERNANCE, kind, signer=approval.signer, tier=tier.value)
        return {"approval": approval}

    def _execute(self, state: RemediationState) -> dict[str, Any]:
        try:
            self._gate.execute(state["proposal"], state["approval"])
        except FailClosedError as exc:
            self._tracer.emit(EventKind.GOVERNANCE, "refused", reason=str(exc))
            return {"executed": False, "verified": False, "refused": str(exc)}
        self._tracer.emit(EventKind.GOVERNANCE, "executed", quote=state["proposal"].quote())
        return {"executed": True}

    def _read_back(self, state: RemediationState) -> dict[str, Any]:
        if not state.get("executed"):
            return {"verified": False}
        verified = self._gate.read_back(state["proposal"])
        self._tracer.emit(EventKind.GOVERNANCE, "read_back", resolved=verified)
        return {"verified": verified}

    def _build(self) -> StateGraph[RemediationState]:
        g = StateGraph(RemediationState)
        g.add_node("assess", self._assess)
        g.add_node("gate", self._gate_node)
        g.add_node("execute", self._execute)
        g.add_node("read_back", self._read_back)
        g.add_edge(START, "assess")
        g.add_edge("assess", "gate")
        g.add_edge("gate", "execute")
        g.add_edge("execute", "read_back")
        g.add_edge("read_back", END)
        return g
