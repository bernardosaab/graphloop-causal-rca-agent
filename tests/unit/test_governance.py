from __future__ import annotations

from dataclasses import replace

import pytest

from graphloop.agents.common import Investigation, Path, Status
from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import Diagnosis, FaultType, RemediationAction
from graphloop.governance.audit import DecisionLog
from graphloop.governance.gate import FailClosedError, WriteGate
from graphloop.governance.policy import (
    Approval,
    ApprovalPolicy,
    ApprovalRequest,
    DecliningReviewer,
    SimulatedOnCall,
)
from graphloop.governance.proposals import RiskTier, assess_risk, propose
from graphloop.governance.workflow import RemediationWorkflow
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.simulator.generator import Scenario
from graphloop.tracing.tracer import EventKind, Tracer

PAY = Diagnosis("pay", FaultType.BAD_DEPLOY)


def _inv(d: Diagnosis | None, status: Status = Status.COMMITTED) -> Investigation:
    return Investigation(
        "graph", "inc-test-1", "web", status, d, (), 0.9, Path.AGENTIC, evidence_ids=("ev:1",)
    )


def test_proposal_is_derived_from_the_typed_diagnosis_only() -> None:
    p = propose("i1", PAY, ("ev:1",), 0.91)
    assert p.action is RemediationAction.ROLLBACK_DEPLOY
    assert p.quote() == "ROLLBACK_DEPLOY on pay (incident i1)"
    assert p.fingerprint() == propose("i1", PAY, (), None).fingerprint()  # payload only
    assert replace(p, target="api").fingerprint() != p.fingerprint()
    assert p.idempotency_key.startswith("i1:")


def test_risk_tiers_follow_blast_radius(chain: Topology) -> None:
    assert (
        assess_risk(propose("i", Diagnosis("db", FaultType.DEPENDENCY_OUTAGE), (), None), chain)
        is RiskTier.CRITICAL
    )
    assert assess_risk(propose("i", PAY, (), None), chain) is RiskTier.HARD  # 3 upstream
    assert (
        assess_risk(propose("i", Diagnosis("lonely", FaultType.BAD_DEPLOY), (), None), chain)
        is RiskTier.SOFT
    )


def test_gate_fails_closed(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy)
    log = DecisionLog()
    gate = WriteGate(env, log, ApprovalPolicy())
    p = propose("i1", PAY, (), None)
    with pytest.raises(FailClosedError, match="not approved"):
        gate.execute(p, Approval(False, "x", p.fingerprint()))
    with pytest.raises(FailClosedError, match="differs"):
        gate.execute(p, Approval(True, "x", "0" * 64))
    tampered = replace(p, target="api")
    with pytest.raises(FailClosedError):
        gate.execute(tampered, Approval(True, "x", p.fingerprint()))
    assert env.applied == []
    first = gate.execute(p, Approval(True, "x", p.fingerprint()))
    again = gate.execute(p, Approval(True, "x", p.fingerprint()))
    assert (first.duplicate, again.duplicate) == (False, True)
    assert env.applied == [(RemediationAction.ROLLBACK_DEPLOY, "pay")]  # exactly once
    assert gate.read_back(p) is True
    assert [r.kind for r in log.records] == [
        "refused",
        "refused",
        "refused",
        "executed",
        "duplicate",
        "read_back",
    ]


def test_audit_log_is_tamper_evident() -> None:
    log = DecisionLog()
    log.append("i1", "proposed", "v1", quote="x")
    log.append("i1", "executed", "v1", quote="x")
    assert log.verify_chain()
    log._records[0] = replace(log._records[0], data={"quote": "y"})
    assert not log.verify_chain()


def test_hard_tier_waits_for_a_named_signer(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy)
    log, tracer = DecisionLog(), Tracer()
    outcome = RemediationWorkflow(env, SimulatedOnCall(), log, tracer).run(_inv(PAY))
    assert outcome.tier is RiskTier.HARD
    assert outcome.approval is not None
    assert outcome.approval.signer == "oncall:sim-sre"
    assert outcome.executed
    assert outcome.verified
    kinds = [r.kind for r in log.for_incident("inc-test-1")]
    assert kinds == ["proposed", "approved", "executed", "read_back"]
    assert {e.name for e in tracer.of(EventKind.GOVERNANCE)} >= {"proposed", "executed"}


def test_soft_tier_is_policy_signed(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy)
    outcome = RemediationWorkflow(env, DecliningReviewer(), DecisionLog(), Tracer()).run(
        _inv(Diagnosis("lonely", FaultType.BAD_DEPLOY))
    )
    assert outcome.tier is RiskTier.SOFT
    assert outcome.approval is not None
    assert outcome.approval.signer == "policy:approval-policy/v1"
    assert outcome.executed
    assert not outcome.verified  # wrong fix: read-back says so


def test_declined_and_escalated_runs_never_write(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy)
    log = DecisionLog()
    declined = RemediationWorkflow(env, DecliningReviewer(), log, Tracer()).run(_inv(PAY))
    assert declined.refused
    assert not declined.executed
    escalated = RemediationWorkflow(env, SimulatedOnCall(), log, Tracer()).run(
        _inv(PAY, Status.ESCALATED)
    )
    assert escalated.proposal is None
    assert env.applied == []
    assert log.records[-1].kind == "escalated_to_human"


def test_approval_request_round_trip() -> None:
    req = ApprovalRequest("i", "Q", RiskTier.HARD, "f", "r", ("e",))
    assert ApprovalRequest.from_payload(req.as_payload()) == req
    assert ApprovalPolicy().requires_human(RiskTier.CRITICAL)
