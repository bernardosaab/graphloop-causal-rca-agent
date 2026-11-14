from __future__ import annotations

import pytest

from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import (
    APPLICABLE_FAULTS,
    Diagnosis,
    FaultType,
    RemediationAction,
    ServiceKind,
    Signal,
    ToolName,
)
from graphloop.simulator.environment import (
    BudgetExhaustedError,
    IncidentEnvironment,
    TransientToolError,
    UnknownServiceError,
)
from graphloop.simulator.generator import (
    Scenario,
    ScenarioGenerator,
    fault_modes,
    generate_topology,
    hotspot_propensity,
)


def test_topology_generation_is_deterministic_and_connected() -> None:
    a, b = generate_topology(3), generate_topology(3)
    assert a == b
    front = a.names[0]
    assert set(a.names) - {front} <= a.downstream(front)


def test_scenarios_are_deterministic_and_well_formed(generated_topology: Topology) -> None:
    gen = ScenarioGenerator(generated_topology)
    for seed in range(40):
        sc = gen.generate(seed)
        assert sc == gen.generate(seed)
        kind = generated_topology.service(sc.root_cause.service).kind
        assert sc.root_cause.fault in APPLICABLE_FAULTS[kind]
        assert kind in (ServiceKind.SERVICE, ServiceKind.DATASTORE)
        assert sc.root_cause.service in generated_topology.downstream(sc.alert_service)
        assert sc.alert_service in sc.affected
        assert sc.depth >= 2
        if sc.root_cause.fault in (FaultType.BAD_DEPLOY, FaultType.CONFIG_DRIFT):
            assert any(c.service == sc.root_cause.service for c in sc.changes)


def test_hotspots_skew_the_fault_distribution(generated_topology: Topology) -> None:
    prop = hotspot_propensity(generated_topology, seed=1, hotspots=2)
    assert sorted(prop.values()).count(8.0) == 2
    assert set(prop) == set(fault_modes(generated_topology))


def _signals(env: IncidentEnvironment, tool: ToolName, svc: str) -> set[Signal]:
    return {f.signal for f in env.call(tool, svc).findings}


def test_telemetry_model(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy, signature_visibility=1.0)
    assert _signals(env, ToolName.GET_LOGS, "pay") == {Signal.CODE_EXCEPTION}
    assert Signal.ERROR_SPIKE in _signals(env, ToolName.GET_METRICS, "pay")
    victim = env.call(ToolName.GET_LOGS, "api").findings[0]
    assert (victim.signal, victim.target) == (Signal.UPSTREAM_TIMEOUT, "pay")
    assert _signals(env, ToolName.GET_LOGS, "inv") == {Signal.HEALTHY}
    assert _signals(env, ToolName.GET_METRICS, "lonely") == {Signal.CPU_SATURATION}
    deps = env.call(ToolName.GET_DEPENDENCIES, "api").findings
    assert {f.target for f in deps} == {"pay", "inv"}


def test_alert_carries_the_change_feed(pay_deploy: Scenario) -> None:
    alert = IncidentEnvironment(pay_deploy).alert()
    assert alert.service == "web"
    assert {f.signal for f in alert.findings[1:]} == {
        Signal.RECENT_DEPLOY,
        Signal.RECENT_CONFIG_CHANGE,
    }
    assert len(alert.changes) == 3


def test_log_sampling_can_hide_the_signature(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy, signature_visibility=0.0)
    assert _signals(env, ToolName.GET_LOGS, "pay") == {Signal.ERROR_SPIKE}


def test_budget_and_hallucinations(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy, budget=2)
    with pytest.raises(UnknownServiceError):
        env.call(ToolName.GET_LOGS, "made-up-service")
    assert env.remaining == 1  # hallucinations still cost budget
    env.call(ToolName.GET_LOGS, "web")
    with pytest.raises(BudgetExhaustedError):
        env.call(ToolName.GET_LOGS, "web")


def test_transient_failures_are_injected_on_given_attempts(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy, fail_on_attempts=frozenset({2}))
    env.call(ToolName.GET_LOGS, "web")
    with pytest.raises(TransientToolError):
        env.call(ToolName.GET_LOGS, "gw")
    env.call(ToolName.GET_LOGS, "gw")
    assert env.remaining == env.budget - 2  # the failed attempt did not consume budget


def test_read_back_reflects_reality_not_claims(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy)
    env.apply_remediation(RemediationAction.ROLLBACK_DEPLOY, "api")
    assert not env.is_resolved()
    env.apply_remediation(RemediationAction.ROLLBACK_DEPLOY, "pay")
    assert env.is_resolved()
    with pytest.raises(UnknownServiceError):
        env.apply_remediation(RemediationAction.FAILOVER, "nope")


def test_planted_instructions_stay_in_the_data(pay_deploy: Scenario) -> None:
    env = IncidentEnvironment(pay_deploy, inject_instruction=True)
    finding = env.call(ToolName.GET_LOGS, "api").findings[0]
    assert "IGNORE PREVIOUS INSTRUCTIONS" in finding.text
    assert finding.signal is Signal.UPSTREAM_TIMEOUT
    assert Diagnosis("pay", FaultType.BAD_DEPLOY) == pay_deploy.root_cause
