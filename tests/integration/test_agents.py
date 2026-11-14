"""Trajectory-level tests: score the path and the stop conditions, not only the answer."""

from __future__ import annotations

import pytest

from graphloop.agents.common import Path, Status
from graphloop.agents.graph_agent.agent import GraphAgent
from graphloop.agents.graph_agent.state import GraphAgentConfig
from graphloop.agents.loop_agent.agent import LoopAgent
from graphloop.domain.ccg import NetworkXCausalContextGraph
from graphloop.domain.vocabulary import REMEDY
from graphloop.governance.audit import DecisionLog
from graphloop.governance.policy import SimulatedOnCall
from graphloop.governance.workflow import RemediationWorkflow
from graphloop.harness.durable import RunInterruptedError
from graphloop.harness.gateway import ModelGateway, RunPolicy
from graphloop.llm import SimulatedLLM
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.simulator.generator import Scenario, ScenarioGenerator, generate_topology
from graphloop.tracing.tracer import EventKind, Tracer


def _run(
    agent: GraphAgent | LoopAgent,
    sc: Scenario,
    policy: RunPolicy | None = None,
    **env_kwargs: object,
) -> tuple[object, Tracer, IncidentEnvironment]:
    tracer = Tracer()
    env = IncidentEnvironment(sc, signature_visibility=1.0, **env_kwargs)  # type: ignore[arg-type]
    llm = ModelGateway(SimulatedLLM(seed=0, slip_rate=0.0), tracer, policy)
    return agent.investigate(env, llm, tracer), tracer, env


def test_graph_agent_solves_the_confounded_incident(pay_deploy: Scenario) -> None:
    inv, tracer, _ = _run(GraphAgent(), pay_deploy)
    assert inv.status is Status.COMMITTED  # type: ignore[attr-defined]
    assert inv.committed == pay_deploy.root_cause  # type: ignore[attr-defined]
    probed = {e.data["service"] for e in tracer.of(EventKind.TOOL)}
    assert "lonely" not in probed  # never leaves the blast radius
    pruned = [e for e in tracer.of(EventKind.NOTE) if e.name == "counterfactual_prune"]
    assert pruned
    assert pruned[0].data["services"] == ["lonely"]


def test_loop_agent_shares_the_same_tools_and_contract(pay_deploy: Scenario) -> None:
    inv, tracer, env = _run(LoopAgent(), pay_deploy)
    assert inv.path is Path.LOOP  # type: ignore[attr-defined]
    assert tracer.summary().tool_calls == len(env.calls)
    assert tracer.summary().llm_calls >= tracer.summary().tool_calls


def test_graph_agent_escalates_instead_of_guessing(pay_deploy: Scenario) -> None:
    tracer = Tracer()
    env = IncidentEnvironment(pay_deploy, signature_visibility=0.0, budget=3)
    config = GraphAgentConfig(chain_acceptance=None)
    inv = GraphAgent(config=config).investigate(env, ModelGateway(SimulatedLLM(), tracer), tracer)
    assert inv.status is Status.ESCALATED
    assert inv.committed is None
    assert any(e.name == "escalate" for e in tracer.of(EventKind.NOTE))


def test_model_caps_degrade_loudly_but_finish(pay_deploy: Scenario) -> None:
    inv, tracer, _ = _run(GraphAgent(), pay_deploy, RunPolicy(max_llm_calls=1))
    assert tracer.summary().degraded
    assert tracer.summary().llm_calls == 1
    assert inv.diagnosis is not None  # type: ignore[attr-defined]


def test_loop_agent_escalates_when_capped(pay_deploy: Scenario) -> None:
    inv, tracer, _ = _run(LoopAgent(), pay_deploy, RunPolicy(max_llm_calls=1))
    assert inv.status is Status.ESCALATED  # type: ignore[attr-defined]
    assert tracer.of(EventKind.DEGRADED)


@pytest.mark.parametrize("agent_factory", [GraphAgent, LoopAgent])
def test_resumes_from_checkpoint_after_transient_failure(
    pay_deploy: Scenario, agent_factory: type[GraphAgent] | type[LoopAgent]
) -> None:
    clean, _, _ = _run(agent_factory(), pay_deploy)
    resumed, tracer, _ = _run(agent_factory(), pay_deploy, fail_on_attempts=frozenset({2}))
    assert resumed.diagnosis == clean.diagnosis  # type: ignore[attr-defined]
    notes = [e.name for e in tracer.of(EventKind.NOTE)]
    assert "resumed_from_checkpoint" in notes


def test_gives_up_after_bounded_resumes(pay_deploy: Scenario) -> None:
    with pytest.raises(RunInterruptedError):
        _run(GraphAgent(), pay_deploy, fail_on_attempts=frozenset({1, 2, 3, 4}))


def test_planted_instructions_cannot_widen_writes() -> None:
    topo = generate_topology(7)
    gen = ScenarioGenerator(topo)
    for seed in range(15):
        sc = gen.generate(seed)
        for agent in (GraphAgent(), LoopAgent()):
            tracer = Tracer()
            env = IncidentEnvironment(sc, inject_instruction=True)
            inv = agent.investigate(env, ModelGateway(SimulatedLLM(seed=seed), tracer), tracer)
            RemediationWorkflow(env, SimulatedOnCall(), DecisionLog(), tracer).run(inv)
            committed = inv.committed
            expected = [] if committed is None else [(REMEDY[committed.fault], committed.service)]
            assert env.applied == expected


def test_memory_compiles_repeated_incidents_into_model_free_recipes(pay_deploy: Scenario) -> None:
    ccg = NetworkXCausalContextGraph(pay_deploy.topology)
    agent = GraphAgent(ccg=ccg)
    llm_calls, paths = [], []
    for t in range(5):
        tracer = Tracer()
        env = IncidentEnvironment(pay_deploy, signature_visibility=1.0, at=t)
        inv = agent.investigate(env, ModelGateway(SimulatedLLM(), tracer), tracer)
        outcome = RemediationWorkflow(env, SimulatedOnCall(), DecisionLog(), tracer).run(inv)
        agent.learn(inv, outcome.verified, at=t)
        llm_calls.append(tracer.summary().llm_calls)
        paths.append(inv.path)
    assert paths[:3] == [Path.AGENTIC] * 3
    assert paths[3:] == [Path.COMPILED] * 2  # corroborated after three verified incidents
    assert llm_calls[-1] == 0  # no model on the normal path
    assert len(ccg.assertions()) == 5


def test_unverified_outcomes_are_never_learned(pay_deploy: Scenario) -> None:
    ccg = NetworkXCausalContextGraph(pay_deploy.topology)
    agent = GraphAgent(ccg=ccg)
    inv, _, _ = _run(agent, pay_deploy)
    agent.learn(inv, verified=False, at=0)  # type: ignore[arg-type]
    assert ccg.assertions() == []
