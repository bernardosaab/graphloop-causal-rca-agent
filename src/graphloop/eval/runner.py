"""Benchmark runner: the same incident stream, the same tools, budget and model, per agent.

Every run goes through the full outer harness: investigate -> remediate (fail-closed gate)
-> read back -> learn. Agents are compared on *verified* outcomes, not on their claims.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from graphloop.agents.common import Investigation, RCAAgent, Status
from graphloop.agents.graph_agent.agent import GraphAgent
from graphloop.agents.loop_agent.agent import LoopAgent
from graphloop.domain.ccg import NetworkXCausalContextGraph
from graphloop.domain.topology import Topology
from graphloop.eval.taxonomy import classify_failure
from graphloop.governance.audit import DecisionLog
from graphloop.governance.policy import SimulatedOnCall
from graphloop.governance.workflow import RemediationOutcome, RemediationWorkflow
from graphloop.harness.gateway import ModelGateway, RunPolicy
from graphloop.llm import build_llm
from graphloop.llm.base import LLMClient
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.simulator.generator import (
    Scenario,
    ScenarioGenerator,
    generate_topology,
    hotspot_propensity,
)
from graphloop.tracing.tracer import Tracer

AGENT_NAMES = ("loop", "graph", "graph+memory")

type LLMFactory = Callable[[int], LLMClient]


@dataclass(frozen=True)
class BenchmarkConfig:
    scenarios: int = 60
    seed: int = 7
    llm: str = "simulated"
    repeats: int = 1  # >1 enables pass^k
    agents: tuple[str, ...] = AGENT_NAMES
    policy: RunPolicy = field(default_factory=RunPolicy)
    hotspots: int = 3


@dataclass(frozen=True)
class RunRecord:
    agent: str
    repeat: int
    index: int
    scenario_id: str
    root_cause: str
    depth: int
    status: str
    diagnosis: str | None
    correct: bool  # committed diagnosis equals ground truth
    resolved: bool  # read-back confirms the incident is fixed
    path: str
    tier: str | None
    tool_calls: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    wall_ms: float
    degraded: bool
    hallucinated: int
    retrieval_hit: bool | None  # ground truth in the agent's top-3 ranking (graph agents)
    failure: str | None


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    topology: Topology
    records: list[RunRecord]
    traces: dict[tuple[str, str], Tracer]  # (agent, scenario_id) for the first repeat
    memory: NetworkXCausalContextGraph | None
    audit: DecisionLog
    scenarios: list[Scenario]

    def for_agent(self, agent: str) -> list[RunRecord]:
        return [r for r in self.records if r.agent == agent]


def make_agent(name: str, topology: Topology) -> tuple[RCAAgent, NetworkXCausalContextGraph | None]:
    if name == "loop":
        return LoopAgent(), None
    if name == "graph":
        return GraphAgent(), None
    if name == "graph+memory":
        ccg = NetworkXCausalContextGraph(topology)
        return GraphAgent(ccg=ccg), ccg
    raise ValueError(f"unknown agent {name!r}; choose from {AGENT_NAMES}")


def run_benchmark(
    config: BenchmarkConfig,
    llm_factory: LLMFactory | None = None,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    factory = llm_factory or (lambda seed: build_llm(config.llm, seed=seed))
    topology = generate_topology(config.seed)
    generator = ScenarioGenerator(
        topology, hotspot_propensity(topology, config.seed, config.hotspots)
    )
    scenarios = [generator.generate(config.seed * 10_000 + i) for i in range(config.scenarios)]
    records: list[RunRecord] = []
    traces: dict[tuple[str, str], Tracer] = {}
    audit = DecisionLog()
    memory: NetworkXCausalContextGraph | None = None

    for repeat in range(config.repeats):
        for name in config.agents:
            agent, ccg = make_agent(name, topology)
            if ccg is not None and repeat == 0:
                memory = ccg
            for i, sc in enumerate(scenarios):
                tracer = Tracer()
                record = _run_one(agent, sc, i, repeat, config, factory, tracer, audit)
                records.append(record)
                if repeat == 0:
                    traces[(name, sc.scenario_id)] = tracer
            if progress:
                progress(f"{name} (repeat {repeat + 1}/{config.repeats}) done")
    return BenchmarkResult(config, topology, records, traces, memory, audit, scenarios)


def _run_one(
    agent: RCAAgent,
    scenario: Scenario,
    index: int,
    repeat: int,
    config: BenchmarkConfig,
    factory: LLMFactory,
    tracer: Tracer,
    audit: DecisionLog,
) -> RunRecord:
    env = IncidentEnvironment(scenario, budget=config.policy.max_tool_calls, at=index)
    # Paired design: every agent sees the same model seed for the same incident and repeat.
    llm = ModelGateway(factory(scenario.seed * 31 + repeat), tracer, config.policy)
    t0 = time.perf_counter()
    investigation = agent.investigate(env, llm, tracer)
    outcome = RemediationWorkflow(env, SimulatedOnCall(), audit, tracer).run(investigation)
    agent.learn(investigation, outcome.verified, at=index)
    wall_ms = (time.perf_counter() - t0) * 1000
    return _record(agent.name, repeat, index, scenario, investigation, outcome, tracer, wall_ms)


def _record(
    agent: str,
    repeat: int,
    index: int,
    sc: Scenario,
    inv: Investigation,
    outcome: RemediationOutcome,
    tracer: Tracer,
    wall_ms: float,
) -> RunRecord:
    s = tracer.summary()
    correct = inv.committed == sc.root_cause
    is_graph = inv.path.value != "loop"
    return RunRecord(
        agent=agent,
        repeat=repeat,
        index=index,
        scenario_id=sc.scenario_id,
        root_cause=str(sc.root_cause),
        depth=sc.depth,
        status=inv.status.value,
        diagnosis=str(inv.diagnosis) if inv.diagnosis else None,
        correct=correct,
        resolved=outcome.verified,
        path=inv.path.value,
        tier=outcome.tier.value if outcome.tier else None,
        tool_calls=s.tool_calls,
        llm_calls=s.llm_calls,
        prompt_tokens=s.prompt_tokens,
        completion_tokens=s.completion_tokens,
        cost_usd=s.cost_usd,
        wall_ms=wall_ms,
        degraded=s.degraded,
        hallucinated=inv.hallucinated_calls,
        retrieval_hit=(sc.root_cause in inv.ranked) if is_graph else None,
        failure=None if correct else classify_failure(sc, inv),
    )


def paired_outcomes(result: BenchmarkResult, a: str, b: str) -> tuple[list[bool], list[bool]]:
    """Resolved outcomes for two agents on the same (repeat, incident) pairs."""
    xa = {(r.repeat, r.index): r.resolved for r in result.for_agent(a)}
    xb = {(r.repeat, r.index): r.resolved for r in result.for_agent(b)}
    common = sorted(set(xa) & set(xb))
    return [xa[k] for k in common], [xb[k] for k in common]


def committed(records: Sequence[RunRecord]) -> list[RunRecord]:
    return [r for r in records if r.status == Status.COMMITTED.value]
