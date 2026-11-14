"""The graph agent: an explicit LangGraph workflow over a Causal Context Graph, with bounded
loops embedded only where iteration pays off.

```
START -> triage -> hypothesize -> recipe --hit--------------------------------+
                                    | miss                                     |
                                    v                                          v
                              [verify loop] -> counterfactual --rejected--> [verify loop]
                                                   | consistent
                                                   v
                                     diagnose -> critique --revise--> [verify loop]
                                                   | accept / unverified (escalate)
                                                   v
                                               finalize -> END

[verify loop] = START -?-> select_probe -> run_probe -?-> (repeat | END)
```

Memory is not written during the run. The harness calls :meth:`GraphAgent.learn` only after
the remediation has been executed and read back, so causal memory is built from verified
outcomes, never from the agent's own claims.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graphloop.agents.common import Investigation, Path, Status
from graphloop.agents.graph_agent.nodes import GraphAgentNodes
from graphloop.agents.graph_agent.state import GraphAgentConfig, GraphState
from graphloop.domain.ccg import CausalContextStore, NetworkXCausalContextGraph
from graphloop.harness.durable import invoke_durably, new_checkpointer
from graphloop.llm.base import LLMClient
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.tracing.tracer import Tracer

MEMORY_WRITER = "graph-agent/learn"


def build_workflow(nodes: GraphAgentNodes) -> StateGraph[GraphState]:
    """Wire the outer workflow and the embedded verification sub-graph."""
    verify = StateGraph(GraphState)
    verify.add_node("select_probe", nodes.select_probe)
    verify.add_node("run_probe", nodes.run_probe)
    verify.add_conditional_edges(START, nodes.should_verify, {True: "select_probe", False: END})
    verify.add_edge("select_probe", "run_probe")
    verify.add_conditional_edges(
        "run_probe", nodes.should_verify, {True: "select_probe", False: END}
    )

    outer = StateGraph(GraphState)
    outer.add_node("triage", nodes.triage)
    outer.add_node("hypothesize", nodes.hypothesize)
    outer.add_node("recipe", nodes.recipe)
    outer.add_node("verify_loop", verify.compile())
    outer.add_node("counterfactual", nodes.counterfactual)
    outer.add_node("diagnose", nodes.diagnose)
    outer.add_node("critique", nodes.critique)
    outer.add_node("finalize", nodes.finalize)

    outer.add_edge(START, "triage")
    outer.add_edge("triage", "hypothesize")
    outer.add_edge("hypothesize", "recipe")
    outer.add_conditional_edges(
        "recipe", nodes.after_recipe, {"hit": "critique", "miss": "verify_loop"}
    )
    outer.add_edge("verify_loop", "counterfactual")
    outer.add_conditional_edges(
        "counterfactual",
        nodes.after_counterfactual,
        {"verify": "verify_loop", "diagnose": "diagnose"},
    )
    outer.add_edge("diagnose", "critique")
    outer.add_conditional_edges(
        "critique", nodes.after_critique, {"verify": "verify_loop", "finalize": "finalize"}
    )
    outer.add_edge("finalize", END)
    return outer


class GraphAgent:
    """Causal-Context-Graph agent.

    Pass a shared ``ccg`` to give the agent memory across incidents; leave it ``None`` to
    start every investigation from a fresh graph built from the service catalog.
    """

    def __init__(
        self,
        ccg: CausalContextStore | None = None,
        config: GraphAgentConfig | None = None,
        name: str | None = None,
    ) -> None:
        self._ccg = ccg
        self._config = config or GraphAgentConfig()
        self._name = name or ("graph+memory" if ccg is not None else "graph")

    @property
    def name(self) -> str:
        return self._name

    @property
    def ccg(self) -> CausalContextStore | None:
        return self._ccg

    def investigate(
        self, env: IncidentEnvironment, llm: LLMClient, tracer: Tracer
    ) -> Investigation:
        ccg = self._ccg if self._ccg is not None else NetworkXCausalContextGraph(env.catalog())
        nodes = GraphAgentNodes(ccg, env, llm, tracer, self._config)
        app = build_workflow(nodes).compile(checkpointer=new_checkpointer())
        final = invoke_durably(app, {}, f"{self.name}:{env.scenario.scenario_id}", tracer)
        accepted = final["verdict"] == "accept"
        return Investigation(
            agent=self.name,
            incident_id=final["incident_id"],
            symptom=final["symptom"],
            status=Status.COMMITTED if accepted else Status.ESCALATED,
            diagnosis=final.get("diagnosis"),
            ranked=tuple(d for d, _ in final["belief"].top(3)),
            confidence=final.get("confidence"),
            path=Path.COMPILED if final.get("recipe") else Path.AGENTIC,
            hallucinated_calls=final.get("hallucinated", 0),
            budget_exhausted=env.remaining <= 0,
            evidence_ids=tuple(final["evidence_ids"]),
        )

    def learn(self, investigation: Investigation, verified: bool, at: int) -> None:
        """Single writer to causal memory, fed only by outcomes scored against reality."""
        if self._ccg is None or not verified or investigation.committed is None:
            return
        self._ccg.record_cause(
            investigation.incident_id,
            investigation.committed,
            investigation.symptom,
            investigation.confidence or 0.0,
            investigation.evidence_ids,
            at=at,
            source=MEMORY_WRITER,
        )
