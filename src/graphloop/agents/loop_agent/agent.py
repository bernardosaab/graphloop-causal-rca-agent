"""Baseline: a classic ReAct-style tool loop (reason -> act -> observe -> repeat).

This is the "loop engineering" reference point. The LLM owns all control flow: at every
step it sees the full, growing transcript of observations and picks the next tool call or
a final answer. It gets exactly the same alert, service catalog, tools, budget and model as
the graph agent; only the engineering around the model differs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from graphloop.agents.common import (
    Investigation,
    Path,
    Status,
    call_tool,
    decide_or_fallback,
    finding_to_fact,
)
from graphloop.domain.vocabulary import SERVICE_PROBES, Diagnosis, FaultType, ToolName
from graphloop.harness.durable import invoke_durably, new_checkpointer
from graphloop.llm.base import DecisionRequest, Fact, LLMClient, Option
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.tracing.tracer import EventKind, Tracer

ESCALATE = "escalate"

INSTRUCTIONS = (
    "Find the root cause of the alert. Each step, either run one diagnostic tool on a "
    "service or submit the final root cause (service and fault type). Tools cost budget."
)


class LoopState(TypedDict, total=False):
    facts: Annotated[list[Fact], operator.add]
    executed: Annotated[list[str], operator.add]
    choice: str
    hallucinated: int


class LoopAgent:
    """ReAct baseline implemented as a two-node LangGraph cycle."""

    def __init__(self, max_steps: int = 40) -> None:
        self._max_steps = max_steps

    @property
    def name(self) -> str:
        return "loop"

    def investigate(
        self, env: IncidentEnvironment, llm: LLMClient, tracer: Tracer
    ) -> Investigation:
        catalog = env.catalog()
        alert = env.alert()
        catalog_fact = Fact(
            "Service catalog: " + "; ".join(f"{a} calls {b}" for a, b in catalog.dependencies),
            frozenset(catalog.names),
            0.05,
        )
        initial = [
            finding_to_fact(alert.findings[0]),
            catalog_fact,
            *(finding_to_fact(f) for f in alert.findings[1:]),
        ]

        def reason(state: LoopState) -> dict[str, Any]:
            tracer.emit(EventKind.NODE, "reason")
            options = _options(catalog.names, set(state.get("executed", [])), env.remaining > 0)
            request = DecisionRequest(
                "next_action", INSTRUCTIONS, tuple(state["facts"]), tuple(options)
            )
            # A plain tool loop has no rule-based fallback: if the model is capped, escalate.
            return {"choice": decide_or_fallback(llm, request, ESCALATE, tracer)}

        def act(state: LoopState) -> dict[str, Any]:
            tracer.emit(EventKind.NODE, "act")
            tool_name, _, service = state["choice"].partition(":")
            call = call_tool(env, tracer, "act", ToolName(tool_name), service)
            if call is None:
                return {
                    "executed": [state["choice"]],
                    "hallucinated": state.get("hallucinated", 0) + 1,
                }
            facts = [finding_to_fact(f) for f in call.findings]
            if not facts:
                facts = [Fact(f"{tool_name}({service}): no results", frozenset({service}), 0.0)]
            return {"facts": facts, "executed": [state["choice"]]}

        def route(state: LoopState) -> str:
            done = state["choice"].startswith("submit:") or state["choice"] == ESCALATE
            return END if done else "act"

        builder = StateGraph(LoopState)
        builder.add_node("reason", reason)
        builder.add_node("act", act)
        builder.add_edge(START, "reason")
        builder.add_conditional_edges("reason", route, {"act": "act", END: END})
        builder.add_edge("act", "reason")
        graph = builder.compile(checkpointer=new_checkpointer())

        final = invoke_durably(
            graph,
            {"facts": initial, "executed": [], "hallucinated": 0},
            f"{self.name}:{env.scenario.scenario_id}",
            tracer,
            recursion_limit=2 * self._max_steps + 2,
        )
        diagnosis = _parse_submit(final["choice"])
        tracer.emit(EventKind.DIAGNOSIS, "submit", diagnosis=str(diagnosis))
        return Investigation(
            agent=self.name,
            incident_id=env.scenario.scenario_id,
            symptom=alert.service,
            status=Status.COMMITTED if diagnosis else Status.ESCALATED,
            diagnosis=diagnosis,
            ranked=(diagnosis,) if diagnosis else (),
            confidence=None,
            path=Path.LOOP,
            hallucinated_calls=final.get("hallucinated", 0),
            budget_exhausted=env.remaining <= 0,
        )

    def learn(self, investigation: Investigation, verified: bool, at: int) -> None:
        """A stateless tool loop has nowhere to put what it learned."""


def _options(services: tuple[str, ...], executed: set[str], can_probe: bool) -> list[Option]:
    options: list[Option] = []
    if can_probe:
        for svc in services:
            for tool in SERVICE_PROBES:
                oid = f"{tool.value}:{svc}"
                if oid not in executed:
                    options.append(
                        Option(oid, f"{tool.value}({svc})", svc, frozenset({svc, tool.value}))
                    )
    for svc in services:
        for fault in FaultType:
            options.append(
                Option(
                    f"submit:{fault.value}@{svc}",
                    f"submit root cause {fault.value} on {svc}",
                    svc,
                    frozenset({svc, fault.value}),
                    commits=True,
                )
            )
    return options


def _parse_submit(choice: str) -> Diagnosis | None:
    if not choice.startswith("submit:"):
        return None
    fault, _, service = choice.removeprefix("submit:").partition("@")
    return Diagnosis(service, FaultType(fault))
