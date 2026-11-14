"""Turn execution traces and the CCG into views: Mermaid (for GitHub) and vis-network JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphloop.domain.ccg import EdgeKind, NetworkXCausalContextGraph, NodeKind
from graphloop.eval.runner import BenchmarkResult, RunRecord
from graphloop.tracing.tracer import EventKind, Tracer


@dataclass(frozen=True)
class Step:
    node: str
    detail: str


def steps(tracer: Tracer, limit: int = 18) -> list[Step]:
    """Collapse a trace into readable steps: each node with what it did."""
    out: list[Step] = []
    for e in tracer.events:
        if e.kind is EventKind.NODE:
            out.append(Step(e.name, ""))
        elif out and e.kind is EventKind.TOOL:
            out[-1] = Step(out[-1].node, f"{e.name}({e.data.get('service')})")
        elif out and e.kind is EventKind.LLM:
            detail = f"LLM: {e.data.get('choice_text', e.data.get('choice'))}"
            out[-1] = Step(out[-1].node, out[-1].detail or detail)
        elif (
            out
            and e.kind is EventKind.NOTE
            and e.name
            in {"recipe_hit", "escalate", "counterfactual_reject", "accept_on_causal_chain"}
        ):
            out[-1] = Step(out[-1].node, e.name.replace("_", " "))
        elif out and e.kind is EventKind.DIAGNOSIS:
            out[-1] = Step(out[-1].node, f"=> {e.data.get('diagnosis')}")
    if len(out) > limit:
        out = [*out[: limit - 2], Step("...", f"{len(out) - limit + 1} more steps"), out[-1]]
    return out


def mermaid(tracer: Tracer, title: str) -> str:
    lines = ["flowchart TB", f"  %% {title}"]
    for i, s in enumerate(steps(tracer)):
        label = s.node if not s.detail else f"{s.node}<br/><small>{_clean(s.detail)}</small>"
        lines.append(f'  s{i}["{label}"]')
        if i:
            lines.append(f"  s{i - 1} --> s{i}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    return text.replace('"', "'").replace("[", "(").replace("]", ")")[:60]


def showcase(result: BenchmarkResult, focus: str = "graph+memory") -> list[str]:
    """Incidents where the loop baseline failed and ``focus`` resolved: the interesting ones."""
    loop = {r.scenario_id: r for r in result.for_agent("loop") if r.repeat == 0}
    good = [r for r in result.for_agent(focus) if r.repeat == 0 and r.resolved]
    picks = [
        r.scenario_id for r in good if r.scenario_id in loop and not loop[r.scenario_id].resolved
    ]
    picks += [r.scenario_id for r in good if r.scenario_id not in picks]
    return picks[:8]


def by_id(records: list[RunRecord]) -> dict[str, RunRecord]:
    return {r.scenario_id: r for r in records if r.repeat == 0}


_GROUP = {
    NodeKind.SERVICE: "service",
    NodeKind.FAULT_MODE: "fault",
    NodeKind.CHANGE: "change",
    NodeKind.INCIDENT: "incident",
    NodeKind.EVIDENCE: "evidence",
}


def incident_network(
    ccg: NetworkXCausalContextGraph, incident_id: str, tracer: Tracer | None
) -> dict[str, Any]:
    """vis-network data for one incident, with the agent's probe order overlaid."""
    sub = ccg.incident_subgraph(incident_id)
    order: dict[str, int] = {}
    if tracer is not None:
        for e in tracer.of(EventKind.TOOL):
            order.setdefault(f"svc:{e.data.get('service')}", len(order) + 1)
    nodes = []
    for n, data in sub.nodes(data=True):
        kind = data["kind"]
        label = str(data.get("label", n))
        if kind is NodeKind.EVIDENCE:
            label = label.replace("_", " ")
        if n in order:
            label = f"#{order[n]} {label}"
        nodes.append(
            {
                "id": n,
                "label": label,
                "group": _GROUP[kind],
                "title": str(data.get("text", label)),
                "probed": n in order,
            }
        )
    edges = [
        {"from": u, "to": v, "label": "" if k is EdgeKind.OBSERVED_ON else str(k), "kind": str(k)}
        for u, v, k in sub.edges(keys=True)
    ]
    return {"nodes": nodes, "edges": edges}


def memory_network(ccg: NetworkXCausalContextGraph) -> dict[str, Any]:
    """The learned causal layer: fault modes -> symptoms, weighted by support."""
    nodes: dict[str, dict[str, Any]] = {}
    edges = []
    for link in ccg.causal_links():
        f, s = f"fault:{link.fault}", f"svc:{link.symptom}"
        nodes.setdefault(f, {"id": f, "label": str(link.fault), "group": "fault"})
        nodes.setdefault(s, {"id": s, "label": link.symptom, "group": "service"})
        edges.append(
            {
                "from": f,
                "to": s,
                "label": f"CAUSES x{link.support}",
                "value": link.support,
                "kind": "CAUSES",
                "title": f"{link.provenance.value}, confidence {link.confidence:.2f}",
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}
