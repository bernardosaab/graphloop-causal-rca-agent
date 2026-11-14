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


#: The workflow each agent runs, in display order (the replay highlights the active node).
WORKFLOWS: dict[str, list[str]] = {
    "graph": [
        "triage",
        "hypothesize",
        "recipe",
        "select_probe",
        "run_probe",
        "counterfactual",
        "diagnose",
        "critique",
        "finalize",
        "remediation",
    ],
    "loop": ["reason", "act", "remediation"],
}

#: Narration for steps that complete without emitting a more specific event.
_SILENT = {
    "triage": "alert received; change feed and alert evidence written to the graph",
    "hypothesize": "hypotheses scoped to the blast radius",
    "recipe": "no corroborated recipe matches; continuing with the agentic loop",
    "select_probe": "one candidate probe left: chosen without the model",
    "run_probe": "no probe left to run",
    "counterfactual": "leading hypothesis is consistent with the graph",
    "critique": "accepted: the failure signature was observed",
    "finalize": "investigation closed",
    "reason": "model consulted",
    "act": "tool executed",
}

_SIGNATURES = {"code_exception", "config_error", "oom_killed", "store_unavailable"}


def _status(signals: list[str]) -> str:
    """How a probe result should be drawn: what this observation says about the service."""
    if any(s in _SIGNATURES for s in signals):
        return "signature"
    if "upstream_timeout" in signals:
        return "victim"
    if signals and all(s == "healthy" for s in signals):
        return "healthy"
    if any(s in {"error_spike", "latency_spike", "cpu_saturation"} for s in signals):
        return "anomalous"
    return "inspected"


def frames(tracer: Tracer) -> list[dict[str, Any]]:
    """Replayable snapshots: one per meaningful event, each carrying the cumulative state."""
    out: list[dict[str, Any]] = []
    node = ""
    belief: list[list[Any]] = []
    probed: dict[str, str] = {}
    tools = llm = tokens = 0
    narrated = True

    def snapshot(kind: str, text: str, focus: str | None = None) -> dict[str, Any]:
        return {
            "node": node,
            "kind": kind,
            "text": text,
            "focus": focus,
            "probed": dict(probed),
            "belief": belief,
            "tools": tools,
            "llm": llm,
            "tokens": tokens,
        }

    for e in tracer.events:
        kind, focus, text = e.kind.value, None, ""
        if e.kind is EventKind.NODE:
            if not narrated:  # the previous step finished silently: still show it
                out.append(snapshot("node", _SILENT.get(node, f"{node} completed")))
            node, narrated = e.name, False
            continue  # the next event carries the narration for this node
        if e.kind is EventKind.TOOL:
            tools += 1
            focus = str(e.data.get("service"))
            status = (
                "hallucinated"
                if e.data.get("hallucinated")
                else _status(list(e.data.get("signals", [])))
            )
            # Stronger evidence wins when a service is probed more than once.
            rank = ["inspected", "healthy", "anomalous", "victim", "signature", "hallucinated"]
            if rank.index(status) >= rank.index(probed.get(focus, "inspected")):
                probed[focus] = status
            text = f"{e.name}({focus}): " + "; ".join(e.data.get("findings", []))
        elif e.kind is EventKind.LLM:
            llm += 1
            tokens += int(e.data["prompt_tokens"]) + int(e.data["completion_tokens"])
            text = (
                f"model chose {e.data.get('choice_text')} (from {e.data.get('n_options')} options)"
            )
        elif e.kind is EventKind.BELIEF:
            belief = [list(pair) for pair in e.data.get("top", [])]
            text = "belief updated: " + ", ".join(f"{h} {p:.0%}" for h, p in belief)
        elif e.kind is EventKind.DEGRADED:
            text = f"DEGRADED: {e.data.get('reason')}; rule-based fallback"
        elif e.kind is EventKind.NOTE:
            text = e.name.replace("_", " ") + _note_detail(e.data)
        elif e.kind is EventKind.DIAGNOSIS:
            text = f"diagnosis: {e.data.get('diagnosis')}"
        elif e.kind is EventKind.GOVERNANCE:
            node = "remediation"
            text = e.name.replace("_", " ") + _note_detail(e.data)
        narrated = True
        out.append(snapshot(kind, text, focus))
    if not narrated:
        out.append(snapshot("node", _SILENT.get(node, f"{node} completed")))
    return out


def _note_detail(data: dict[str, Any]) -> str:
    keys = (
        "reason",
        "diagnosis",
        "recipe",
        "quote",
        "signer",
        "tier",
        "resolved",
        "probe",
        "services",
    )
    parts = [f"{k}={data[k]}" for k in keys if k in data]
    return f" ({', '.join(parts)})" if parts else ""


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
