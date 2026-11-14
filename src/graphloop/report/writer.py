"""Write benchmark artefacts: RESULTS.md (renders on GitHub), report.html, SVGs, raw data."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from graphloop.eval.runner import BenchmarkResult
from graphloop.report.summary import AgentSummary, learning_curve, paired_test, summaries
from graphloop.report.svg import Bar, bar_chart, colour, line_chart
from graphloop.report.trace_views import (
    WORKFLOWS,
    by_id,
    frames,
    incident_network,
    memory_network,
    mermaid,
    showcase,
)


def write_report(result: BenchmarkResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "img").mkdir(exist_ok=True)
    sums = summaries(result)
    charts = _charts(result, sums)
    paths: dict[str, Path] = {}
    for name, svg in charts.items():
        path = out_dir / "img" / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        paths[name] = path
    paths["records"] = _write_csv(result, out_dir / "runs.csv")
    paths["summary"] = out_dir / "summary.json"
    paths["summary"].write_text(
        json.dumps([_summary_json(s) for s in sums], indent=2), encoding="utf-8"
    )
    paths["audit"] = out_dir / "audit.jsonl"
    result.audit.to_jsonl(paths["audit"])
    paths["markdown"] = out_dir / "RESULTS.md"
    paths["markdown"].write_text(render_markdown(result, sums), encoding="utf-8")
    paths["html"] = out_dir / "report.html"
    paths["html"].write_text(render_html(result, sums, charts), encoding="utf-8")
    return paths


# ---- charts ----------------------------------------------------------------------------------
def _charts(result: BenchmarkResult, sums: list[AgentSummary]) -> dict[str, str]:
    charts = {
        "resolved": bar_chart(
            "Incidents resolved (verified by read-back, 95% CI)",
            [
                Bar(s.agent, s.resolved.rate, s.resolved.low, s.resolved.high, colour(s.agent, i))
                for i, s in enumerate(sums)
            ],
            unit="%",
            max_value=1.0,
        ),
        "wrong_writes": bar_chart(
            "Wrong remediations executed (lower is better)",
            [
                Bar(
                    s.agent,
                    float(s.wrong_writes),
                    colour=colour(s.agent, i),
                    text=str(s.wrong_writes),
                )
                for i, s in enumerate(sums)
            ],
            max_value=max(1.0, max(s.wrong_writes for s in sums) * 1.2),
        ),
        "tokens": bar_chart(
            "Tokens per incident (lower is better)",
            [Bar(s.agent, s.tokens, colour=colour(s.agent, i)) for i, s in enumerate(sums)],
        ),
    }
    curves = {a: learning_curve(result.for_agent(a)) for a in result.config.agents}
    first = next(iter(curves.values()))
    if first:
        labels = [f"{p.start}-{p.end}" for p in first]
        charts["learning_llm"] = line_chart(
            "Model calls per incident over the incident stream",
            labels,
            {a: [p.llm_calls for p in pts] for a, pts in curves.items()},
        )
    return charts


# ---- markdown --------------------------------------------------------------------------------
def render_markdown(result: BenchmarkResult, sums: list[AgentSummary]) -> str:
    cfg = result.config
    test = paired_test(result, "loop", "graph+memory")
    lines = [
        "# Benchmark results",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `graphloop bench` · "
        f"{cfg.scenarios} incidents x {cfg.repeats} repeat(s) · model `{cfg.llm}` · "
        f"seed {cfg.seed} · budget {cfg.policy.max_tool_calls} probes._",
        "",
        "Every agent receives the same incident stream, alert, service catalog, tools, probe "
        "budget and model seed (a paired design). A run counts as **resolved** only when its "
        "remediation passed the fail-closed gate, executed, and the read-back confirmed the "
        "incident was fixed.",
        "",
        "| Agent | Resolved (95% CI) | Precision | Coverage | Wrong writes | Escalated | "
        "Tool calls | LLM calls | Tokens | Cost / resolved | Compiled path | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sums:
        cpr = f"${s.cost_per_resolved:.5f}" if s.cost_per_resolved else "n/a"
        lines.append(
            f"| `{s.agent}` | {s.resolved} | {s.precision:.0%} | {s.coverage:.0%} | "
            f"{s.wrong_writes} | {s.escalated} | {s.tool_calls:.1f} | {s.llm_calls:.1f} | "
            f"{s.tokens:,.0f} | {cpr} | {s.compiled_share:.0%} | {s.p95_ms:.0f} ms |"
        )
    lines += [
        "",
        "![resolved](img/resolved.svg) ![wrong writes](img/wrong_writes.svg)",
        "",
        "![tokens](img/tokens.svg) ![learning](img/learning_llm.svg)",
        "",
    ]
    if test is not None:
        lines += [
            "## Paired significance",
            "",
            f"Exact McNemar test, `loop` vs `graph+memory` on identical incidents: "
            f"{test.b} incidents only the loop resolved, {test.c} only the graph agent "
            f"resolved, **p = {test.p_value:.2g}**.",
            "",
        ]
    lines += ["## Error analysis", "", "| Agent | Failure clusters |", "|---|---|"]
    for s in sums:
        clusters = ", ".join(f"`{k}` x {v}" for k, v in s.failures.items()) or "none"
        lines.append(f"| `{s.agent}` | {clusters} |")
    retrieval = [s for s in sums if s.retrieval_at_3 is not None]
    if retrieval:
        lines += [
            "",
            "**Retrieval vs generation.** Root cause in the graph agent's top-3 "
            "hypotheses (recall@3): "
            + ", ".join(f"`{s.agent}` {s.retrieval_at_3:.0%}" for s in retrieval)
            + ". Misses after a retrieval hit are therefore decision or evidence "
            "problems, not retrieval problems.",
        ]
    lines += _showcase_md(result)
    lines += [
        "",
        "## How to read these numbers",
        "",
        "* Results with `simulated` are produced by a deterministic, agent-agnostic stand-in "
        "for a model (see `docs/adr/0005-simulated-llm-evaluation.md`). They measure the "
        "*architecture* under controlled conditions; they are not claims about any specific "
        "model. Re-run with `--llm ollama:<model>` for a real open-weight model.",
        "* Cost uses a reference price of $0.15 / $0.60 per million input / output tokens "
        "so agents are comparable; local models cost nothing at the margin.",
        "* Escalations are not failures of safety: the agent declined to act without direct "
        "evidence and handed the incident to a person with its best suggestion.",
    ]
    return "\n".join(lines) + "\n"


def _showcase_md(result: BenchmarkResult) -> list[str]:
    picks = showcase(result)
    if not picks or "loop" not in result.config.agents:
        return []
    sid = picks[0]
    loop, graph = by_id(result.for_agent("loop")), by_id(result.for_agent("graph+memory"))
    if sid not in loop or sid not in graph:
        return []
    out = [
        "",
        f"## One incident, two architectures (`{sid}`)",
        "",
        f"Ground truth: `{graph[sid].root_cause}`. The loop agent concluded "
        f"`{loop[sid].diagnosis}`; the graph agent concluded `{graph[sid].diagnosis}`.",
        "",
    ]
    for agent in ("loop", "graph+memory"):
        tracer = result.traces.get((agent, sid))
        if tracer is not None:
            out += [f"**{agent}**", "", "```mermaid", mermaid(tracer, agent), "```", ""]
    return out


# ---- html ------------------------------------------------------------------------------------
def render_html(result: BenchmarkResult, sums: list[AgentSummary], charts: dict[str, str]) -> str:
    env = Environment(
        loader=PackageLoader("graphloop.report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )
    picks = showcase(result)
    incidents: list[dict[str, Any]] = []
    for sid in picks:
        rec = by_id(result.for_agent("graph+memory")).get(sid) if result.memory else None
        tracers = {a: result.traces.get((a, sid)) for a in result.config.agents}
        incidents.append(
            {
                "id": sid,
                "truth": rec.root_cause if rec else "",
                "outcomes": {a: by_id(result.for_agent(a)).get(sid) for a in result.config.agents},
                "frames": {a: frames(t) for a, t in tracers.items() if t is not None},
                "network": (
                    incident_network(result.memory, sid, tracers.get("graph+memory"))
                    if result.memory is not None
                    else None
                ),
            }
        )
    test = paired_test(result, "loop", "graph+memory")
    return env.get_template("report.html.j2").render(
        cfg=result.config,
        sums=sums,
        charts=charts,
        incidents=incidents,
        test=test,
        memory=memory_network(result.memory) if result.memory is not None else None,
        generated=f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        networks_json=json.dumps({i["id"]: i["network"] for i in incidents}),
        memory_json=json.dumps(memory_network(result.memory) if result.memory else None),
        replays_json=json.dumps({i["id"]: i["frames"] for i in incidents}),
        outcomes_json=json.dumps(
            {
                i["id"]: {
                    a: {
                        "resolved": r.resolved,
                        "status": r.status,
                        "diagnosis": r.diagnosis,
                        "truth": r.root_cause,
                        "path": r.path,
                    }
                    for a, r in i["outcomes"].items()
                    if r is not None
                }
                for i in incidents
            }
        ),
        workflows_json=json.dumps(WORKFLOWS),
    )


# ---- raw data --------------------------------------------------------------------------------
def _write_csv(result: BenchmarkResult, path: Path) -> Path:
    names = [f.name for f in fields(result.records[0])] if result.records else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for r in result.records:
            writer.writerow(asdict(r))
    return path


def _summary_json(s: AgentSummary) -> dict[str, object]:
    data = asdict(s)
    data["resolved"] = {
        "rate": s.resolved.rate,
        "low": s.resolved.low,
        "high": s.resolved.high,
        "n": s.resolved.n,
    }
    return data
