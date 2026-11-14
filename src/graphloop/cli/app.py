"""Command-line interface: ``graphloop run`` (one incident) and ``graphloop bench``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from graphloop.eval.runner import AGENT_NAMES, BenchmarkConfig, make_agent, run_benchmark
from graphloop.governance.audit import DecisionLog
from graphloop.governance.policy import Approval, ApprovalRequest, Approver, SimulatedOnCall
from graphloop.governance.workflow import RemediationWorkflow
from graphloop.harness.gateway import REFERENCE_PRICING, ModelGateway, RunPolicy, quote_run
from graphloop.llm import LLMUnavailableError, build_llm
from graphloop.report.summary import summaries
from graphloop.report.trace_views import steps
from graphloop.report.writer import write_report
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.simulator.generator import ScenarioGenerator, generate_topology, hotspot_propensity
from graphloop.tracing.tracer import EventKind, Tracer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Causal Context Graph agents vs. loop agents for incident RCA.",
)
console = Console()

#: Expected model usage per agent, from the reference benchmark (used for cost quotes).
_PROFILE = {"loop": (4.0, 2000.0), "graph": (4.0, 300.0), "graph+memory": (3.0, 300.0)}


class ApproveMode(StrEnum):
    AUTO = "auto"
    PROMPT = "prompt"


@dataclass(frozen=True)
class TerminalApprover:
    """A real person signs high-risk writes from the terminal."""

    name: str

    def review(self, request: ApprovalRequest) -> Approval:
        console.print(
            Panel.fit(
                f"[bold]{request.quote}[/]\ntier: {request.tier.value}\n"
                f"rationale: {request.rationale}\nfingerprint: {request.fingerprint[:16]}...",
                title="Approval required",
                border_style="yellow",
            )
        )
        ok = typer.confirm(f"Sign as {self.name}?", default=False)
        return Approval(ok, self.name, request.fingerprint, "terminal review")


@app.command()
def run(
    incident: Annotated[int, typer.Option(help="Incident seed within the stream.")] = 0,
    agent: Annotated[str, typer.Option(help=f"One of {', '.join(AGENT_NAMES)}.")] = "graph",
    llm: Annotated[str, typer.Option(help="'simulated' or 'ollama[:model]'.")] = "simulated",
    seed: Annotated[int, typer.Option(help="Topology seed.")] = 7,
    approve: Annotated[ApproveMode, typer.Option(help="Who signs hard/critical writes.")] = (
        ApproveMode.AUTO
    ),
    signer: Annotated[str, typer.Option(help="Name recorded when you sign.")] = "oncall:you",
    budget: Annotated[int, typer.Option(help="Probe budget.")] = 12,
) -> None:
    """Investigate and remediate one incident, showing the full trajectory."""
    topology = generate_topology(seed)
    scenario = ScenarioGenerator(topology, hotspot_propensity(topology, seed)).generate(
        seed * 10_000 + incident
    )
    rca, _ = make_agent(agent, topology)
    policy = RunPolicy(max_tool_calls=budget)
    calls, tokens = _PROFILE.get(agent, (4.0, 1000.0))
    quote = quote_run(policy, REFERENCE_PRICING, calls, tokens)
    console.print(
        f"[bold]{scenario.scenario_id}[/]: alert on [cyan]{scenario.alert_service}[/] "
        f"· agent [bold]{agent}[/] | model {llm} | quote ~${quote.expected_usd:.5f} "
        f"(cap ${quote.cap_usd})"
    )
    tracer = Tracer()
    env = IncidentEnvironment(scenario, budget=budget)
    gateway = ModelGateway(build_llm(llm, seed=scenario.seed * 31), tracer, policy)
    try:
        inv = rca.investigate(env, gateway, tracer)
    except LLMUnavailableError as exc:  # pragma: no cover - depends on local setup
        console.print(f"[red]{exc}[/]. Is `ollama serve` running?")
        raise typer.Exit(1) from exc
    approver: Approver = (
        TerminalApprover(signer) if approve is ApproveMode.PROMPT else SimulatedOnCall()
    )
    log = DecisionLog()
    outcome = RemediationWorkflow(env, approver, log, tracer).run(inv)

    table = Table(title="Trajectory", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("step")
    table.add_column("detail")
    for i, s in enumerate(steps(tracer, limit=40), 1):
        table.add_row(str(i), s.node, s.detail)
    console.print(table)
    summary = tracer.summary()
    verdict = (
        "[green]resolved[/]"
        if outcome.verified
        else "[yellow]escalated to a person[/]"
        if outcome.proposal is None
        else "[red]not resolved[/]"
    )
    console.print(
        Panel.fit(
            f"diagnosis: [bold]{inv.diagnosis}[/] ({inv.status.value}, path {inv.path.value})\n"
            f"ground truth: {scenario.root_cause}\noutcome: {verdict}"
            + (
                f" | tier {outcome.tier.value} | signed by {outcome.approval.signer}"
                if outcome.tier and outcome.approval
                else ""
            )
            + f"\ncost: {summary.tool_calls} tool calls, {summary.llm_calls} model calls, "
            f"{summary.total_tokens} tokens, ${summary.cost_usd:.5f}"
            + (" | [yellow]DEGRADED[/]" if tracer.of(EventKind.DEGRADED) else ""),
            title="Result",
        )
    )


@app.command()
def bench(
    scenarios: Annotated[int, typer.Option(help="Incidents in the stream.")] = 100,
    seed: Annotated[int, typer.Option(help="Benchmark seed.")] = 7,
    llm: Annotated[str, typer.Option(help="'simulated' or 'ollama[:model]'.")] = "simulated",
    repeats: Annotated[int, typer.Option(help="Repeats per incident (pass^k).")] = 1,
    agents: Annotated[str, typer.Option(help="Comma-separated agents.")] = ",".join(AGENT_NAMES),
    budget: Annotated[int, typer.Option(help="Probe budget per incident.")] = 12,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("results"),
) -> None:
    """Run the paired benchmark and write RESULTS.md, report.html, SVG charts and raw data."""
    config = BenchmarkConfig(
        scenarios=scenarios,
        seed=seed,
        llm=llm,
        repeats=repeats,
        agents=tuple(a.strip() for a in agents.split(",") if a.strip()),
        policy=RunPolicy(max_tool_calls=budget),
    )
    with console.status("running benchmark..."):
        result = run_benchmark(config, progress=console.log)
    paths = write_report(result, out)
    table = Table(title=f"{scenarios} incidents | {llm}")
    for col in (
        "agent",
        "resolved (95% CI)",
        "precision",
        "wrong writes",
        "escalated",
        "LLM calls",
        "tokens",
        "cost/resolved",
    ):
        table.add_column(col)
    for s in summaries(result):
        table.add_row(
            s.agent,
            str(s.resolved),
            f"{s.precision:.0%}",
            str(s.wrong_writes),
            str(s.escalated),
            f"{s.llm_calls:.1f}",
            f"{s.tokens:,.0f}",
            f"${s.cost_per_resolved:.5f}" if s.cost_per_resolved else "n/a",
        )
    console.print(table)
    console.print(f"report: {paths['html']}\nmarkdown: {paths['markdown']}")


if __name__ == "__main__":  # pragma: no cover
    app()
