from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphloop.cli.app import app
from graphloop.eval.runner import BenchmarkConfig, BenchmarkResult, run_benchmark
from graphloop.report.summary import learning_curve, paired_test, summaries, summarise
from graphloop.report.svg import Bar, bar_chart, line_chart
from graphloop.report.trace_views import incident_network, memory_network, mermaid, showcase
from graphloop.report.writer import write_report


@pytest.fixture(scope="module")
def small() -> BenchmarkResult:
    return run_benchmark(BenchmarkConfig(scenarios=24, seed=7, repeats=2))


def test_summaries_are_consistent(small: BenchmarkResult) -> None:
    for s in summaries(small):
        assert s.n == 48
        assert s.escalated + round(s.coverage * s.n) == s.n
        assert 0.0 <= s.resolved.low <= s.resolved.rate <= s.resolved.high <= 1.0
        assert s.pass_hat_k is not None
        assert sum(s.failures.values()) == s.n - sum(
            1 for r in small.for_agent(s.agent) if r.correct
        )
    with pytest.raises(ValueError, match="no records"):
        summarise([])
    assert paired_test(small, "loop", "missing") is None
    curve = learning_curve(small.for_agent("graph+memory"), buckets=4)
    assert [p.start for p in curve] == [1, 7, 13, 19]


def test_views(small: BenchmarkResult) -> None:
    picks = showcase(small)
    assert picks
    tracer = small.traces[("graph+memory", picks[0])]
    assert mermaid(tracer, "x").startswith("flowchart TB")
    assert small.memory is not None
    net = incident_network(small.memory, picks[0], tracer)
    incidents = [n for n in net["nodes"] if n["group"] == "incident"]
    assert len(incidents) == 1  # never leaks other incidents through shared fault modes
    assert any(n["probed"] for n in net["nodes"])
    assert memory_network(small.memory)["edges"]


def test_svg_charts_are_well_formed() -> None:
    svg = bar_chart("t", [Bar("a", 0.5, 0.4, 0.6), Bar("b", 0.9)], unit="%", max_value=1.0)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "50%" in svg
    line = line_chart("t", ["1", "2"], {"loop": [1.0, 2.0], "custom": [3.0, 1500.0]})
    assert "polyline" in line
    assert "1.5k" in bar_chart("t", [Bar("a", 1500.0)])


def test_write_report(small: BenchmarkResult, tmp_path: Path) -> None:
    paths = write_report(small, tmp_path)
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "Paired significance" in md
    assert "```mermaid" in md
    html = paths["html"].read_text(encoding="utf-8")
    assert "vis-network@9.1.9" in html
    assert "Inside one incident" in html
    data = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert {d["agent"] for d in data} == {"loop", "graph", "graph+memory"}
    assert (tmp_path / "img" / "resolved.svg").exists()
    assert paths["records"].read_text(encoding="utf-8").startswith("agent,")


runner = CliRunner()


def test_cli_run_and_bench(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--incident", "3", "--agent", "graph+memory"])
    assert result.exit_code == 0, result.output
    assert "diagnosis" in result.output
    result = runner.invoke(app, ["run", "--agent", "loop", "--approve", "prompt"], input="n\n")
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["bench", "--scenarios", "6", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.html").exists()
