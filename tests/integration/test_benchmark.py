"""Benchmark reproducibility and an eval *release gate*: regressions fail the build."""

from __future__ import annotations

from dataclasses import replace

import pytest

from graphloop.eval.runner import BenchmarkConfig, BenchmarkResult, paired_outcomes, run_benchmark
from graphloop.eval.stats import mcnemar_exact, pass_hat_k
from graphloop.eval.taxonomy import FailureMode


@pytest.fixture(scope="module")
def result() -> BenchmarkResult:
    return run_benchmark(BenchmarkConfig(scenarios=60, seed=7))


def _strip(r: object) -> object:
    return replace(r, wall_ms=0.0)  # type: ignore[type-var]


def test_benchmark_is_reproducible() -> None:
    cfg = BenchmarkConfig(scenarios=8, seed=3)
    a, b = run_benchmark(cfg), run_benchmark(cfg)
    assert [_strip(r) for r in a.records] == [_strip(r) for r in b.records]


def test_release_gate_graph_agent_never_writes_a_wrong_fix(result: BenchmarkResult) -> None:
    for agent in ("graph", "graph+memory"):
        wrong_writes = [
            r for r in result.for_agent(agent) if r.status == "committed" and not r.correct
        ]
        assert wrong_writes == [], agent


def test_release_gate_beats_the_loop_baseline(result: BenchmarkResult) -> None:
    loop, graph = paired_outcomes(result, "loop", "graph+memory")
    test = mcnemar_exact(loop, graph)
    assert sum(graph) > sum(loop)
    assert test.p_value < 0.05
    assert sum(graph) / len(graph) >= 0.70


def test_memory_reduces_model_usage(result: BenchmarkResult) -> None:
    def llm_calls(agent: str) -> float:
        rs = result.for_agent(agent)
        return sum(r.llm_calls for r in rs) / len(rs)

    assert llm_calls("graph+memory") < llm_calls("graph") < llm_calls("loop") + 1
    assert any(r.path == "compiled" for r in result.for_agent("graph+memory"))


def test_every_failure_is_clustered_and_audit_chain_holds(result: BenchmarkResult) -> None:
    known = {m.value for m in FailureMode}
    assert all(r.failure in known for r in result.records if not r.correct)
    assert result.audit.verify_chain()


def test_pass_hat_k_with_repeats() -> None:
    res = run_benchmark(BenchmarkConfig(scenarios=6, seed=5, repeats=2, agents=("graph",)))
    by_case: dict[int, list[bool]] = {}
    for r in res.records:
        by_case.setdefault(r.index, []).append(r.resolved)
    assert all(len(v) == 2 for v in by_case.values())
    assert 0.0 <= pass_hat_k(list(by_case.values())) <= 1.0
