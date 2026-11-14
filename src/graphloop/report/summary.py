"""Aggregate run records into the numbers people actually decide with."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from graphloop.eval.runner import BenchmarkResult, RunRecord, paired_outcomes
from graphloop.eval.stats import McNemar, Proportion, mcnemar_exact, pass_hat_k, percentile, wilson


@dataclass(frozen=True)
class AgentSummary:
    agent: str
    n: int
    resolved: Proportion  # verified by read-back: the headline "useful rate"
    precision: float  # of committed diagnoses, how many were right
    coverage: float  # share of incidents where the agent committed
    escalated: int
    wrong_writes: int  # remediations executed for a wrong diagnosis
    tool_calls: float
    llm_calls: float
    tokens: float
    cost_usd: float
    cost_per_resolved: float | None
    p50_ms: float
    p95_ms: float
    compiled_share: float
    retrieval_at_3: float | None
    degraded_runs: int
    failures: dict[str, int]
    pass_hat_k: float | None


@dataclass(frozen=True)
class CurvePoint:
    bucket: int
    start: int
    end: int
    resolved: float
    llm_calls: float
    tool_calls: float


def summarise(records: Sequence[RunRecord], repeats: int = 1) -> AgentSummary:
    n = len(records)
    if n == 0:
        raise ValueError("no records to summarise")
    committed = [r for r in records if r.status == "committed"]
    resolved = sum(r.resolved for r in records)
    cost = sum(r.cost_usd for r in records)
    retrieval = [r.retrieval_hit for r in records if r.retrieval_hit is not None]
    walls = [r.wall_ms for r in records]
    by_case: dict[int, list[bool]] = {}
    for r in records:
        by_case.setdefault(r.index, []).append(r.resolved)
    return AgentSummary(
        agent=records[0].agent,
        n=n,
        resolved=wilson(resolved, n),
        precision=(sum(r.correct for r in committed) / len(committed)) if committed else 0.0,
        coverage=len(committed) / n,
        escalated=n - len(committed),
        wrong_writes=sum(1 for r in committed if not r.correct),
        tool_calls=sum(r.tool_calls for r in records) / n,
        llm_calls=sum(r.llm_calls for r in records) / n,
        tokens=sum(r.prompt_tokens + r.completion_tokens for r in records) / n,
        cost_usd=cost,
        cost_per_resolved=(cost / resolved) if resolved else None,
        p50_ms=percentile(walls, 50),
        p95_ms=percentile(walls, 95),
        compiled_share=sum(r.path == "compiled" for r in records) / n,
        retrieval_at_3=(sum(retrieval) / len(retrieval)) if retrieval else None,
        degraded_runs=sum(r.degraded for r in records),
        failures=dict(Counter(r.failure for r in records if r.failure).most_common()),
        pass_hat_k=pass_hat_k(list(by_case.values())) if repeats > 1 else None,
    )


def summaries(result: BenchmarkResult) -> list[AgentSummary]:
    return [summarise(result.for_agent(a), result.config.repeats) for a in result.config.agents]


def learning_curve(records: Sequence[RunRecord], buckets: int = 5) -> list[CurvePoint]:
    """Outcomes over the incident stream (first repeat), in equal-sized buckets."""
    stream = sorted((r for r in records if r.repeat == 0), key=lambda r: r.index)
    if not stream:
        return []
    size = max(1, -(-len(stream) // buckets))
    points = []
    for b, start in enumerate(range(0, len(stream), size)):
        chunk = stream[start : start + size]
        points.append(
            CurvePoint(
                bucket=b,
                start=chunk[0].index + 1,
                end=chunk[-1].index + 1,
                resolved=sum(r.resolved for r in chunk) / len(chunk),
                llm_calls=sum(r.llm_calls for r in chunk) / len(chunk),
                tool_calls=sum(r.tool_calls for r in chunk) / len(chunk),
            )
        )
    return points


def paired_test(result: BenchmarkResult, a: str, b: str) -> McNemar | None:
    if a not in result.config.agents or b not in result.config.agents:
        return None
    xa, xb = paired_outcomes(result, a, b)
    return mcnemar_exact(xa, xb)
