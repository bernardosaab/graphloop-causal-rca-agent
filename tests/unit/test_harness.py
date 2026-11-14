from __future__ import annotations

import pytest

from graphloop.harness.gateway import (
    REFERENCE_PRICING,
    CapExceededError,
    ModelGateway,
    Pricing,
    RunPolicy,
    quote_run,
)
from graphloop.llm import SimulatedLLM
from graphloop.llm.base import DecisionRequest, Option, TokenUsage
from graphloop.tracing.tracer import EventKind, Tracer

REQ = DecisionRequest("p", "i", (), (Option("x", "x", "x", frozenset({"x"})),))


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        RunPolicy(max_llm_calls=0)
    with pytest.raises(ValueError, match="positive"):
        RunPolicy(max_spend_usd=0)


def test_gateway_records_cost_and_trace() -> None:
    tracer = Tracer()
    gw = ModelGateway(SimulatedLLM(), tracer)
    gw.decide(REQ)
    event = tracer.of(EventKind.LLM)[0]
    assert event.data["cost_usd"] > 0
    assert gw.spent_usd == pytest.approx(event.data["cost_usd"])
    assert tracer.summary().llm_calls == 1
    assert gw.name.startswith("simulated")


def test_call_cap_cannot_be_waived() -> None:
    gw = ModelGateway(SimulatedLLM(), Tracer(), RunPolicy(max_llm_calls=2))
    gw.decide(REQ)
    gw.decide(REQ)
    with pytest.raises(CapExceededError) as exc:
        gw.decide(REQ)
    assert exc.value.cap == "llm_calls"


def test_spend_and_wall_time_caps() -> None:
    expensive = Pricing(1e6, 1e6)
    gw = ModelGateway(SimulatedLLM(), Tracer(), RunPolicy(max_spend_usd=0.5), pricing=expensive)
    gw.decide(REQ)
    with pytest.raises(CapExceededError, match="spend"):
        gw.decide(REQ)
    clock = FakeClock()
    gw = ModelGateway(SimulatedLLM(), Tracer(), RunPolicy(max_wall_seconds=5), clock=clock)
    clock.t = 6
    with pytest.raises(CapExceededError, match="wall_time"):
        gw.decide(REQ)


def test_quote_before_run_never_exceeds_cap() -> None:
    policy = RunPolicy(max_spend_usd=0.001)
    q = quote_run(policy, REFERENCE_PRICING, expected_llm_calls=3, expected_prompt_tokens=1000)
    assert 0 < q.expected_usd <= q.cap_usd
    huge = quote_run(policy, REFERENCE_PRICING, 1e6, 1e6)
    assert huge.expected_usd == policy.max_spend_usd
    assert REFERENCE_PRICING.cost(TokenUsage(1_000_000, 0)) == pytest.approx(0.15)
