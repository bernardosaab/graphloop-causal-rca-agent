"""The harness around the model: run caps, a single model gateway, cost quote and attribution.

The model proposes; the harness decides what actually happens. Caps live here, in code,
where no prompt or model output can waive them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from graphloop.llm.base import Decision, DecisionRequest, LLMClient, TokenUsage
from graphloop.tracing.tracer import EventKind, Tracer


@dataclass(frozen=True)
class RunPolicy:
    """Hard per-run limits. Exceeding one ends the model's part of the run."""

    max_tool_calls: int = 12
    max_llm_calls: int = 20
    max_spend_usd: float = 0.01
    max_wall_seconds: float = 300.0

    def __post_init__(self) -> None:
        if min(self.max_tool_calls, self.max_llm_calls) < 1:
            raise ValueError("call caps must be positive")
        if self.max_spend_usd <= 0 or self.max_wall_seconds <= 0:
            raise ValueError("spend and wall-time caps must be positive")


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Local open-weight models have zero marginal price."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: TokenUsage) -> float:
        return (
            usage.prompt_tokens * self.input_per_mtok
            + usage.completion_tokens * self.output_per_mtok
        ) / 1_000_000


#: Reference price used for *comparable* cost reporting across agents (a small hosted model
#: class). Real local runs are free at the margin; we still report this equivalent cost.
REFERENCE_PRICING = Pricing(input_per_mtok=0.15, output_per_mtok=0.60)


@dataclass(frozen=True)
class CostQuote:
    expected_usd: float
    cap_usd: float


def quote_run(
    policy: RunPolicy, pricing: Pricing, expected_llm_calls: float, expected_prompt_tokens: float
) -> CostQuote:
    """Quote before running: expected spend from a profile, never more than the cap."""
    per_call = pricing.cost(TokenUsage(int(expected_prompt_tokens), 40))
    return CostQuote(min(policy.max_spend_usd, expected_llm_calls * per_call), policy.max_spend_usd)


class CapExceededError(RuntimeError):
    def __init__(self, cap: str, detail: str) -> None:
        super().__init__(f"{cap} cap reached: {detail}")
        self.cap = cap


class ModelGateway:
    """Every model call goes through here: caps, cost, latency and trace in one place."""

    def __init__(
        self,
        inner: LLMClient,
        tracer: Tracer,
        policy: RunPolicy | None = None,
        pricing: Pricing = REFERENCE_PRICING,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._tracer = tracer
        self._policy = policy or RunPolicy()
        self._pricing = pricing
        self._clock = clock
        self._started = clock()
        self.calls = 0
        self.spent_usd = 0.0

    @property
    def name(self) -> str:
        return self._inner.name

    def decide(self, request: DecisionRequest) -> Decision:
        self._enforce_caps()
        t0 = self._clock()
        decision = self._inner.decide(request)
        latency_ms = (self._clock() - t0) * 1000
        cost = self._pricing.cost(decision.usage)
        self.calls += 1
        self.spent_usd += cost
        self._tracer.emit(
            EventKind.LLM,
            request.purpose,
            model=self._inner.name,
            choice=decision.option_id,
            choice_text=request.option(decision.option_id).text,
            n_options=len(request.options),
            n_facts=len(request.facts),
            prompt_tokens=decision.usage.prompt_tokens,
            completion_tokens=decision.usage.completion_tokens,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
            valid=decision.valid,
        )
        return decision

    def _enforce_caps(self) -> None:
        p = self._policy
        if self.calls >= p.max_llm_calls:
            raise CapExceededError("llm_calls", f"{self.calls}/{p.max_llm_calls}")
        if self.spent_usd >= p.max_spend_usd:
            raise CapExceededError("spend", f"${self.spent_usd:.4f}/${p.max_spend_usd}")
        elapsed = self._clock() - self._started
        if elapsed >= p.max_wall_seconds:
            raise CapExceededError("wall_time", f"{elapsed:.1f}s/{p.max_wall_seconds}s")
