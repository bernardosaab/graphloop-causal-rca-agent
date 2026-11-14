"""Contract and helpers shared by every root-cause agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from graphloop.domain.vocabulary import (
    SIGNATURE_OF,
    Diagnosis,
    FaultType,
    Finding,
    Signal,
    ToolName,
)
from graphloop.harness.gateway import CapExceededError
from graphloop.llm.base import DecisionRequest, Fact, LLMClient
from graphloop.llm.ollama import LLMUnavailableError
from graphloop.simulator.environment import (
    BudgetExhaustedError,
    IncidentEnvironment,
    ToolCall,
    UnknownServiceError,
)
from graphloop.tracing.tracer import EventKind, Tracer

_FAULT_HINT: dict[Signal, FaultType] = {
    **SIGNATURE_OF,
    Signal.RECENT_DEPLOY: FaultType.BAD_DEPLOY,
    Signal.RECENT_CONFIG_CHANGE: FaultType.CONFIG_DRIFT,
    Signal.CPU_SATURATION: FaultType.RESOURCE_EXHAUSTION,
}


class Status(StrEnum):
    COMMITTED = "committed"  # the agent stands behind a diagnosis
    ESCALATED = "escalated"  # the agent declined to guess and handed over to a person


class Path(StrEnum):
    COMPILED = "compiled"  # deterministic recipe, no model on the path
    AGENTIC = "agentic"  # graph workflow with model-assisted judgement
    LOOP = "loop"  # model-driven tool loop


@dataclass(frozen=True)
class Investigation:
    """The outcome of one agent run on one incident."""

    agent: str
    incident_id: str
    symptom: str
    status: Status
    diagnosis: Diagnosis | None  # committed diagnosis, or the suggestion when escalated
    ranked: tuple[Diagnosis, ...]
    confidence: float | None
    path: Path
    hallucinated_calls: int = 0
    budget_exhausted: bool = False
    evidence_ids: tuple[str, ...] = field(default=())

    @property
    def committed(self) -> Diagnosis | None:
        return self.diagnosis if self.status is Status.COMMITTED else None


class RCAAgent(Protocol):
    @property
    def name(self) -> str: ...

    def investigate(
        self, env: IncidentEnvironment, llm: LLMClient, tracer: Tracer
    ) -> Investigation: ...

    def learn(self, investigation: Investigation, verified: bool, at: int) -> None:
        """Called by the harness after the write was scored against the system of record."""
        ...


def finding_to_fact(finding: Finding) -> Fact:
    """Translate an observation into LLM context. Used identically by every agent."""
    refs = {finding.service}
    if finding.target:
        refs.add(finding.target)
    if (hint := _FAULT_HINT.get(finding.signal)) is not None:
        refs.add(hint.value)
    exonerates = (
        frozenset({finding.service}) if finding.signal is Signal.UPSTREAM_TIMEOUT else frozenset()
    )
    return Fact(finding.text, frozenset(refs), finding.salience, exonerates)


def probe_key(tool: ToolName, service: str) -> str:
    return f"{tool.value}:{service}"


def parse_probe(key: str) -> tuple[ToolName, str]:
    tool, _, service = key.partition(":")
    return ToolName(tool), service


def decide_or_fallback(
    llm: LLMClient, request: DecisionRequest, fallback_id: str, tracer: Tracer
) -> str:
    """Ask the model; if a cap or outage stops it, take the rule-based choice *loudly*."""
    try:
        return llm.decide(request).option_id
    except (CapExceededError, LLMUnavailableError) as exc:
        tracer.emit(EventKind.DEGRADED, request.purpose, reason=str(exc), fallback=fallback_id)
        return fallback_id


def call_tool(
    env: IncidentEnvironment, tracer: Tracer, node: str, tool: ToolName, service: str
) -> ToolCall | None:
    """Execute a probe, trace it, and turn hallucinations or exhausted budget into ``None``."""
    try:
        call = env.call(tool, service)
    except UnknownServiceError as exc:
        tracer.emit(
            EventKind.TOOL,
            tool.value,
            node=node,
            service=service,
            error=str(exc),
            hallucinated=True,
        )
        return None
    except BudgetExhaustedError as exc:
        tracer.emit(EventKind.NOTE, "budget_exhausted", node=node, error=str(exc))
        return None
    tracer.emit(
        EventKind.TOOL,
        tool.value,
        node=node,
        service=service,
        findings=[f.text for f in call.findings],
        signals=[f.signal.value for f in call.findings],
    )
    return call
